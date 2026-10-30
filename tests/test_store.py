import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dynmemory import Link, MemoryStore, MemoryType, Relationship

# The schema exactly as version 0.1 created it, to test upgrading real files.
VERSION_0_1_SCHEMA = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN (
        'user_preference', 'strategic_insight',
        'technical_constraint', 'correction', 'task',
        'fact', 'episode'
    )),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    memory_key TEXT,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    importance REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    archived_at TEXT,
    superseded_by TEXT REFERENCES memories(id)
);
CREATE UNIQUE INDEX one_active_memory_per_key
    ON memories(memory_key)
    WHERE memory_key IS NOT NULL AND archived_at IS NULL;
CREATE INDEX memories_active_kind ON memories(archived_at, kind);
CREATE TABLE relationships (
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN (
        'related_to', 'supports', 'contradicts',
        'supersedes', 'depends_on'
    )),
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation),
    CHECK (source_id <> target_id)
);
"""
SECOND = "2026-01-01T00:00:00+00:00"


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memories.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def backdate(self, memory, days, now=None):
        now = now or datetime.now(timezone.utc)
        self.store._set_activity(
            memory.id,
            created_at=(now - timedelta(days=days)).isoformat(timespec="microseconds"),
        )

    def supersedes_edges(self, store=None):
        with (store or self.store)._transaction() as connection:
            return {
                (row["source_id"], row["target_id"])
                for row in connection.execute(
                    "SELECT source_id, target_id FROM relationships "
                    "WHERE relation = 'supersedes'"
                )
            }

    def test_typed_memory_round_trips_and_validates(self):
        memory = self.store.add_memory(
            "The user prefers concise answers.",
            MemoryType.USER_PREFERENCE,
            confidence=0.95,
        )

        self.assertEqual(self.store.get_memory(memory.id), memory)
        self.assertEqual(len(self.store.list_memories()), 1)
        with self.assertRaises(ValueError):
            self.store.add_memory(" ", "fact")
        with self.assertRaises(ValueError):
            self.store.add_memory("A fact", "unknown")
        with self.assertRaises(ValueError):
            self.store.add_memory("A fact", "fact", importance=1.1)

    def test_same_key_replaces_active_memory_and_records_supersession(self):
        earlier = self.store.add_memory(
            "Use the legacy endpoint.",
            MemoryType.TECHNICAL_CONSTRAINT,
            memory_key="api.endpoint",
        )
        later = self.store.add_memory(
            "Use the v2 endpoint.",
            MemoryType.CORRECTION,
            memory_key="api.endpoint",
        )

        archived_earlier = self.store.get_memory(earlier.id)
        self.assertEqual(archived_earlier.superseded_by, later.id)
        self.assertIsNotNone(archived_earlier.archived_at)
        self.assertEqual([item.id for item in self.store.list_memories()], [later.id])

    def test_search_combines_text_and_one_hop_relationships(self):
        preference = self.store.add_memory(
            "Prefer SQLite for local memory storage.",
            MemoryType.USER_PREFERENCE,
        )
        constraint = self.store.add_memory(
            "The agent memory database must stay local.",
            MemoryType.TECHNICAL_CONSTRAINT,
        )
        self.store.relate(preference.id, constraint.id, Relationship.SUPPORTS)

        results = self.store.search("SQLite")

        self.assertEqual({result.memory.id for result in results}, {preference.id, constraint.id})
        by_id = {result.memory.id: result for result in results}
        self.assertEqual(by_id[preference.id].matched_by, "lexical")
        self.assertEqual(by_id[constraint.id].matched_by, "graph")
        self.assertEqual(by_id[constraint.id].memory.access_count, 1)
        self.assertEqual(len(self.store.search("SQLite", kind=MemoryType.FACT)), 0)

    def test_prune_previews_then_archives_only_stale_low_value_memory(self):
        now = datetime.now(timezone.utc)
        stale = self.store.add_memory(
            "A transient detail.",
            MemoryType.EPISODE,
            confidence=0.2,
            importance=0.2,
        )
        protected = self.store.add_memory(
            "Never prune this constraint.",
            MemoryType.TECHNICAL_CONSTRAINT,
            confidence=0.2,
            importance=0.95,
        )
        with self.store._transaction() as connection:
            old_timestamp = (now - timedelta(days=120)).isoformat(timespec="seconds")
            connection.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                (old_timestamp, old_timestamp, stale.id),
            )
            connection.execute(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                (old_timestamp, old_timestamp, protected.id),
            )

        preview = self.store.prune(
            threshold=0.1,
            min_age_days=30,
            dry_run=True,
            as_of=now,
        )
        self.assertEqual([candidate.memory.id for candidate in preview], [stale.id])
        self.assertIsNone(self.store.get_memory(stale.id).archived_at)

        archived = self.store.prune(
            threshold=0.1,
            min_age_days=30,
            dry_run=False,
            as_of=now,
        )
        self.assertIsNotNone(archived[0].memory.archived_at)
        self.assertEqual([memory.id for memory in self.store.list_memories()], [protected.id])

    def test_frequently_accessed_memory_outlives_unused_one(self):
        now = datetime.now(timezone.utc)
        used = self.store.add_memory("Used often alpha.", "fact", importance=0.5, confidence=0.5)
        unused = self.store.add_memory("Never touched beta.", "fact", importance=0.5, confidence=0.5)
        for _ in range(20):
            self.store.search("alpha")
        old_timestamp = (now - timedelta(days=60)).isoformat(timespec="seconds")
        with self.store._transaction() as connection:
            connection.execute(
                "UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?",
                (old_timestamp, old_timestamp, used.id),
            )
            connection.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                (old_timestamp, unused.id),
            )
        scores = {
            c.memory.id: c.score
            for c in self.store.prune(threshold=1.0, min_age_days=30, as_of=now)
        }
        self.assertGreater(scores[used.id], scores[unused.id])

    def test_peek_search_does_not_record_access(self):
        memory = self.store.add_memory("Peekable gamma.", "fact")
        self.store.search("gamma", record_access=False)
        self.assertEqual(self.store.get_memory(memory.id).access_count, 0)
        self.store.search("gamma")
        self.assertEqual(self.store.get_memory(memory.id).access_count, 1)

    def test_history_walks_supersedes_chain(self):
        v1 = self.store.add_memory("v1", "fact", memory_key="k")
        v2 = self.store.add_memory("v2", "fact", memory_key="k")
        v3 = self.store.add_memory("v3", "fact", memory_key="k")
        self.assertEqual([m.id for m in self.store.history(v3.id)], [v3.id, v2.id, v1.id])
        self.assertEqual([m.id for m in self.store.history(v1.id)], [v3.id, v2.id, v1.id])
        self.assertEqual(self.store.history("missing"), [])

    def test_search_links_tell_contradiction_from_support(self):
        claim = self.store.add_memory("Deploy target is staging zeta.", MemoryType.FACT)
        conflict = self.store.add_memory("Production only.", MemoryType.FACT)
        support = self.store.add_memory("Use blue green.", MemoryType.FACT)
        self.store.relate(conflict.id, claim.id, Relationship.CONTRADICTS)
        self.store.relate(support.id, claim.id, Relationship.SUPPORTS)

        by_id = {
            result.memory.id: result
            for result in self.store.search("zeta", record_access=False)
        }

        self.assertEqual(by_id[conflict.id].matched_by, "graph")
        self.assertEqual(
            by_id[conflict.id].links,
            (Link(conflict.id, claim.id, Relationship.CONTRADICTS),),
        )
        self.assertEqual(
            by_id[support.id].links,
            (Link(support.id, claim.id, Relationship.SUPPORTS),),
        )
        self.assertEqual(
            {link.relation for link in by_id[claim.id].links},
            {Relationship.CONTRADICTS, Relationship.SUPPORTS},
        )

    def test_linked_matches_boost_each_other_whichever_way_the_link_points(self):
        first = self.store.add_memory("alpha one", MemoryType.FACT)
        second = self.store.add_memory("alpha two", MemoryType.FACT)
        unlinked = self.store.add_memory("alpha six", MemoryType.FACT)
        self.store.relate(second.id, first.id)

        by_id = {
            result.memory.id: result
            for result in self.store.search("alpha", record_access=False)
        }

        self.assertEqual(by_id[first.id].matched_by, "lexical+graph")
        self.assertEqual(by_id[second.id].matched_by, "lexical+graph")
        self.assertEqual(by_id[unlinked.id].matched_by, "lexical")
        self.assertGreater(by_id[second.id].score, by_id[unlinked.id].score)

    def test_key_rewrite_records_that_the_new_version_supersedes_the_old(self):
        earlier = self.store.add_memory("Use the legacy endpoint.", "fact", memory_key="api")
        later = self.store.add_memory("Use the v2 endpoint.", "fact", memory_key="api")

        self.assertEqual(self.supersedes_edges(), {(later.id, earlier.id)})

    def test_superseding_by_link_matches_superseding_by_key(self):
        old = self.store.add_memory(
            "Use the legacy endpoint.",
            MemoryType.TECHNICAL_CONSTRAINT,
            memory_key="api.endpoint",
        )
        new = self.store.add_memory("Use the v2 endpoint.", MemoryType.CORRECTION)
        self.store.relate(new.id, old.id, Relationship.SUPERSEDES)

        archived = self.store.get_memory(old.id)
        self.assertIsNotNone(archived.archived_at)
        self.assertEqual(archived.superseded_by, new.id)
        self.assertEqual(self.store.get_memory(new.id).memory_key, "api.endpoint")
        self.assertEqual(self.supersedes_edges(), {(new.id, old.id)})

        newest = self.store.add_memory(
            "Use the v3 endpoint.", MemoryType.CORRECTION, memory_key="api.endpoint"
        )
        self.assertEqual(
            [memory.id for memory in self.store.history(old.id)],
            [newest.id, new.id, old.id],
        )

    def test_a_memory_cannot_supersede_a_newer_one(self):
        old = self.store.add_memory("Old plan.", MemoryType.FACT)
        new = self.store.add_memory("New plan.", MemoryType.FACT)
        self.backdate(old, 1)

        with self.assertRaises(ValueError):
            self.store.relate(old.id, new.id, Relationship.SUPERSEDES)
        self.assertIsNone(self.store.get_memory(new.id).archived_at)
        self.assertEqual(self.supersedes_edges(), set())

    def test_rewriting_a_pruned_key_keeps_its_history(self):
        pruned = self.store.add_memory("v1", "fact", memory_key="k", importance=0.1)
        self.backdate(pruned, 300)
        self.store.prune(dry_run=False)
        self.assertIsNotNone(self.store.get_memory(pruned.id).archived_at)

        current = self.store.add_memory("v2", "fact", memory_key="k")

        self.assertEqual(
            [memory.id for memory in self.store.history(current.id)],
            [current.id, pruned.id],
        )
        self.assertEqual(self.store.get_memory(pruned.id).superseded_by, current.id)

    def test_equal_scores_rank_the_newer_memory_first(self):
        older = self.store.add_memory("tie word", MemoryType.FACT)
        newer = self.store.add_memory("tie word", MemoryType.FACT)
        self.backdate(older, 10)
        self.backdate(newer, 1)

        results = self.store.search("tie", record_access=False)

        self.assertEqual(results[0].score, results[1].score)
        self.assertEqual([result.memory.id for result in results], [newer.id, older.id])

    def test_restore_reactivates_a_pruned_memory(self):
        now = datetime.now(timezone.utc)
        memory = self.store.add_memory("Stale detail.", MemoryType.EPISODE, importance=0.1)
        self.backdate(memory, 200, now)
        self.store.prune(dry_run=False, as_of=now)

        restored = self.store.restore(memory.id)

        self.assertIsNone(restored.archived_at)
        self.assertIsNotNone(restored.last_accessed_at)
        self.assertEqual(self.store.prune(as_of=now), [])
        with self.assertRaises(ValueError):
            self.store.restore(memory.id)
        with self.assertRaises(ValueError):
            self.store.restore("missing")

    def test_restore_refuses_superseded_memories_and_taken_keys(self):
        replaced = self.store.add_memory("v1", "fact")
        replacement = self.store.add_memory("v2", "fact")
        self.store.relate(replacement.id, replaced.id, Relationship.SUPERSEDES)
        with self.assertRaisesRegex(ValueError, "superseded by"):
            self.store.restore(replaced.id)

        pruned = self.store.add_memory("w1", "fact", memory_key="w", importance=0.1)
        self.backdate(pruned, 300)
        self.store.prune(dry_run=False)
        self.store.add_memory("w2", "fact", memory_key="w")
        with self.store._transaction() as connection:
            # Data written before pruned keys were linked to their successor.
            connection.execute(
                "UPDATE memories SET superseded_by = NULL WHERE id = ?", (pruned.id,)
            )
        with self.assertRaisesRegex(ValueError, "key w is held"):
            self.store.restore(pruned.id)

    def test_opening_an_old_database_flips_key_rewrite_edges_only(self):
        a1 = self.store.add_memory("a v1", "fact", memory_key="a")
        a2 = self.store.add_memory("a v2", "fact", memory_key="a")
        b1 = self.store.add_memory("b v1", "fact", memory_key="b")
        b2 = self.store.add_memory("b v2", "fact", memory_key="b")
        c1 = self.store.add_memory("c one", "fact")
        c2 = self.store.add_memory("c two", "fact")
        with self.store._transaction() as connection:
            connection.execute(
                "UPDATE relationships SET source_id = ?, target_id = ? "
                "WHERE source_id = ? AND target_id = ?",
                (a1.id, a2.id, a2.id, a1.id),
            )
            for source, target in ((b1, b2), (c1, c2)):
                connection.execute(
                    "INSERT INTO relationships VALUES (?, ?, 'supersedes', ?)",
                    (source.id, target.id, a1.created_at),
                )
            connection.execute("PRAGMA user_version = 0")
        self.assertIn((a1.id, a2.id), self.supersedes_edges())

        reopened = MemoryStore(self.store.db_path)

        self.assertEqual(
            self.supersedes_edges(reopened),
            {(a2.id, a1.id), (b2.id, b1.id), (c1.id, c2.id)},
        )
        with reopened._transaction() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_a_version_0_1_database_is_upgraded_in_place(self):
        path = Path(self.temp_dir.name) / "v0.1.sqlite3"
        memory_rows = [
            # id, key, archived_at, superseded_by; all created in one second
            ("v1", "k", SECOND, "v2"),
            ("v2", "k", None, None),
            ("m1", None, None, None),
            ("m2", None, None, None),
            ("x", None, None, None),
            ("y", None, None, None),
            ("dangling", None, SECOND, "gone"),
        ]
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(VERSION_0_1_SCHEMA)
            connection.executemany(
                "INSERT INTO memories (id, kind, content, memory_key, source, "
                "confidence, importance, created_at, updated_at, archived_at, "
                "superseded_by) VALUES (?, 'fact', ?, ?, 'test', 0.8, 0.5, ?, ?, ?, ?)",
                [
                    (memory_id, "memory " + memory_id, key, SECOND, SECOND, archived, superseded_by)
                    for memory_id, key, archived, superseded_by in memory_rows
                ],
            )
            connection.executemany(
                "INSERT INTO relationships VALUES (?, ?, 'supersedes', ?)",
                [
                    ("v1", "v2", SECOND),  # written by a key rewrite: old -> new
                    ("m1", "m2", SECOND),  # written by hand: left alone
                    ("dangling", "gone", SECOND),  # target row no longer exists
                ],
            )
            connection.commit()

        store = MemoryStore(path)

        self.assertEqual(
            self.supersedes_edges(store),
            {("v2", "v1"), ("m1", "m2"), ("dangling", "gone")},
        )
        self.assertEqual([memory.id for memory in store.history("v1")], ["v2", "v1"])
        # Same-second timestamps: insertion order decides which is older.
        with self.assertRaises(ValueError):
            store.relate("x", "y", Relationship.SUPERSEDES)
        store.relate("y", "x", Relationship.SUPERSEDES)
        self.assertEqual(store.get_memory("x").superseded_by, "y")

    def test_history_is_newest_first_when_one_version_replaced_several(self):
        first_n = self.store.add_memory("n1", "fact", memory_key="n")
        first_o = self.store.add_memory("o1", "fact", memory_key="o")
        second_o = self.store.add_memory("o2", "fact", memory_key="o")
        second_n = self.store.add_memory("n2", "fact", memory_key="n")
        self.store.relate(second_n.id, second_o.id, Relationship.SUPERSEDES)

        self.assertEqual(
            [memory.id for memory in self.store.history(first_o.id)],
            [second_n.id, second_o.id, first_o.id, first_n.id],
        )

    def test_concurrent_writes_to_one_key_keep_a_single_history(self):
        self.store.add_memory("v1", "fact", memory_key="k")
        paused, resume = threading.Event(), threading.Event()

        class PausingStore(MemoryStore):
            def _record_supersession(self, connection, new_id, old_id, now):
                paused.set()
                resume.wait(5)
                super()._record_supersession(connection, new_id, old_id, now)

        slow_writer = threading.Thread(
            target=PausingStore(self.store.db_path).add_memory,
            args=("v3", "fact"),
            kwargs={"memory_key": "k"},
        )
        slow_writer.start()
        self.assertTrue(paused.wait(5))
        other_writer = threading.Thread(
            target=self.store.add_memory,
            args=("v2", "fact"),
            kwargs={"memory_key": "k"},
        )
        other_writer.start()
        time.sleep(0.3)  # let the second writer start while the first holds its place
        resume.set()
        slow_writer.join(10)
        other_writer.join(10)

        memories = self.store.list_memories(include_archived=True)
        active = [memory for memory in memories if memory.archived_at is None]
        self.assertEqual(len(active), 1)
        self.assertEqual(len(self.store.history(active[0].id)), 3)
        self.assertTrue(
            all(memory.superseded_by for memory in memories if memory.archived_at)
        )

    def test_contradictions_stay_visible_outside_the_limit_and_kind_filter(self):
        claim = self.store.add_memory("Deploy target is staging zeta.", MemoryType.FACT)
        conflict = self.store.add_memory("Production only.", MemoryType.CORRECTION)
        self.store.relate(conflict.id, claim.id, Relationship.CONTRADICTS)
        expected = (Link(conflict.id, claim.id, Relationship.CONTRADICTS),)

        for options in ({"limit": 1}, {"kind": MemoryType.FACT}):
            results = self.store.search("zeta", record_access=False, **options)
            self.assertEqual([result.memory.id for result in results], [claim.id])
            self.assertEqual(results[0].links, expected, options)

        newer = self.store.add_memory("Staging is fine now.", MemoryType.CORRECTION)
        self.store.relate(newer.id, conflict.id, Relationship.SUPERSEDES)
        self.assertEqual(self.store.search("zeta", limit=1)[0].links, ())

    def test_a_graph_bonus_is_explained_even_when_its_source_is_cut_off(self):
        strong = self.store.add_memory("alpha alpha beta", MemoryType.FACT)
        weak = self.store.add_memory("alpha gamma delta epsilon zeta", MemoryType.FACT)
        self.store.relate(weak.id, strong.id, Relationship.SUPPORTS)

        results = self.store.search("alpha", limit=1, record_access=False)

        self.assertEqual([result.memory.id for result in results], [strong.id])
        self.assertEqual(results[0].matched_by, "lexical+graph")
        self.assertEqual(
            results[0].links, (Link(weak.id, strong.id, Relationship.SUPPORTS),)
        )

    def test_decay_can_differ_by_memory_type(self):
        now = datetime.now(timezone.utc)
        episode = self.store.add_memory(
            "Meeting detail.", MemoryType.EPISODE, importance=0.5, confidence=1.0
        )
        constraint = self.store.add_memory(
            "Constraint detail.",
            MemoryType.TECHNICAL_CONSTRAINT,
            importance=0.5,
            confidence=1.0,
        )
        for memory in (episode, constraint):
            self.backdate(memory, 60, now)

        self.assertEqual(self.store.prune(as_of=now), [])
        flagged = self.store.prune(as_of=now, decay_days_by_kind={"episode": 14})
        self.assertEqual([candidate.memory.id for candidate in flagged], [episode.id])
        with self.assertRaises(ValueError):
            self.store.prune(decay_days_by_kind={"bogus": 3})
        with self.assertRaises(ValueError):
            self.store.prune(decay_days_by_kind={"episode": 0})

    def test_importance_protection_is_configurable(self):
        now = datetime.now(timezone.utc)
        memory = self.store.add_memory(
            "Important but stale.", MemoryType.FACT, importance=0.95, confidence=0.1
        )
        self.backdate(memory, 400, now)

        self.assertEqual(self.store.prune(as_of=now), [])
        self.assertEqual(
            [
                candidate.memory.id
                for candidate in self.store.prune(as_of=now, protect_importance=None)
            ],
            [memory.id],
        )

    def test_link_rejects_self_links_and_unknown_memories(self):
        memory = self.store.add_memory("A fact.", MemoryType.FACT)
        with self.assertRaises(ValueError):
            self.store.relate(memory.id, memory.id)
        with self.assertRaises(ValueError):
            self.store.relate(memory.id, "missing")

    def test_in_memory_database_stays_available_between_operations(self):
        store = MemoryStore(":memory:")
        memory = store.add_memory("Keep this for this process.", MemoryType.FACT)

        self.assertEqual(store.get_memory(memory.id), memory)


if __name__ == "__main__":
    unittest.main()
