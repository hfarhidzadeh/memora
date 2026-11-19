"""Regression tests: claims are resolved at read time, from what the viewers can see.

Found in review on 2026-11-18. Claims used to be resolved when written, using
every claim in the channel, and two failures followed:

1. A claim drawn from a narrower audience (a source message in a direct
   message) replaced or resolved claims that the rest of the channel could see.
   They saw their claim disappear and the hidden claim's ID in
   ``superseded_by``.
2. Retracting the message behind a replacing claim left the key with no
   current value, because the claims it had replaced stayed replaced.

Most failure tests compare what a viewer reads with a world in which the hidden
or retracted write never happened. Each class also has controls that passed
before the fix: the same write from a source everyone can see still replaces,
and viewers who can see the narrower source still get the replacement.
"""

import tempfile
import unittest
from pathlib import Path

from dynmemory import MemoryStore, MemoryType, Relationship

FACT = MemoryType.FACT


class ClaimResolutionCase(unittest.TestCase):
    """alice, bob and carol are in ``design``; only alice and bob are in ``dm``."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stores = 0

    def tearDown(self):
        self.temp_dir.cleanup()

    def workspace(self, admins=()):
        self.stores += 1
        store = MemoryStore(
            Path(self.temp_dir.name) / "world{}.sqlite3".format(self.stores)
        )
        for member in ("alice", "bob", "carol"):
            store.join_channel("design", member)
        for member in ("alice", "bob"):
            store.join_channel("dm", member)
        for admin in admins:
            store.add_admin(admin)
        return store

    @staticmethod
    def claim(store, speaker, content, *, source_channel="design", key="launch"):
        """Record a message and the keyed ``design`` claim derived from it."""
        writer = store.for_writer(speaker)
        message = writer.add_message(content, channel=source_channel)
        memory = writer.add_memory(
            content, FACT, channel="design", memory_key=key,
            source_message_ids=[message.id],
        )
        return message, memory

    @staticmethod
    def note(store, speaker, content, *, source_channel="design"):
        """Record a message and an unkeyed ``design`` memory derived from it."""
        writer = store.for_writer(speaker)
        message = writer.add_message(content, channel=source_channel)
        memory = writer.add_memory(
            content, FACT, channel="design", source_message_ids=[message.id]
        )
        return message, memory

    @staticmethod
    def observed(store, viewers, query="launch"):
        """Everything a search shows, with memory IDs replaced by contents."""
        def label(memory_id):
            memory = store.get_memory(memory_id)
            return memory.content if memory is not None else memory_id

        return [
            (
                result.memory.content,
                result.claim_status,
                result.memory.speaker,
                result.score,
                result.matched_by,
                result.lexical_score,
                result.vector_score,
                result.coverage,
                sorted(
                    (link.relation.value, label(link.source_id), label(link.target_id))
                    for link in result.links
                ),
            )
            for result in store.search(query, viewers=viewers, record_access=False)
        ]

    @staticmethod
    def shown(store, viewers, query="launch"):
        return {
            result.memory.content: result.claim_status
            for result in store.search(query, viewers=viewers, record_access=False)
        }

    @staticmethod
    def listed(store, viewers):
        return sorted(
            (memory.content, memory.archived_at is None, memory.superseded_by is None)
            for memory in store.list_memories(include_archived=True, viewers=viewers)
        )


class NarrowerAudienceTests(ClaimResolutionCase):
    def test_dm_sourced_correction_changes_nothing_an_outsider_reads(self):
        with_hidden, without = self.workspace(), self.workspace()
        for store in (with_hidden, without):
            self.claim(store, "alice", "Launch is March 3")
        self.claim(with_hidden, "alice", "Launch is April", source_channel="dm")

        self.assertEqual(self.shown(with_hidden, "carol"), {"Launch is March 3": "current"})
        self.assertEqual(
            self.observed(with_hidden, "carol"), self.observed(without, "carol")
        )
        self.assertEqual(self.listed(with_hidden, "carol"), self.listed(without, "carol"))

    def test_outsider_reads_expose_no_hidden_replacer(self):
        store = self.workspace()
        _, march = self.claim(store, "alice", "Launch is March 3")
        _, april = self.claim(store, "alice", "Launch is April", source_channel="dm")

        seen = store.get_memory(march.id, viewers="carol")
        self.assertIsNone(seen.archived_at)
        self.assertIsNone(seen.superseded_by)
        self.assertEqual([m.id for m in store.history(march.id, viewers="carol")], [march.id])
        self.assertNotIn(
            april.id,
            {m.superseded_by for m in store.list_memories(include_archived=True, viewers="carol")},
        )

    def test_dm_sourced_correction_applies_for_viewers_who_see_it(self):
        # Control: passed before the fix.
        store = self.workspace()
        _, march = self.claim(store, "alice", "Launch is March 3")
        _, april = self.claim(store, "alice", "Launch is April", source_channel="dm")

        self.assertEqual(
            self.shown(store, ["alice", "bob"]), {"Launch is April": "current"}
        )
        self.assertEqual(
            [m.id for m in store.history(april.id, viewers=["alice", "bob"])],
            [april.id, march.id],
        )
        self.assertEqual(
            store.get_memory(march.id, viewers=["alice", "bob"]).superseded_by, april.id
        )

    def test_dm_sourced_admin_claim_leaves_a_visible_dispute_open(self):
        with_hidden, without = self.workspace(admins=["alice"]), self.workspace(admins=["alice"])
        for store in (with_hidden, without):
            self.claim(store, "carol", "Launch is March 3")
            self.claim(store, "bob", "Launch is March 10")
        self.claim(with_hidden, "alice", "Launch is April", source_channel="dm")

        self.assertEqual(
            self.shown(with_hidden, "carol"),
            {"Launch is March 3": "disputed", "Launch is March 10": "disputed"},
        )
        self.assertEqual(
            self.observed(with_hidden, "carol"), self.observed(without, "carol")
        )

    def test_dm_sourced_admin_claim_resolves_for_viewers_who_see_it(self):
        # Control: passed before the fix.
        store = self.workspace(admins=["alice"])
        self.claim(store, "carol", "Launch is March 3")
        self.claim(store, "bob", "Launch is March 10")
        self.claim(store, "alice", "Launch is April", source_channel="dm")

        self.assertEqual(
            self.shown(store, ["alice", "bob"]), {"Launch is April": "current"}
        )

    def test_correction_everyone_can_see_still_replaces(self):
        # Control: passed before the fix.
        store = self.workspace()
        _, march = self.claim(store, "alice", "Launch is March 3")
        _, april = self.claim(store, "alice", "Launch is April")

        self.assertEqual(self.shown(store, "carol"), {"Launch is April": "current"})
        self.assertEqual(store.get_memory(march.id, viewers="carol").superseded_by, april.id)

    def test_explicit_supersession_from_a_narrower_source_stays_hidden(self):
        store = self.workspace()
        _, plan = self.note(store, "alice", "Launch plan A")
        _, secret = self.note(store, "alice", "Launch plan B", source_channel="dm")
        store.relate(secret.id, plan.id, Relationship.SUPERSEDES)

        self.assertEqual(self.shown(store, "carol"), {"Launch plan A": "current"})
        self.assertIsNone(store.get_memory(plan.id, viewers="carol").superseded_by)
        # Control: the members of the direct message see the supersession.
        self.assertEqual(self.shown(store, ["alice", "bob"]), {"Launch plan B": "current"})
        self.assertEqual(
            store.get_memory(plan.id, viewers=["alice", "bob"]).superseded_by, secret.id
        )


class RetractionTests(ClaimResolutionCase):
    def test_retracting_a_correction_reinstates_the_earlier_claim(self):
        retracted, never = self.workspace(), self.workspace()
        for store in (retracted, never):
            self.claim(store, "alice", "Review is Thursday", key="review")
        message, _ = self.claim(retracted, "alice", "Review is Friday", key="review")
        self.assertEqual(self.shown(retracted, "carol", "review"), {"Review is Friday": "current"})

        retracted.for_writer("alice").retract_message(message.id)

        self.assertEqual(
            self.shown(retracted, "carol", "review"), {"Review is Thursday": "current"}
        )
        self.assertEqual(
            self.observed(retracted, "carol", "review"),
            self.observed(never, "carol", "review"),
        )

    def test_retracting_an_admin_resolution_reopens_the_dispute(self):
        retracted, never = self.workspace(admins=["carol"]), self.workspace(admins=["carol"])
        for store in (retracted, never):
            self.claim(store, "alice", "Review is Thursday", key="review")
            self.claim(store, "bob", "Review is Friday", key="review")
        message, _ = self.claim(retracted, "carol", "Review is Thursday", key="review")
        self.assertEqual(
            self.shown(retracted, "carol", "review"), {"Review is Thursday": "current"}
        )

        retracted.for_writer("carol").retract_message(message.id)

        self.assertEqual(
            self.shown(retracted, "carol", "review"),
            {"Review is Thursday": "disputed", "Review is Friday": "disputed"},
        )
        self.assertEqual(
            self.observed(retracted, "carol", "review"),
            self.observed(never, "carol", "review"),
        )

    def test_retracting_a_dm_sourced_correction_reinstates_it_for_dm_members(self):
        store = self.workspace()
        self.claim(store, "alice", "Launch is March 3")
        message, _ = self.claim(store, "alice", "Launch is April", source_channel="dm")
        store.for_writer("alice").retract_message(message.id)

        self.assertEqual(
            self.shown(store, ["alice", "bob"]), {"Launch is March 3": "current"}
        )

    def test_retracting_an_explicit_supersession_source_reinstates_the_target(self):
        store = self.workspace()
        _, plan = self.note(store, "alice", "Launch plan A")
        message, newer = self.note(store, "alice", "Launch plan B")
        store.relate(newer.id, plan.id, Relationship.SUPERSEDES)
        store.for_writer("alice").retract_message(message.id)

        self.assertEqual(self.shown(store, "carol"), {"Launch plan A": "current"})
        self.assertIsNone(store.get_memory(plan.id, viewers="carol").superseded_by)

    def test_retracting_a_claim_that_replaced_nothing_changes_nothing_else(self):
        # Control: passed before the fix.
        retracted, never = self.workspace(), self.workspace()
        for store in (retracted, never):
            self.claim(store, "alice", "Review is Thursday", key="review")
        message, _ = self.claim(retracted, "bob", "Review is Friday", key="review")

        retracted.for_writer("bob").retract_message(message.id)

        self.assertEqual(
            self.observed(retracted, "carol", "review"),
            self.observed(never, "carol", "review"),
        )


class RelateTests(ClaimResolutionCase):
    """``relate`` checks only the two memories' own state, never other memories.

    Found by the noninterference experiment (version 2, first run): ``relate``
    refused an end that a hidden write had replaced in the administrative
    view, so the viewers lost a link they would otherwise have seen.
    """

    def test_a_hidden_replacement_does_not_change_whether_a_relate_succeeds(self):
        with_hidden, without = self.workspace(), self.workspace()
        ends = {}
        for store in (with_hidden, without):
            _, march = self.claim(store, "alice", "Launch is March 3")
            _, note = self.note(store, "bob", "Launch checklist ready")
            ends[store] = (note, march)
        self.claim(with_hidden, "alice", "Launch is April", source_channel="dm")

        for store, (note, march) in ends.items():
            store.relate(note.id, march.id, Relationship.SUPPORTS)

        self.assertEqual(
            self.observed(with_hidden, "carol"), self.observed(without, "carol")
        )
        self.assertIn(
            ("supports", "Launch checklist ready", "Launch is March 3"),
            [link for row in self.observed(with_hidden, "carol") for link in row[8]],
        )

    def test_a_link_to_a_replaced_claim_counts_only_where_the_claim_stands(self):
        store = self.workspace()
        _, march = self.claim(store, "alice", "Launch is March 3")
        message, _ = self.claim(store, "alice", "Launch is April")
        _, note = self.note(store, "bob", "Launch checklist ready")
        store.relate(note.id, march.id, Relationship.SUPPORTS)

        shown = {
            result.memory.content: result
            for result in store.search("launch", viewers="carol", record_access=False)
        }
        self.assertNotIn("Launch is March 3", shown)
        self.assertEqual(shown["Launch checklist ready"].links, ())
        # Once the correction is retracted, March 3 stands again, with its link.
        store.for_writer("alice").retract_message(message.id)
        shown = {
            result.memory.content: result
            for result in store.search("launch", viewers="carol", record_access=False)
        }
        self.assertEqual(
            [link.relation for link in shown["Launch checklist ready"].links],
            [Relationship.SUPPORTS],
        )

    def test_relate_still_refuses_retracted_and_pruned_memories(self):
        # Control: passed before the change.
        store = self.workspace()
        message, retracted = self.claim(store, "alice", "Launch is March 3")
        _, note = self.note(store, "bob", "Launch checklist ready")
        _, old = self.note(store, "bob", "Launch notes from last year")
        store.for_writer("alice").retract_message(message.id)
        store._set_activity(old.id, created_at="2020-01-01T00:00:00+00:00")
        self.assertEqual([c.memory.id for c in store.prune(dry_run=False)], [old.id])

        for target in (retracted, old):
            with self.assertRaisesRegex(ValueError, "exist and be active"):
                store.relate(note.id, target.id)


class ReadTimeResolutionTests(ClaimResolutionCase):
    """What changes, besides the two fixes, once channel claims resolve at read time."""

    def backdate(self, store, memory):
        store._set_activity(memory.id, created_at="2020-01-01T00:00:00+00:00")

    def test_prune_leaves_replaced_claims_as_history(self):
        store = self.workspace()
        _, thursday = self.claim(store, "alice", "Review is Thursday", key="review")
        message, _ = self.claim(store, "alice", "Review is Friday", key="review")
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memories SET importance = 0.01, last_accessed_at = ?, "
                "created_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00", thursday.id),
            )
        self.assertNotIn(thursday.id, {c.memory.id for c in store.prune(dry_run=False)})

        store.for_writer("alice").retract_message(message.id)

        # Not pruned while replaced, so the retraction can reinstate it.
        self.assertEqual(self.shown(store, "carol", "review"), {"Review is Thursday": "current"})

    def test_restore_refuses_a_claim_its_speaker_replaced(self):
        store = self.workspace()
        _, thursday = self.claim(store, "alice", "Review is Thursday", key="review")
        self.backdate(store, thursday)
        store.prune(dry_run=False, protect_importance=None)
        _, friday = self.claim(store, "alice", "Review is Friday", key="review")

        with self.assertRaisesRegex(ValueError, "superseded by {}".format(friday.id)):
            store.restore(thursday.id)
        with self.assertRaisesRegex(ValueError, "already active"):
            store.restore(friday.id)

    def test_relate_does_not_supersede_a_keyed_channel_claim(self):
        store = self.workspace()
        _, claim = self.claim(store, "alice", "Launch is March 3")
        _, note = self.note(store, "alice", "Launch moved")
        with self.assertRaisesRegex(ValueError, "writing a new claim"):
            store.relate(note.id, claim.id, Relationship.SUPERSEDES)
        # Control: a keyed claim may still supersede an unkeyed note.
        _, older_note = self.note(store, "alice", "Launch planning started")
        _, newer = self.claim(store, "alice", "Launch is April")
        store.relate(newer.id, older_note.id, Relationship.SUPERSEDES)
        self.assertEqual(store.get_memory(older_note.id).superseded_by, newer.id)

    def test_version_5_write_time_supersessions_are_undone_on_upgrade(self):
        store = self.workspace()
        _, thursday = self.claim(store, "alice", "Review is Thursday", key="review")
        _, friday = self.claim(store, "alice", "Review is Friday", key="review")
        _, monday = self.claim(store, "bob", "Review is Monday", key="review")
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memories SET archived_at = ? WHERE id = ?",
                ("2026-01-01T00:00:00.000000+00:00", monday.id),
            )
        _, sunday = self.claim(store, "bob", "Review is Sunday", key="review")
        _, plan_a = self.note(store, "alice", "Review plan A")
        _, plan_b = self.note(store, "alice", "Review plan B")
        store.relate(plan_b.id, plan_a.id, Relationship.SUPERSEDES)
        expected = self.observed(store, "carol", "review")
        expected_listed = self.listed(store, "carol")

        # Reshape into what schema version 5 stored: write-time supersession.
        with store._transaction() as connection:
            for older, newer in ((thursday, friday), (monday, sunday)):
                connection.execute(
                    "UPDATE memories SET superseded_by = ? WHERE id = ?",
                    (newer.id, older.id),
                )
                connection.execute(
                    "INSERT INTO relationships VALUES (?, ?, 'supersedes', ?)",
                    (newer.id, older.id, newer.created_at),
                )
            connection.execute(
                "UPDATE memories SET archived_at = ? WHERE id = ?",
                (friday.created_at, thursday.id),
            )
            (linked_at,) = connection.execute(
                "SELECT created_at FROM relationships WHERE source_id = ?", (plan_b.id,)
            ).fetchone()
            connection.execute(
                "UPDATE memories SET archived_at = ?, superseded_by = ? WHERE id = ?",
                (linked_at, plan_b.id, plan_a.id),
            )
            connection.execute(
                """CREATE UNIQUE INDEX one_active_memory_per_channel_key
                ON memories(channel, memory_key, speaker)
                WHERE memory_key IS NOT NULL AND archived_at IS NULL
                  AND channel IS NOT NULL AND speaker IS NOT NULL"""
            )
            connection.execute("PRAGMA user_version = 5")

        reopened = MemoryStore(store.db_path)
        with reopened._transaction(write=False) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 6)
            rows = {
                row["id"]: (row["archived_at"] is None, row["superseded_by"])
                for row in connection.execute("SELECT * FROM memories")
            }
            supersedes = {
                (row["source_id"], row["target_id"])
                for row in connection.execute(
                    "SELECT * FROM relationships WHERE relation = 'supersedes'"
                )
            }
        self.assertEqual(rows[thursday.id], (True, None))
        self.assertEqual(rows[monday.id], (False, None))  # pruned first: stays archived
        self.assertEqual(rows[plan_a.id], (True, None))
        self.assertEqual(supersedes, {(plan_b.id, plan_a.id)})
        self.assertEqual(self.observed(reopened, "carol", "review"), expected)
        self.assertEqual(self.listed(reopened, "carol"), expected_listed)
        self.assertEqual(reopened.get_memory(thursday.id).superseded_by, friday.id)
        self.assertEqual(reopened.get_memory(plan_a.id).superseded_by, plan_b.id)
        # The per-speaker unique index is gone, so a speaker can write again.
        self.claim(reopened, "alice", "Review is Tuesday", key="review")


if __name__ == "__main__":
    unittest.main()
