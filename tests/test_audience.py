import sqlite3
import tempfile
import unittest
from contextlib import closing, ExitStack
from pathlib import Path
from unittest import mock

from dynmemory import (
    AudienceReader,
    Link,
    LinkCarryover,
    MemoryStore,
    MemoryType,
    NGramHashEmbedder,
    Relationship,
)

# The schema exactly as version 0.2 created it (schema version 1).
VERSION_0_2_SCHEMA = """
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
CREATE INDEX memories_superseded_by ON memories(superseded_by);
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
CREATE TABLE embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (memory_id, model)
);
PRAGMA user_version = 1;
"""


def signals(results):
    """Everything a viewer can read from a search, in rank order."""
    return [
        (
            result.memory.content,
            result.score,
            result.matched_by,
            result.lexical_score,
            result.vector_score,
            result.coverage,
            len(result.links),
        )
        for result in results
    ]


class AudienceCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stores = 0
        self.store = self.new_store()

    def tearDown(self):
        self.temp_dir.cleanup()

    def new_store(self, **options):
        self.stores += 1
        path = Path(self.temp_dir.name) / "store{}.sqlite3".format(self.stores)
        return MemoryStore(path, **options)

    def say(self, content, channel, speaker="alice", store=None, **options):
        options.setdefault("kind", MemoryType.FACT)
        kind = options.pop("kind")
        return (store or self.store).add_memory(
            content, kind, channel=channel, speaker=speaker, **options
        )

    def contents(self, viewers, query="launch", **options):
        return [
            result.memory.content
            for result in self.store.search(query, viewers=viewers, **options)
        ]


class MembershipTests(AudienceCase):
    def test_join_and_leave_change_the_current_members(self):
        self.store.join_channel("design", "alice")
        self.store.join_channel("design", "bob")
        self.store.join_channel("ops", "alice")
        self.assertEqual(self.store.members_of("design"), ["alice", "bob"])
        self.assertEqual(self.store.channels_of("alice"), ["design", "ops"])

        self.store.leave_channel("design", "alice")

        self.assertEqual(self.store.members_of("design"), ["bob"])
        self.assertEqual(self.store.channels_of("alice"), ["ops"])

    def test_joining_twice_changes_nothing_and_rejoining_works(self):
        self.store.join_channel("design", "alice")
        self.store.join_channel("design", " alice ")
        self.assertEqual(self.store.members_of("design"), ["alice"])

        self.store.leave_channel("design", "alice")
        self.store.join_channel("design", "alice")

        self.assertEqual(self.store.members_of("design"), ["alice"])
        with self.store._transaction(write=False) as connection:
            stays = connection.execute("SELECT COUNT(*) FROM channel_members").fetchone()[0]
        self.assertEqual(stays, 2)

    def test_leaving_a_channel_one_is_not_in_is_an_error(self):
        self.store.join_channel("design", "alice")
        with self.assertRaisesRegex(ValueError, "alcie is not a member of design"):
            self.store.leave_channel("design", "alcie")
        self.store.leave_channel("design", "alice")
        with self.assertRaisesRegex(ValueError, "not a member"):
            self.store.leave_channel("design", "alice")

    def test_names_must_not_be_empty(self):
        for call in (
            lambda: self.store.join_channel(" ", "alice"),
            lambda: self.store.join_channel("design", ""),
            lambda: self.store.leave_channel("design", " "),
            lambda: self.store.add_memory("x", "fact", channel=" "),
            lambda: self.store.add_memory("x", "fact", speaker=""),
        ):
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                call()


class WriteTests(AudienceCase):
    def test_a_memory_keeps_its_channel_and_speaker(self):
        memory = self.say("The launch is on Friday.", " design ", speaker=" alice ")
        self.assertEqual((memory.channel, memory.speaker), ("design", "alice"))
        stored = self.store.get_memory(memory.id)
        self.assertEqual((stored.channel, stored.speaker), ("design", "alice"))

        plain = self.store.add_memory("No channel.", "fact")
        self.assertEqual((plain.channel, plain.speaker), (None, None))

    def test_a_key_belongs_to_its_channel(self):
        design = self.say("Launch on Friday.", "design", memory_key="launch")
        ops = self.say("Launch on Monday.", "ops", memory_key="launch")
        plain = self.store.add_memory("Launch some day.", "fact", memory_key="launch")

        for memory in (design, ops, plain):
            self.assertIsNone(self.store.get_memory(memory.id).archived_at)

        newer = self.say("Launch on Saturday.", "design", memory_key="launch")

        self.assertEqual(self.store.get_memory(design.id).superseded_by, newer.id)
        for memory in (ops, plain, newer):
            self.assertIsNone(self.store.get_memory(memory.id).archived_at)
        self.assertEqual(
            [memory.id for memory in self.store.history(newer.id)], [newer.id, design.id]
        )
        self.assertEqual([memory.id for memory in self.store.history(ops.id)], [ops.id])

    def test_a_memory_can_only_supersede_one_in_its_own_channel(self):
        old = self.say("Launch on Friday.", "design")
        other = self.say("Launch on Monday.", "ops")
        plain = self.store.add_memory("Launch some day.", "fact")
        for newer in (other, plain):
            with self.assertRaisesRegex(ValueError, "same channel"):
                self.store.relate(newer.id, old.id, Relationship.SUPERSEDES)
        self.assertIsNone(self.store.get_memory(old.id).archived_at)

        same = self.say("Launch on Saturday.", "design")
        self.store.relate(same.id, old.id, Relationship.SUPERSEDES)
        self.assertEqual(self.store.get_memory(old.id).superseded_by, same.id)

    def test_restore_checks_the_key_within_the_channel(self):
        pruned = self.say("Launch on Friday.", "design", memory_key="launch", importance=0.1)
        self.store._set_activity(pruned.id, created_at="2020-01-01T00:00:00+00:00")
        self.assertEqual(len(self.store.prune(dry_run=False)), 1)
        self.say("Launch on Monday.", "ops", memory_key="launch")

        self.assertIsNone(self.store.restore(pruned.id).archived_at)


class VisibilityTests(AudienceCase):
    def setUp(self):
        super().setUp()
        self.store.join_channel("design", "alice")
        self.store.join_channel("design", "bob")
        self.store.join_channel("ops", "alice")
        self.design = self.say("The launch design is final.", "design")
        self.ops = self.say("The launch rollback plan is ready.", "ops")
        self.plain = self.store.add_memory("The launch has no channel.", "fact")

    def test_a_viewer_finds_only_memories_in_their_channels(self):
        self.assertEqual(
            sorted(self.contents("alice")),
            ["The launch design is final.", "The launch rollback plan is ready."],
        )
        self.assertEqual(self.contents("bob"), ["The launch design is final."])
        self.assertEqual(self.contents(["bob"]), ["The launch design is final."])

    def test_without_viewers_everything_is_searched(self):
        self.assertEqual(len(self.store.search("launch")), 3)

    def test_a_memory_without_a_channel_is_hidden_from_every_viewer(self):
        self.assertNotIn("The launch has no channel.", self.contents("alice"))

    def test_a_stranger_finds_nothing(self):
        self.assertEqual(self.contents("mallory"), [])

    def test_several_viewers_see_only_what_all_of_them_may_see(self):
        self.assertEqual(self.contents(["alice", "bob"]), ["The launch design is final."])
        self.assertEqual(self.contents(["bob", "alice", "bob"]), ["The launch design is final."])
        self.assertEqual(self.contents(["alice", "mallory"]), [])

    def test_viewers_must_name_someone(self):
        for viewers in ([], "", [" "], ["alice", None]):
            with self.assertRaises(ValueError):
                self.store.search("launch", viewers=viewers)

    def test_a_viewer_name_is_one_member_not_its_letters(self):
        self.store.join_channel("design", "b")
        self.assertEqual(self.contents("bob"), ["The launch design is final."])
        self.assertEqual(self.contents(" bob "), ["The launch design is final."])

    def test_a_joiner_reads_older_history(self):
        self.assertEqual(self.contents("carol"), [])
        self.store.join_channel("ops", "carol")
        self.assertEqual(self.contents("carol"), ["The launch rollback plan is ready."])

    def test_a_leaver_loses_the_channel_including_their_own_words(self):
        self.say("I, Bob, approve the launch.", "design", speaker="bob")
        self.assertEqual(len(self.contents("bob")), 2)

        self.store.leave_channel("design", "bob")

        self.assertEqual(self.contents("bob"), [])
        self.assertEqual(len(self.contents("alice")), 3)

    def test_get_history_and_list_respect_viewers(self):
        newer = self.say("The launch rollback plan changed.", "ops")
        self.store.relate(newer.id, self.ops.id, Relationship.SUPERSEDES)

        self.assertEqual(self.store.get_memory(self.ops.id, viewers="alice").id, self.ops.id)
        self.assertIsNone(self.store.get_memory(self.ops.id, viewers="bob"))
        self.assertIsNone(self.store.get_memory(self.plain.id, viewers="alice"))
        self.assertIsNone(self.store.get_memory("missing", viewers="alice"))

        self.assertEqual(
            [memory.id for memory in self.store.history(self.ops.id, viewers="alice")],
            [newer.id, self.ops.id],
        )
        self.assertEqual(self.store.history(self.ops.id, viewers="bob"), [])
        self.assertEqual(self.store.history(newer.id, viewers=["alice", "bob"]), [])

        self.assertEqual(
            {memory.id for memory in self.store.list_memories(viewers="alice")},
            {self.design.id, newer.id},
        )
        self.assertEqual(
            {
                memory.id
                for memory in self.store.list_memories(viewers="alice", include_archived=True)
            },
            {self.design.id, newer.id, self.ops.id},
        )
        self.assertEqual(
            [memory.id for memory in self.store.list_memories(viewers="bob")],
            [self.design.id],
        )
        self.assertEqual(len(self.store.list_memories()), 3)

    def test_only_shown_memories_count_as_accessed(self):
        self.store.search("launch", viewers="bob")
        self.assertEqual(self.store.get_memory(self.design.id).access_count, 1)
        self.assertEqual(self.store.get_memory(self.ops.id).access_count, 0)


class HiddenMemoriesChangeNothingTests(AudienceCase):
    """A viewer's search must read the same with and without hidden memories."""

    VISIBLE = (
        "The launch date is Friday the ninth.",
        "Carol owns the launch checklist.",
        "The budget review moved to next week.",
        "Tuna treats are for the office cat.",
    )
    HIDDEN = (
        "The launch is cancelled because of the layoffs.",
        "Layoffs are planned for the launch team.",
        "The budget for the launch was cut.",
    )
    QUERIES = ("launch layoffs", "launch", "budget review", "layoffs", "cat treats")

    def build(self, with_hidden, **options):
        store = self.new_store(**options)
        store.join_channel("design", "alice")
        store.join_channel("board", "dana")
        for content in self.VISIBLE:
            self.say(content, "design", store=store)
        if with_hidden:
            for content in self.HIDDEN:
                self.say(content, "board", speaker="dana", store=store)
            store.add_memory("A launch note with no channel.", "fact")
        return store

    def assert_same_for_viewer(self, **options):
        without = self.build(False, **options)
        with_hidden = self.build(True, **options)
        for query in self.QUERIES:
            expected = signals(without.search(query, viewers="alice", record_access=False))
            self.assertEqual(
                signals(with_hidden.search(query, viewers="alice", record_access=False)),
                expected,
                query,
            )
            # The test can tell the difference: without viewers, scores move.
            if query == "launch layoffs":
                unfiltered = [
                    row
                    for row in signals(with_hidden.search(query, record_access=False))
                    if row[0] in self.VISIBLE
                ]
                self.assertNotEqual(unfiltered, expected)

    def test_lexical_scores_and_coverage(self):
        self.assert_same_for_viewer()

    def test_fused_lexical_and_vector_ranking(self):
        self.assert_same_for_viewer(embedder=NGramHashEmbedder())


class LinkTests(AudienceCase):
    def setUp(self):
        super().setUp()
        self.store.join_channel("design", "alice")
        self.store.join_channel("board", "dana")
        self.store.join_channel("board", "erin")
        self.store.join_channel("design", "erin")

    def test_a_hidden_neighbour_is_not_reached_through_a_link(self):
        seed = self.say("The launch date is Friday.", "design")
        hidden = self.say("Severance terms are being drafted.", "board", speaker="dana")
        shown = self.say("Carol owns the checklist.", "design")
        self.store.relate(seed.id, hidden.id)
        self.store.relate(seed.id, shown.id)

        results = self.store.search("launch", viewers="alice")

        self.assertEqual(
            [result.memory.id for result in results], [seed.id, shown.id]
        )
        self.assertEqual(results[0].links, (Link(seed.id, shown.id, Relationship.RELATED_TO),))
        # Positive control: a member of both channels reaches it.
        self.assertIn(
            hidden.id,
            [result.memory.id for result in self.store.search("launch", viewers="erin")],
        )

    def test_a_hidden_seed_does_not_pull_in_a_visible_neighbour(self):
        hidden = self.say("The launch is cancelled.", "board", speaker="dana")
        shown = self.say("Carol owns the checklist.", "design")
        self.store.relate(hidden.id, shown.id)

        self.assertEqual(self.store.search("launch", viewers="alice"), [])
        self.assertEqual(
            {result.memory.id for result in self.store.search("launch", viewers="erin")},
            {hidden.id, shown.id},
        )

    def test_a_hidden_disputant_is_not_flagged(self):
        claim = self.say("The launch date is Friday.", "design")
        hidden = self.say("The date is in fact Monday.", "board", speaker="dana")
        self.store.relate(hidden.id, claim.id, Relationship.CONTRADICTS)

        (result,) = self.store.search("launch", viewers="alice")
        self.assertEqual(result.links, ())

        # Positive control: someone in both channels sees the dispute.
        results = self.store.search("launch", viewers="erin", limit=1)
        self.assertEqual(
            results[0].links, (Link(hidden.id, claim.id, Relationship.CONTRADICTS),)
        )

    def test_a_visible_disputant_outside_the_kind_filter_is_still_flagged(self):
        claim = self.say("The launch date is Friday.", "design")
        dispute = self.say(
            "The date is in fact Monday.", "design", kind=MemoryType.CORRECTION
        )
        self.store.relate(dispute.id, claim.id, Relationship.CONTRADICTS)

        (result,) = self.store.search("launch", viewers="alice", kind="fact")

        self.assertEqual(
            result.links, (Link(dispute.id, claim.id, Relationship.CONTRADICTS),)
        )

    def topic_store(self):
        store = self.new_store(link_carryover=LinkCarryover.TOPIC)
        store.join_channel("design", "alice")
        store.join_channel("board", "dana")
        store.join_channel("design", "erin")
        store.join_channel("board", "erin")
        return store

    def test_topic_carry_over_does_not_cross_into_a_hidden_channel(self):
        store = self.topic_store()
        seed = self.say("The launch date is Friday.", "design", store=store)
        old = self.say("Severance is two weeks.", "board", store=store, memory_key="severance")
        store.relate(seed.id, old.id)
        new = self.say("Severance is four weeks.", "board", store=store, memory_key="severance")

        (result,) = store.search("launch", viewers="alice")
        self.assertEqual((result.memory.id, result.links), (seed.id, ()))

        # Positive control: with access to both, the link reaches the new version.
        results = store.search("launch", viewers="erin")
        self.assertEqual([r.memory.id for r in results], [seed.id, new.id])

    def test_topic_carry_over_still_works_inside_visible_channels(self):
        store = self.topic_store()
        seed = self.say("The launch date is Friday.", "design", store=store)
        old = self.say("Carol owns the checklist.", "design", store=store, memory_key="owner")
        store.relate(seed.id, old.id)
        new = self.say("Dave owns the checklist.", "design", store=store, memory_key="owner")

        results = store.search("launch", viewers="alice")

        self.assertEqual([r.memory.id for r in results], [seed.id, new.id])

    def test_a_version_chain_through_a_hidden_channel_is_not_followed(self):
        # The store refuses to write such a chain; this one is forced in by
        # hand to check that reading fails closed anyway.
        store = self.topic_store()
        seed = self.say("The launch date is Friday.", "design", store=store)
        first = self.say("Owner one.", "design", store=store)
        middle = self.say("Owner two.", "board", store=store)
        last = self.say("Owner three.", "design", store=store)
        store.relate(seed.id, first.id)
        with store._transaction() as connection:
            for older, newer in ((first, middle), (middle, last)):
                connection.execute(
                    "UPDATE memories SET archived_at = ?, superseded_by = ? WHERE id = ?",
                    (older.created_at, newer.id, older.id),
                )

        self.assertEqual(
            [r.memory.id for r in store.search("launch", viewers="alice")], [seed.id]
        )
        self.assertEqual(
            [r.memory.id for r in store.search("launch", viewers="erin")],
            [seed.id, last.id],
        )

    def test_each_layer_hides_links_on_its_own(self):
        # Search hides a link to a hidden memory in more than one place. Each
        # place is checked here directly, so none of them rests on another.
        store = self.topic_store()
        seed = self.say("The launch date is Friday.", "design", store=store)
        hidden = self.say("The date is in fact Monday.", "board", store=store)
        shown = self.say("Carol owns the checklist.", "design", store=store)
        store.relate(hidden.id, seed.id, Relationship.CONTRADICTS)
        store.relate(seed.id, shown.id)
        # A forced chain: the seed's earlier version sits in the hidden channel.
        earlier = self.say("The launch date was Thursday.", "board", store=store)
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memories SET archived_at = ?, superseded_by = ? WHERE id = ?",
                (earlier.created_at, seed.id, earlier.id),
            )
        alice, erin = ("alice",), ("erin",)
        visible_link = Link(seed.id, shown.id, Relationship.RELATED_TO)
        hidden_link = Link(hidden.id, seed.id, Relationship.CONTRADICTS)

        with store._transaction(write=False) as connection:
            self.assertEqual(
                store._links_touching(connection, [seed.id], alice), [visible_link]
            )
            self.assertEqual(
                sorted(store._links_touching(connection, [seed.id], erin), key=repr),
                sorted([visible_link, hidden_link], key=repr),
            )
            self.assertEqual(store._search_links(connection, [seed.id], alice), [visible_link])

            ids = [seed.id, hidden.id, shown.id]
            self.assertEqual(store._active_ids(connection, ids, alice), {seed.id, shown.id})
            self.assertEqual(store._active_ids(connection, ids, erin), set(ids))

            self.assertEqual(store._earlier_versions(connection, [seed.id], alice), set())
            self.assertEqual(store._earlier_versions(connection, [seed.id], erin), {earlier.id})

            self.assertEqual(store._current_versions(connection, [earlier.id], alice), {})
            self.assertEqual(
                store._current_versions(connection, [earlier.id], erin), {earlier.id: seed.id}
            )

    def test_search_hands_the_viewers_to_every_lookup(self):
        # The layers overlap, so a search result cannot show that one of them
        # was skipped. This checks the wiring instead.
        store = self.topic_store()
        seed = self.say("The launch date is Friday.", "design", store=store)
        other = self.say("Carol owns the launch checklist.", "design", store=store)
        store.relate(other.id, seed.id, Relationship.CONTRADICTS)
        helpers = (
            "_search_links",
            "_links_touching",
            "_earlier_versions",
            "_current_versions",
            "_active_ids",
        )
        with ExitStack() as stack:
            spies = {
                name: stack.enter_context(
                    mock.patch.object(store, name, wraps=getattr(store, name))
                )
                for name in helpers
            }
            store.search("launch", viewers=["erin", "alice"], limit=1)

        for name, spy in spies.items():
            self.assertTrue(spy.call_args_list, name)
            for call in spy.call_args_list:
                self.assertEqual(call.args[-1], ("alice", "erin"), name)


class DisputeTests(AudienceCase):
    def setUp(self):
        super().setUp()
        for member in ("alice", "bob", "dana", "erin"):
            self.store.join_channel("design", member)

    def test_different_speakers_keep_both_claims_and_mark_a_dispute(self):
        alice = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say("Review is Friday.", "design", speaker="bob", memory_key="review")

        self.assertIsNone(self.store.get_memory(alice.id).archived_at)
        self.assertIsNone(self.store.get_memory(bob.id).archived_at)
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual(set(results), {alice.id, bob.id})
        self.assertEqual({r.claim_status for r in results.values()}, {"disputed"})
        link = Link(bob.id, alice.id, Relationship.CONTRADICTS)
        self.assertIn(link, results[alice.id].links)
        self.assertIn(link, results[bob.id].links)

    def test_dispute_status_survives_kind_filter(self):
        alice = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say(
            "Review is Friday.", "design", speaker="bob",
            memory_key="review", kind=MemoryType.CORRECTION,
        )
        (result,) = self.store.search("review", viewers="alice", kind="fact")
        self.assertEqual(result.memory.id, alice.id)
        self.assertEqual(result.claim_status, "disputed")
        self.assertIn(Link(bob.id, alice.id, Relationship.CONTRADICTS), result.links)

    def test_matching_text_from_another_speaker_is_agreement(self):
        alice = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say(" review is thursday. ", "design", speaker="bob", memory_key="review")
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual(set(results), {alice.id, bob.id})
        self.assertEqual({r.claim_status for r in results.values()}, {"current"})
        self.assertEqual(results[alice.id].links, ())
        self.assertEqual(results[bob.id].links, ())

    def test_same_speaker_corrects_own_claim_without_erasing_another(self):
        old = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say("Review is Friday.", "design", speaker="bob", memory_key="review")
        new = self.say("Review is Tuesday.", "design", memory_key="review")

        self.assertEqual(self.store.get_memory(old.id).superseded_by, new.id)
        self.assertIsNotNone(self.store.get_memory(old.id).archived_at)
        self.assertIsNone(self.store.get_memory(bob.id).archived_at)
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual(set(results), {new.id, bob.id})
        self.assertEqual({r.claim_status for r in results.values()}, {"disputed"})
        self.assertEqual([m.id for m in self.store.history(new.id)], [new.id, old.id])

    def test_correction_to_agreement_clears_carried_contradictions(self):
        for policy in LinkCarryover:
            with self.subTest(policy=policy):
                store = self.new_store(link_carryover=policy)
                store.join_channel("design", "alice")
                self.say("Review is Thursday.", "design", store=store, memory_key="review")
                bob = self.say(
                    "Review is Friday.", "design", speaker="bob",
                    store=store, memory_key="review",
                )
                corrected = self.say(
                    "Review is Friday.", "design", store=store, memory_key="review"
                )
                results = {r.memory.id: r for r in store.search("review", viewers="alice")}
                self.assertEqual(set(results), {bob.id, corrected.id})
                self.assertEqual({r.claim_status for r in results.values()}, {"current"})
                self.assertFalse(any(
                    link.relation is Relationship.CONTRADICTS
                    for result in results.values() for link in result.links
                ))

    def test_admin_resolves_dispute_and_later_nonadmin_does_not_replace_it(self):
        alice = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say("Review is Friday.", "design", speaker="bob", memory_key="review")
        self.store.add_admin("dana")
        self.assertTrue(self.store.is_admin("dana"))
        dana = self.say("Review is Thursday.", "design", speaker="dana", memory_key="review")

        for prior in (alice, bob):
            self.assertEqual(self.store.get_memory(prior.id).superseded_by, dana.id)
        self.assertEqual(
            [r.memory.id for r in self.store.search("review", viewers="alice")],
            [dana.id],
        )
        self.assertEqual(
            {m.id for m in self.store.history(dana.id)}, {alice.id, bob.id, dana.id}
        )

        later = self.say("Review is Monday.", "design", speaker="bob", memory_key="review")
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual(set(results), {dana.id, later.id})
        self.assertEqual(results[dana.id].claim_status, "current")
        self.assertEqual(results[later.id].claim_status, "overruled")
        self.assertIsNone(self.store.get_memory(dana.id).archived_at)

        agreeing = self.say(
            "Review is Thursday.", "design", speaker="erin", memory_key="review"
        )
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual(results[agreeing.id].claim_status, "agrees")
        self.assertEqual(results[dana.id].claim_status, "current")

    def test_admin_role_does_not_retroactively_promote_an_old_claim(self):
        alice = self.say("Review is Thursday.", "design", memory_key="review")
        bob = self.say("Review is Friday.", "design", speaker="bob", memory_key="review")
        self.store.add_admin("alice")
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual({r.claim_status for r in results.values()}, {"disputed"})
        self.assertFalse(self.store.get_memory(alice.id).asserted_as_admin)

        resolved = self.say("Review is Thursday.", "design", memory_key="review")
        self.assertTrue(resolved.asserted_as_admin)
        self.assertEqual(self.store.get_memory(bob.id).superseded_by, resolved.id)
        self.assertEqual(
            [r.memory.id for r in self.store.search("review", viewers="alice")],
            [resolved.id],
        )

    def test_two_admins_disagree_until_one_restates(self):
        self.store.add_admin("dana")
        self.store.add_admin("erin")
        dana = self.say("Review is Thursday.", "design", speaker="dana", memory_key="review")
        erin = self.say("Review is Friday.", "design", speaker="erin", memory_key="review")
        results = {r.memory.id: r for r in self.store.search("review", viewers="alice")}
        self.assertEqual({r.claim_status for r in results.values()}, {"disputed"})
        self.assertEqual(set(results), {dana.id, erin.id})

        restated = self.say("Review is Thursday.", "design", speaker="dana", memory_key="review")
        self.assertEqual(self.store.get_memory(erin.id).superseded_by, restated.id)
        self.assertEqual(self.store.get_memory(dana.id).superseded_by, restated.id)
        self.assertEqual(
            [r.memory.id for r in self.store.search("review", viewers="alice")],
            [restated.id],
        )

    def test_direct_supersession_cannot_bypass_speaker_rule(self):
        alice = self.say("Review is Thursday.", "design")
        bob = self.say("Review is Friday.", "design", speaker="bob")
        with self.assertRaisesRegex(ValueError, "higher-authority holder"):
            self.store.relate(bob.id, alice.id, Relationship.SUPERSEDES)
        self.store.add_admin("bob")
        with self.assertRaisesRegex(ValueError, "higher-authority holder"):
            self.store.relate(bob.id, alice.id, Relationship.SUPERSEDES)
        admin_bob = self.say("Review is Monday.", "design", speaker="bob")
        self.store.relate(admin_bob.id, alice.id, Relationship.SUPERSEDES)
        self.assertEqual(self.store.get_memory(alice.id).superseded_by, admin_bob.id)

    def test_keyed_channel_claim_requires_speaker(self):
        with self.assertRaisesRegex(ValueError, "require a speaker"):
            self.store.add_memory("Review is Thursday.", "fact", channel="design", memory_key="review")


class GuardedRevisionTests(AudienceCase):
    def setUp(self):
        super().setUp()
        for member in ("alice", "bob", "carol", "dana"):
            self.store.join_channel("design", member)
        self.store.add_admin("dana")

    def claim(self, speaker, content, key="review", subject=None, channel="design"):
        return self.store.for_writer(speaker).add_memory(
            content, "fact", channel=channel, memory_key=key, fact_subject=subject
        )

    def statuses(self, query="review"):
        return {r.memory.id: r.claim_status for r in self.store.search(query, viewers="alice")}

    def test_bound_writer_uses_its_identity_and_current_membership(self):
        guarded = self.new_store(require_writer=True)
        guarded.join_channel("design", "alice")
        with self.assertRaisesRegex(ValueError, "bound writer"):
            guarded.add_memory("Review is Monday", "fact", channel="design", speaker="alice")
        writer = guarded.for_writer("alice")
        claim = writer.add_memory("Review is Monday", "fact", channel="design", memory_key="review")
        self.assertEqual(claim.speaker, "alice")
        self.assertFalse(hasattr(writer, "add_admin"))
        self.assertFalse(hasattr(writer, "relate"))
        with self.assertRaises(TypeError):
            writer.add_memory("x", "fact", channel="design", speaker="bob")
        with self.assertRaisesRegex(ValueError, "current channel member"):
            writer.add_memory("x", "fact", channel="board")
        guarded.leave_channel("design", "alice")
        with self.assertRaisesRegex(ValueError, "current channel member"):
            writer.add_memory("Review is Tuesday", "fact", channel="design", memory_key="review")
        self.assertEqual(guarded.get_memory(claim.id).content, "Review is Monday")

    def test_personal_owner_resolves_reports_even_against_admin(self):
        bob = self.claim("bob", "Carol prefers tea", "drink", "carol")
        dana = self.claim("dana", "Carol prefers coffee", "drink", "carol")
        self.assertEqual(set(self.statuses("Carol").values()), {"disputed"})
        carol = self.claim("carol", "Carol prefers tea", "drink", "carol")
        self.assertEqual(self.store.get_memory(bob.id).superseded_by, carol.id)
        self.assertEqual(self.store.get_memory(dana.id).superseded_by, carol.id)
        later = self.claim("dana", "Carol prefers coffee", "drink", "carol")
        self.assertEqual(self.statuses("Carol"), {carol.id: "current", later.id: "overruled"})
        self.assertEqual(carol.fact_subject, "carol")

    def test_personal_subject_and_team_scope_cannot_change_for_a_key(self):
        personal = self.claim("bob", "Carol prefers tea", "drink", "carol")
        with self.assertRaisesRegex(ValueError, "scope or personal subject"):
            self.claim("dana", "Team prefers tea", "drink")
        with self.assertRaisesRegex(ValueError, "scope or personal subject"):
            self.claim("alice", "Bob prefers tea", "drink", "bob")
        with self.assertRaisesRegex(ValueError, "personal facts cannot"):
            self.store.grant_delegation("design", "drink", "bob", granted_by="dana")
        unscoped = self.claim("carol", "Carol now prefers water", "other")
        with self.assertRaisesRegex(ValueError, "same claim scope"):
            self.store.relate(unscoped.id, personal.id, Relationship.SUPERSEDES)

    def test_delegate_controls_only_granted_team_fact_and_admin_outranks(self):
        bob = self.claim("bob", "Review is Friday")
        with self.assertRaisesRegex(ValueError, "only an admin"):
            self.store.grant_delegation("design", "review", "alice", granted_by="bob")
        self.store.grant_delegation("design", "review", "alice", granted_by="dana")
        self.store.grant_delegation("design", "review", "alice", granted_by="dana")
        alice = self.claim("alice", "Review is Thursday")
        self.assertTrue(alice.asserted_as_delegate)
        self.assertEqual(self.store.get_memory(bob.id).superseded_by, alice.id)
        later = self.claim("bob", "Review is Tuesday")
        self.assertEqual(self.statuses(), {alice.id: "current", later.id: "overruled"})
        dana = self.claim("dana", "Review is Monday")
        self.assertEqual(self.store.get_memory(alice.id).superseded_by, dana.id)
        other = self.claim("alice", "Budget is high", "budget")
        self.assertFalse(other.asserted_as_delegate)

    def test_equal_delegates_dispute_until_one_restates(self):
        for member in ("alice", "bob"):
            self.store.grant_delegation("design", "review", member, granted_by="dana")
        alice = self.claim("alice", "Review is Thursday")
        bob = self.claim("bob", "Review is Friday")
        self.assertEqual(self.statuses(), {alice.id: "disputed", bob.id: "disputed"})
        resolved = self.claim("alice", "Review is Thursday")
        self.assertEqual(self.store.get_memory(bob.id).superseded_by, resolved.id)
        self.assertEqual(self.statuses(), {resolved.id: "current"})

    def test_pruned_claim_can_be_restored_beside_another_speakers_claim(self):
        alice = self.claim("alice", "Review is Thursday")
        self.store._set_activity(alice.id, created_at="2020-01-01T00:00:00+00:00")
        self.assertEqual(len(self.store.prune(dry_run=False)), 1)
        bob = self.claim("bob", "Review is Friday")
        self.store.restore(alice.id)
        self.assertEqual(self.statuses(), {alice.id: "disputed", bob.id: "disputed"})

    def test_revocation_and_departure_end_future_delegate_authority(self):
        self.store.grant_delegation("design", "review", "alice", granted_by="dana")
        earlier = self.claim("alice", "Review is Thursday")
        with self.assertRaisesRegex(ValueError, "only an admin"):
            self.store.revoke_delegation("design", "review", "alice", revoked_by="bob")
        self.store.revoke_delegation("design", "review", "alice", revoked_by="dana")
        self.assertTrue(earlier.asserted_as_delegate)
        with self.assertRaisesRegex(ValueError, "lapsed authority"):
            self.claim("alice", "Review is Friday")
        ordinary = self.store.for_writer("alice").add_memory(
            "Date update is Friday", "fact", channel="design"
        )
        with self.assertRaisesRegex(ValueError, "retained authority"):
            self.store.relate(ordinary.id, earlier.id, Relationship.SUPERSEDES)
        later = self.claim("bob", "Review is Friday")
        self.assertEqual(self.statuses(), {earlier.id: "current", later.id: "overruled"})
        self.store.grant_delegation("design", "review", "alice", granted_by="dana")
        self.store.leave_channel("design", "alice")
        self.store.join_channel("design", "alice")
        with self.assertRaisesRegex(ValueError, "lapsed authority"):
            self.claim("alice", "Review is Monday")
        with self.store._transaction(write=False) as connection:
            events = connection.execute("SELECT COUNT(*) FROM delegations").fetchone()[0]
        self.assertEqual(events, 2)


class AudienceReaderTests(AudienceCase):
    def test_reader_binds_all_read_paths_and_checks_membership_on_each_call(self):
        self.store.join_channel("design", "alice")
        self.store.join_channel("design", "bob")
        self.store.join_channel("board", "dana")
        shown = self.say("The launch is Friday.", "design", memory_key="launch")
        hidden = self.say("The launch is cancelled.", "board")
        reader = self.store.for_viewers(["bob", "alice", "bob"])

        self.assertIsInstance(reader, AudienceReader)
        self.assertEqual([r.memory.id for r in reader.search("launch")], [shown.id])
        self.assertEqual(reader.get_memory(shown.id), self.store.get_memory(shown.id))
        self.assertIsNone(reader.get_memory(hidden.id))
        self.assertEqual([m.id for m in reader.history(shown.id)], [shown.id])
        self.assertEqual(reader.history(hidden.id), [])
        self.assertEqual([m.id for m in reader.list_memories()], [shown.id])
        with self.assertRaises(TypeError):
            reader.search("launch", viewers="dana")
        self.assertFalse(hasattr(reader, "add_memory"))

        self.store.leave_channel("design", "bob")
        self.assertEqual(reader.search("launch"), [])
        self.assertIsNone(reader.get_memory(shown.id))
        self.assertEqual(reader.history(shown.id), [])
        self.assertEqual(reader.list_memories(), [])

    def test_reader_requires_nonempty_viewers(self):
        for viewers in (None, [], [" "]):
            with self.subTest(viewers=viewers), self.assertRaises(ValueError):
                self.store.for_viewers(viewers)
            with self.subTest(constructor=viewers), self.assertRaises(ValueError):
                AudienceReader(self.store, viewers)

    def test_guarded_store_rejects_every_unscoped_user_read(self):
        store = self.new_store(require_viewers=True)
        store.join_channel("design", "alice")
        memory = self.say("launch plan", "design", store=store)
        calls = (
            lambda: store.search("launch"),
            lambda: store.get_memory(memory.id),
            lambda: store.history(memory.id),
            lambda: store.list_memories(),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                ValueError, "requires viewers"
            ):
                call()
        reader = store.for_viewers("alice")
        self.assertEqual([r.memory.id for r in reader.search("launch")], [memory.id])

    def test_scoped_vector_search_never_embeds_hidden_content(self):
        class RejectHidden:
            name = "reject-hidden"
            min_similarity = 0.0

            def __init__(self):
                self.calls = []

            def __call__(self, texts):
                self.calls.extend(texts)
                if any("PRIVATE" in text for text in texts):
                    raise ValueError("hidden content reached embedder")
                return [[1.0, 0.0] for _ in texts]

        self.store.join_channel("design", "alice")
        self.store.join_channel("board", "dana")
        shown = self.say("launch plan", "design")
        self.say("PRIVATE board note", "board", speaker="dana")
        embedder = RejectHidden()
        self.store.embedder = embedder

        reader = self.store.for_viewers("alice")
        self.assertEqual(
            [r.memory.id for r in reader.search("launch", record_access=False)],
            [shown.id],
        )
        self.assertEqual(embedder.calls, ["launch plan", "launch"])
        with self.assertRaisesRegex(ValueError, "hidden content reached embedder"):
            self.store.search("launch", record_access=False)


class LineageTests(AudienceCase):
    def setUp(self):
        super().setUp()
        for channel, member in (
            ("design", "alice"), ("design", "bob"),
            ("board", "alice"), ("board", "dana"),
        ):
            self.store.join_channel(channel, member)

    def test_sources_intersect_audiences_on_every_read_path(self):
        writer = self.store.for_writer("alice")
        design = writer.add_message("Launch is Friday", channel="design")
        board = writer.add_message("Launch might slip", channel="board")
        derived = writer.add_memory(
            "Launch date is under review", "fact", channel="design",
            memory_key="launch", source_message_ids=[design.id, board.id],
        )
        visible = self.store.for_viewers("alice")
        hidden = self.store.for_viewers("bob")
        self.assertEqual(visible.get_memory(derived.id).id, derived.id)
        self.assertEqual([m.id for m in visible.sources_of(derived.id)], [design.id, board.id])
        self.assertEqual([r.memory.id for r in visible.search("launch")], [derived.id])
        self.assertIsNone(hidden.get_memory(derived.id))
        self.assertEqual(hidden.history(derived.id), [])
        self.assertEqual(hidden.list_memories(), [])
        self.assertEqual(hidden.search("launch"), [])
        self.assertEqual(hidden.sources_of(derived.id), [])
        self.assertIsNone(hidden.get_message(board.id))
        self.store.join_channel("board", "bob")
        self.assertEqual(hidden.get_memory(derived.id).id, derived.id)
        self.store.leave_channel("board", "bob")
        self.assertIsNone(hidden.get_memory(derived.id))

    def test_hidden_source_changes_no_ranking_signal_or_link(self):
        store = self.new_store(embedder=NGramHashEmbedder())
        for channel, member in (
            ("design", "alice"), ("design", "bob"), ("board", "alice")
        ):
            store.join_channel(channel, member)
        writer = store.for_writer("alice")
        visible = writer.add_memory("Launch plan approved", "fact", channel="design")
        baseline = signals(store.search("launch", viewers="bob", record_access=False))
        source = writer.add_message("Secret launch delay", channel="board")
        hidden = writer.add_memory(
            "Launch plan delayed", "fact", channel="design",
            source_message_ids=[source.id],
        )
        store.relate(hidden.id, visible.id)
        self.assertEqual(signals(store.search("launch", viewers="bob", record_access=False)), baseline)
        self.assertEqual({r.memory.id for r in store.search("launch", viewers="alice", record_access=False)},
                         {visible.id, hidden.id})

    def test_bound_writer_checks_sources_and_cannot_spoof_or_retract_others(self):
        guarded = self.new_store(require_writer=True, require_viewers=True)
        guarded.join_channel("design", "alice")
        guarded.join_channel("design", "bob")
        guarded.join_channel("board", "dana")
        with self.assertRaisesRegex(ValueError, "bound writer"):
            guarded.add_message("x", channel="design", speaker="alice")
        alice = guarded.for_writer("alice")
        bob = guarded.for_writer("bob")
        dana = guarded.for_writer("dana")
        source = alice.add_message("Launch Friday", channel="design")
        secret = dana.add_message("Board secret", channel="board")
        with self.assertRaisesRegex(ValueError, "current channel member"):
            alice.add_message("Board note", channel="board")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            alice.add_memory("Secret", "fact", channel="design", source_message_ids=[secret.id])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            alice.add_memory("Missing", "fact", channel="design", source_message_ids=["missing"])
        with self.assertRaisesRegex(ValueError, "not available for retraction"):
            bob.retract_message(source.id)
        self.assertEqual(guarded.for_viewers("bob").get_message(source.id).content, "Launch Friday")
        with self.assertRaises(ValueError):
            alice.add_memory("Empty", "fact", channel="design", source_message_ids=[])
        with self.assertRaises(ValueError):
            alice.add_memory("String", "fact", channel="design", source_message_ids=source.id)

    def test_admin_can_retract_a_message_without_channel_membership(self):
        self.store.add_admin("dana")
        source = self.store.for_writer("bob").add_message(
            "Launch note", channel="design"
        )
        self.store.retract_message(source.id, actor_id="dana")
        self.assertIsNone(self.store.get_message(source.id, viewers="bob"))
        self.assertIsNone(self.store.get_message(source.id).content)
        self.assertEqual(self.store.get_message(source.id).retracted_by, "dana")

    def test_retraction_stales_all_dependents_and_rebuild_uses_surviving_source(self):
        store = self.new_store(embedder=NGramHashEmbedder())
        store.join_channel("design", "alice")
        writer = store.for_writer("alice")
        first = writer.add_message("Launch was Friday", channel="design")
        second = writer.add_message("Launch moved to Monday", channel="design")
        claim = writer.add_memory(
            "Launch is Monday", "fact", channel="design", memory_key="launch",
            source_message_ids=[first.id, second.id],
        )
        summary = writer.add_memory(
            "Launch timing summary", "strategic_insight", channel="design",
            source_message_ids=[first.id],
        )
        independent = writer.add_memory(
            "Launch owner is Alice", "fact", channel="design",
            source_message_ids=[second.id],
        )
        with store._transaction(write=False) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 3)

        writer.retract_message(first.id)

        self.assertIsNone(store.for_viewers("alice").get_message(first.id))
        self.assertIsNone(store.for_viewers("alice").get_memory(claim.id))
        self.assertEqual(store.for_viewers("alice").history(claim.id), [])
        self.assertEqual({m.id for m in store.for_viewers("alice").list_memories()}, {independent.id})
        self.assertEqual([r.memory.id for r in store.search("launch", viewers="alice")], [independent.id])
        self.assertEqual({m.id for m in store.stale_memories()}, {claim.id, summary.id})
        self.assertEqual(store.get_message(first.id).content, None)
        self.assertIsNotNone(store.get_message(first.id).retracted_at)
        self.assertEqual([m.id for m in store.sources_of(claim.id, include_stale=True)], [first.id, second.id])
        with store._transaction(write=False) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 1)
        with self.assertRaisesRegex(ValueError, "must be rebuilt"):
            store.restore(claim.id)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            writer.add_memory(
                "Launch again", "fact", channel="design", memory_key="launch",
                source_message_ids=[first.id],
            )

        rebuilt = writer.add_memory(
            "Launch is Monday", "fact", channel="design", memory_key="launch",
            source_message_ids=[second.id],
        )
        self.assertEqual(store.get_memory(claim.id), None)
        self.assertEqual(store.get_memory(rebuilt.id).id, rebuilt.id)
        self.assertEqual([m.id for m in store.sources_of(rebuilt.id)], [second.id])
        self.assertEqual([m.id for m in store.stale_memories()], [summary.id])

    def test_retraction_does_not_carry_stale_links_to_rebuilt_memory(self):
        for policy in LinkCarryover:
            with self.subTest(policy=policy):
                store = self.new_store(link_carryover=policy)
                store.join_channel("design", "alice")
                writer = store.for_writer("alice")
                source = writer.add_message("Review Thursday", channel="design")
                surviving = writer.add_message("Review Friday", channel="design")
                old = writer.add_memory(
                    "Review Thursday", "fact", channel="design",
                    memory_key="review", source_message_ids=[source.id],
                )
                neighbor = writer.add_memory("Review notes", "fact", channel="design")
                store.relate(old.id, neighbor.id)
                writer.retract_message(source.id)
                current = writer.add_memory(
                    "Review Friday", "fact", channel="design",
                    memory_key="review", source_message_ids=[surviving.id],
                )
                results = {r.memory.id: r for r in store.search("review", viewers="alice")}
                self.assertEqual(set(results), {current.id, neighbor.id})
                self.assertFalse(any(link.relation is Relationship.RELATED_TO
                                     for link in results[current.id].links))


class UpgradeTests(AudienceCase):
    def test_version_4_database_adds_lineage_without_changing_old_memories(self):
        path = Path(self.temp_dir.name) / "v4.sqlite3"
        store = MemoryStore(path)
        legacy = store.add_memory("Launch plan", "fact", channel="design", speaker="alice")
        with store._transaction() as connection:
            connection.execute("DROP TABLE memory_sources")
            connection.execute("DROP TABLE messages")
            connection.execute("ALTER TABLE memories DROP COLUMN has_sources")
            connection.execute("ALTER TABLE memories DROP COLUMN stale_at")
            connection.execute("PRAGMA user_version = 4")
        reopened = MemoryStore(path)
        reopened.join_channel("design", "alice")
        self.assertEqual(reopened.get_memory(legacy.id).content, "Launch plan")
        self.assertEqual([r.memory.id for r in reopened.search("launch", viewers="alice")], [legacy.id])
        source = reopened.for_writer("alice").add_message("Launch plan confirmed", channel="design")
        derived = reopened.for_writer("alice").add_memory(
            "Launch confirmed", "fact", channel="design", source_message_ids=[source.id]
        )
        self.assertEqual([m.id for m in reopened.sources_of(derived.id)], [source.id])
        with reopened._transaction(write=False) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)

    def test_version_2_channel_key_index_is_rebuilt_for_speakers(self):
        path = Path(self.temp_dir.name) / "v2.sqlite3"
        store = MemoryStore(path)
        store.add_memory(
            "Review is Thursday.", "fact", channel="design",
            speaker="alice", memory_key="review",
        )
        with store._transaction() as connection:
            connection.execute("DROP INDEX one_active_memory_per_channel_key")
            connection.execute(
                """
                CREATE UNIQUE INDEX one_active_memory_per_channel_key
                ON memories(channel, memory_key)
                WHERE memory_key IS NOT NULL AND archived_at IS NULL
                  AND channel IS NOT NULL
                """
            )
            connection.execute("PRAGMA user_version = 2")

        reopened = MemoryStore(path)
        reopened.add_memory(
            "Review is Friday.", "fact", channel="design",
            speaker="bob", memory_key="review",
        )
        with reopened._transaction(write=False) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
        self.assertEqual(len(reopened.list_memories()), 2)

    def test_a_version_0_2_database_is_upgraded_in_place(self):
        path = Path(self.temp_dir.name) / "v0.2.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(VERSION_0_2_SCHEMA)
            connection.execute(
                "INSERT INTO memories (id, kind, content, memory_key, source, "
                "confidence, importance, created_at, updated_at) VALUES "
                "('old', 'fact', 'The launch is on Friday.', 'launch', 'test', 0.8, 0.5, ?, ?)",
                ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
            connection.commit()

        store = MemoryStore(path)
        MemoryStore(path)  # opening again changes nothing

        with store._transaction(write=False) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
        old = store.get_memory("old")
        self.assertEqual((old.channel, old.speaker), (None, None))
        self.assertEqual([r.memory.id for r in store.search("launch")], ["old"])
        self.assertEqual(store.search("launch", viewers="alice"), [])

        # Keys without a channel still name one active memory...
        newer = store.add_memory("The launch is on Monday.", "fact", memory_key="launch")
        self.assertEqual(store.get_memory("old").superseded_by, newer.id)
        with self.assertRaises(sqlite3.IntegrityError):
            with store._transaction() as connection:
                connection.execute(
                    "INSERT INTO memories (id, kind, content, memory_key, source, "
                    "confidence, importance, created_at, updated_at) VALUES "
                    "('dup', 'fact', 'x', 'launch', 'test', 0.8, 0.5, 'now', 'now')"
                )
        # ...and the same key can now be used inside a channel.
        store.join_channel("design", "alice")
        store.add_memory(
            "Design launch.", "fact", memory_key="launch", channel="design", speaker="alice"
        )
        self.assertEqual(
            [r.memory.content for r in store.search("launch", viewers="alice")],
            ["Design launch."],
        )


if __name__ == "__main__":
    unittest.main()
