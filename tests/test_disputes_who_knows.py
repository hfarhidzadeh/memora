"""Disputes table and who-knows-what (H8), each judged only on what the viewers can see."""

import tempfile
import unittest
from pathlib import Path

from dynmemory import Dispute, Knowledge, MemoryStore, MemoryType

FACT = MemoryType.FACT


class Workspace(unittest.TestCase):
    """design: alice, bob, carol, dana (admin); dm: bob, dana."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.count = 0

    def store(self, **options):
        self.count += 1
        store = MemoryStore(Path(self.directory.name) / "w{}.sqlite3".format(self.count), **options)
        for member in ("alice", "bob", "carol", "dana"):
            store.join_channel("design", member)
        for member in ("bob", "dana"):
            store.join_channel("dm", member)
        store.add_admin("dana")
        return store

    @staticmethod
    def claim(store, speaker, content, *, key="review", source="design", subject=None):
        writer = store.for_writer(speaker)
        message = writer.add_message(content, channel=source)
        memory = writer.add_memory(
            content, FACT, channel="design", memory_key=key, fact_subject=subject,
            source_message_ids=[message.id],
        )
        return message, memory

    @staticmethod
    def table(disputes):
        return [
            (d.memory_key, d.open, [(r.speaker, r.says, r.status, r.standing) for r in d.rows])
            for d in disputes
        ]


class DisputeTests(Workspace):
    def test_two_speakers_disagreeing_make_an_open_dispute(self):
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Friday")

        (dispute,) = store.disputes(viewers="carol")

        self.assertIsInstance(dispute, Dispute)
        self.assertEqual(dispute.channel, "design")
        self.assertTrue(dispute.open)
        self.assertEqual(
            [(r.speaker, r.says, r.status, r.where) for r in dispute.rows],
            [("alice", "Review is Thursday", "disputed", "design"),
             ("bob", "Review is Friday", "disputed", "design")],
        )
        self.assertLess(dispute.rows[0].when, dispute.rows[1].when)

    def test_an_admin_resolution_keeps_the_table_as_history(self):
        # The design document's example: Dana's row current, Alice's agrees, Bob's overruled.
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Friday")
        self.claim(store, "dana", "Review is Thursday")

        self.assertEqual(
            self.table(store.disputes(viewers="carol")),
            [("review", False, [
                ("alice", "Review is Thursday", "agrees", False),
                ("bob", "Review is Friday", "overruled", False),
                ("dana", "Review is Thursday", "current", True),
            ])],
        )

    def test_agreement_and_a_speakers_own_correction_are_not_disputes(self):
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", " review is thursday ")
        self.claim(store, "carol", "Budget is low", key="budget")
        self.claim(store, "carol", "Budget is high", key="budget")

        self.assertEqual(store.disputes(viewers="carol"), [])

    def test_a_hidden_claim_adds_no_row_and_no_dispute(self):
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Friday", source="dm")

        self.assertEqual(store.disputes(viewers="carol"), [])
        # Control: bob and dana can see the dm, so for them it is a dispute.
        self.assertEqual(len(store.disputes(viewers=["bob", "dana"])), 1)

    def test_a_hidden_admin_claim_does_not_close_a_visible_dispute(self):
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Friday")
        self.claim(store, "dana", "Review is Monday", source="dm")

        (dispute,) = store.disputes(viewers="carol")
        self.assertTrue(dispute.open)
        self.assertEqual({r.speaker for r in dispute.rows}, {"alice", "bob"})

    def test_filters_and_the_bound_reader(self):
        store = self.store()
        self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Friday")
        self.claim(store, "alice", "Budget is low", key="budget")
        self.claim(store, "bob", "Budget is high", key="budget")

        reader = store.for_viewers("carol")
        self.assertEqual([d.memory_key for d in reader.disputes()], ["budget", "review"])
        self.assertEqual([d.memory_key for d in reader.disputes(memory_key="budget")], ["budget"])
        self.assertEqual(reader.disputes(channel="dm"), [])

    def test_a_guarded_store_needs_an_audience(self):
        store = self.store(require_viewers=True)
        with self.assertRaisesRegex(ValueError, "requires viewers"):
            store.disputes()


class WhoKnowsTests(Workspace):
    def test_presence_is_access_and_a_reaction_is_knowing(self):
        store = self.store()
        message, memory = self.claim(store, "alice", "Review is Thursday")
        store.for_writer("bob").acknowledge(message.id, "replied")

        knowledge = store.who_knows(memory.id, viewers="carol")

        self.assertIsInstance(knowledge, Knowledge)
        self.assertEqual(knowledge.stated, ("alice",))
        self.assertEqual(knowledge.acknowledged, ("bob",))
        self.assertEqual(knowledge.had_access, ("carol", "dana"))

    def test_holding_the_same_value_counts_as_stating_it(self):
        store = self.store()
        _, memory = self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "carol", "review is thursday")
        self.claim(store, "bob", "Review is Friday")

        knowledge = store.who_knows(memory.id, viewers="dana")
        self.assertEqual(knowledge.stated, ("alice", "carol"))
        self.assertNotIn("bob", knowledge.stated)
        self.assertIn("bob", knowledge.had_access)

    def test_a_hidden_claim_does_not_make_someone_a_stater(self):
        store = self.store()
        _, memory = self.claim(store, "alice", "Review is Thursday")
        self.claim(store, "bob", "Review is Thursday", source="dm")

        self.assertNotIn("bob", store.who_knows(memory.id, viewers="carol").stated)
        # Control: for viewers who can see the dm claim, bob stated it.
        self.assertIn("bob", store.who_knows(memory.id, viewers=["bob", "dana"]).stated)

    def test_a_memory_hidden_from_the_viewers_has_no_answer(self):
        store = self.store()
        _, memory = self.claim(store, "bob", "Review is Friday", source="dm")
        self.assertIsNone(store.who_knows(memory.id, viewers="carol"))
        self.assertIsNone(store.for_viewers("carol").who_knows(memory.id))

    def test_had_access_follows_current_membership_of_every_source_channel(self):
        store = self.store()
        _, memory = self.claim(store, "bob", "Review is Friday", source="dm")
        self.assertEqual(store.who_knows(memory.id, viewers="dana").had_access, ("dana",))
        store.leave_channel("design", "dana")
        self.assertIsNone(store.who_knows(memory.id, viewers="dana"))

    def test_acknowledging_needs_a_readable_message(self):
        store = self.store()
        message = store.for_writer("bob").add_message("dm only", channel="dm")
        with self.assertRaisesRegex(ValueError, "not available to acknowledge"):
            store.for_writer("alice").acknowledge(message.id)
        with self.assertRaisesRegex(ValueError, "replied' or 'reacted"):
            store.for_writer("dana").acknowledge(message.id, "liked")
        store.for_writer("bob").retract_message(message.id)
        with self.assertRaisesRegex(ValueError, "not available to acknowledge"):
            store.for_writer("dana").acknowledge(message.id)

    def test_acknowledging_any_one_source_counts(self):
        store = self.store()
        first = store.for_writer("alice").add_message("Review is Thursday", channel="design")
        second = store.for_writer("alice").add_message("Thursday confirmed", channel="design")
        store.for_writer("bob").acknowledge(first.id)
        memory = store.for_writer("alice").add_memory(
            "Review is Thursday", FACT, channel="design", source_message_ids=[first.id, second.id]
        )
        self.assertEqual(store.who_knows(memory.id, viewers="carol").acknowledged, ("bob",))


if __name__ == "__main__":
    unittest.main()
