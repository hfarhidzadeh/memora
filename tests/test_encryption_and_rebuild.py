"""Encryption at rest with a separate key file, and rebuilding stale memories."""

import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from dynmemory import MemoryStore, MemoryType, NGramHashEmbedder
from dynmemory.consolidation import ExtractiveConsolidator

FACT = MemoryType.FACT
CANARY = "canarypq8"


class EncryptionCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.db = self.directory / "main.sqlite3"
        self.keys = self.directory / "main.keys"

    def build(self, *, encrypted=True, **options):
        store = MemoryStore(self.db, key_path=self.keys if encrypted else None, **options)
        for member in ("alice", "bob"):
            store.join_channel("design", member)
        return store

    def derive(self, store, text, *, key=None):
        writer = store.for_writer("alice")
        message = writer.add_message(text, channel="design")
        memory = writer.add_memory(
            "summary: " + text, FACT, channel="design", memory_key=key,
            source_message_ids=[message.id],
        )
        return message, memory


class EncryptionTests(EncryptionCase):
    def test_the_main_file_holds_no_plaintext_and_reads_are_unchanged(self):
        store = self.build(embedder=NGramHashEmbedder())
        message, memory = self.derive(store, "launch code {} friday".format(CANARY), key="launch")

        self.assertNotIn(CANARY.encode(), self.db.read_bytes())
        self.assertIn(CANARY, store.get_memory(memory.id, viewers="bob").content)
        self.assertIn(CANARY, store.get_message(message.id, viewers="bob").content)
        self.assertEqual([r.memory.id for r in store.search("launch", viewers="bob")], [memory.id])
        self.assertGreater(store.search(CANARY, viewers="bob")[0].vector_score, 0)

    def test_a_copy_made_before_a_retraction_cannot_be_read_afterwards(self):
        store = self.build()
        message, memory = self.derive(store, "launch code {} friday".format(CANARY))
        backup = self.directory / "backup.sqlite3"
        shutil.copyfile(self.db, backup)

        store.for_writer("alice").retract_message(message.id)

        # The backup still holds the old rows, but their pads are gone.
        old = MemoryStore(backup, key_path=self.keys)
        self.assertEqual(old.get_message(message.id).content, MemoryStore.UNREADABLE_CONTENT)
        self.assertEqual(old.get_memory(memory.id).content, MemoryStore.UNREADABLE_CONTENT)
        with closing(sqlite3.connect(self.keys)) as connection:
            pads = {row[0] for row in connection.execute("SELECT id FROM pads")}
        self.assertNotIn("s:" + message.id, pads)
        self.assertNotIn("m:" + memory.id, pads)

    def test_without_a_key_file_the_backup_keeps_the_text(self):
        # Control: the same steps without encryption leave the text in the copy.
        store = self.build(encrypted=False)
        message, memory = self.derive(store, "launch code {} friday".format(CANARY))
        backup = self.directory / "backup.sqlite3"
        shutil.copyfile(self.db, backup)
        store.for_writer("alice").retract_message(message.id)

        self.assertIn(CANARY.encode(), backup.read_bytes())

    def test_other_memories_stay_readable_after_a_retraction(self):
        store = self.build()
        message, _ = self.derive(store, "launch code {}".format(CANARY))
        other, kept = self.derive(store, "budget is high", key="budget")
        store.for_writer("alice").retract_message(message.id)

        self.assertEqual(store.get_memory(kept.id, viewers="bob").content, "summary: budget is high")
        self.assertEqual(store.get_message(other.id, viewers="bob").content, "budget is high")

    def test_claim_resolution_compares_readable_text(self):
        store = self.build()
        store.for_writer("alice").add_memory("Review is Thursday", FACT, channel="design", memory_key="review")
        store.for_writer("bob").add_memory(" review is thursday ", FACT, channel="design", memory_key="review")
        statuses = {r.claim_status for r in store.search("review", viewers="bob")}
        self.assertEqual(statuses, {"current"})
        self.assertEqual(store.disputes(viewers="bob"), [])

    def test_existing_plain_rows_stay_readable_when_keys_are_turned_on(self):
        plain = self.build(encrypted=False)
        _, old = self.derive(plain, "written before encryption")
        store = MemoryStore(self.db, key_path=self.keys)
        _, new = self.derive(store, "written after encryption")
        self.assertEqual(store.get_memory(old.id).content, "summary: written before encryption")
        self.assertEqual(store.get_memory(new.id).content, "summary: written after encryption")
        self.assertNotIn(b"written after encryption", self.db.read_bytes())


class RebuildTests(EncryptionCase):
    def test_a_memory_is_rebuilt_from_its_remaining_sources_only(self):
        for encrypted in (False, True):
            with self.subTest(encrypted=encrypted):
                self.setUp()
                store = self.build(encrypted=encrypted)
                writer = store.for_writer("alice")
                first = writer.add_message("review was thursday {}".format(CANARY), channel="design")
                second = writer.add_message("review moved to friday", channel="design")
                stale = writer.add_memory(
                    "review is friday", FACT, channel="design", memory_key="review",
                    source_message_ids=[first.id, second.id],
                )
                writer.retract_message(first.id)

                (result,) = store.rebuild_stale(ExtractiveConsolidator())

                self.assertEqual((result.stale_id, result.reason), (stale.id, "rebuilt"))
                rebuilt = store.get_memory(result.rebuilt_id, viewers="bob")
                self.assertEqual(rebuilt.content, "review moved to friday")
                self.assertNotIn(CANARY, rebuilt.content)
                self.assertEqual(
                    (rebuilt.memory_key, rebuilt.channel, rebuilt.speaker), ("review", "design", "alice")
                )
                self.assertEqual([m.id for m in store.sources_of(rebuilt.id)], [second.id])
                self.assertEqual(store.stale_memories(), [])

    def test_a_memory_without_remaining_sources_stays_queued(self):
        store = self.build()
        message, memory = self.derive(store, "only source")
        store.for_writer("alice").retract_message(message.id)
        (result,) = store.rebuild_stale(ExtractiveConsolidator())
        self.assertEqual((result.rebuilt_id, result.reason), (None, "no remaining sources"))
        self.assertEqual([m.id for m in store.stale_memories()], [memory.id])

    def test_the_consolidator_sees_only_remaining_sources_and_may_decline(self):
        store = self.build()
        writer = store.for_writer("alice")
        gone = writer.add_message("secret {}".format(CANARY), channel="design")
        kept = writer.add_message("public note", channel="design")
        writer.add_memory("joined", FACT, channel="design", source_message_ids=[gone.id, kept.id])
        writer.retract_message(gone.id)
        seen = []

        def declining(memory, sources):
            seen.append((memory.content, [s.content for s in sources]))
            return None

        (result,) = store.rebuild_stale(declining)
        self.assertEqual(result.reason, "consolidator declined")
        self.assertEqual(seen, [(MemoryStore.RETRACTED_CONTENT, ["public note"])])

    def test_a_rebuild_keeps_the_original_authority(self):
        store = self.build()
        store.add_admin("alice")
        writer = store.for_writer("alice")
        first = writer.add_message("review thursday", channel="design")
        second = writer.add_message("confirmed thursday", channel="design")
        writer.add_memory(
            "review is thursday", FACT, channel="design", memory_key="review",
            source_message_ids=[first.id, second.id],
        )
        store.for_writer("bob").add_memory("review is friday", FACT, channel="design", memory_key="review")
        writer.retract_message(first.id)

        (result,) = store.rebuild_stale(ExtractiveConsolidator())
        self.assertTrue(store.get_memory(result.rebuilt_id).asserted_as_admin)
        statuses = {r.memory.speaker: r.claim_status for r in store.search("review", viewers="bob")}
        self.assertEqual(statuses, {"alice": "current", "bob": "overruled"})

    def test_an_unchannelled_memory_is_rebuilt_and_superseded(self):
        store = self.build()
        first = store.add_message("note one", channel="design", speaker="alice")
        second = store.add_message("note two", channel="design", speaker="alice")
        stale = store.add_memory("both notes", FACT, source_message_ids=[first.id, second.id])
        store.retract_message(first.id, actor_id="alice")
        (result,) = store.rebuild_stale(ExtractiveConsolidator())
        self.assertEqual(result.reason, "rebuilt")
        self.assertEqual(store.stale_memories(), [])
        self.assertEqual(store.get_memory(result.rebuilt_id).content, "note two")


if __name__ == "__main__":
    unittest.main()
