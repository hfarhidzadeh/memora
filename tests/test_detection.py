"""Claim equivalence and update-detection baselines (mechanism checks, not evaluations)."""

import tempfile
import unittest
from pathlib import Path

from dynmemory import MemoryStore, MemoryType, NGramHashEmbedder
from dynmemory.detection import (
    EmbeddingEquivalence,
    EmbeddingUpdateDetector,
    ExactEquivalence,
    OverlapUpdateDetector,
)

FACT = MemoryType.FACT


class Fixed:
    """An embedder whose vectors are set by hand, so thresholds can be checked exactly."""

    name = "fixed"
    min_similarity = 0.0

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [self.vectors[text] for text in texts]


class EquivalenceTests(unittest.TestCase):
    def test_exact_equivalence_ignores_case_and_surrounding_space(self):
        same = ExactEquivalence()
        self.assertTrue(same(" Review is Thursday ", "review is thursday"))
        self.assertFalse(same("Review is Thursday", "Review is Friday"))

    def test_embedding_equivalence_uses_the_threshold_and_caches(self):
        embedder = Fixed({"a": [1, 0], "b": [0.95, 0.31], "c": [0, 1]})
        same = EmbeddingEquivalence(embedder, threshold=0.9)
        self.assertTrue(same("a", "b"))
        self.assertFalse(same("a", "c"))
        self.assertTrue(same("A ", "a"))  # exact match needs no vectors
        self.assertEqual(embedder.calls, 2)
        with self.assertRaises(ValueError):
            EmbeddingEquivalence(embedder, threshold=2)

    def test_a_store_uses_its_claim_equivalence_for_statuses_and_disputes(self):
        embedder = Fixed({"review thursday": [1, 0], "review on thursday": [0.99, 0.1], "review friday": [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            for equivalence, expected in ((None, {"disputed"}), (EmbeddingEquivalence(embedder, 0.9), {"current"})):
                with self.subTest(equivalence=equivalence):
                    store = MemoryStore(
                        Path(directory) / "s{}.sqlite3".format(id(equivalence)),
                        claim_equivalence=equivalence,
                    )
                    for member in ("alice", "bob"):
                        store.join_channel("design", member)
                    store.for_writer("alice").add_memory("review thursday", FACT, channel="design", memory_key="r")
                    store.for_writer("bob").add_memory("review on thursday", FACT, channel="design", memory_key="r")
                    self.assertEqual({r.claim_status for r in store.search("review", viewers="alice")}, expected)
                    self.assertEqual(len(store.disputes(viewers="alice")), 0 if equivalence else 1)


class UpdateDetectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(Path(self.directory.name) / "s.sqlite3")
        for member in ("alice", "bob"):
            self.store.join_channel("design", member)
        self.store.join_channel("board", "bob")
        writer = self.store.for_writer("bob")
        self.time = writer.add_memory("my 5k personal best is 27 minutes", FACT, channel="design")
        self.other = writer.add_memory("the office cat likes tuna treats", FACT, channel="design")
        self.hidden = writer.add_memory("my 5k personal best is 31 minutes", FACT, channel="board")

    def test_overlap_detector_picks_the_fact_with_the_same_words_and_a_new_value(self):
        found = self.store.detect_update(
            "my 5k personal best is now 25 minutes", OverlapUpdateDetector(0.5), viewers="alice"
        )
        self.assertEqual(found.id, self.time.id)

    def test_nothing_is_detected_for_an_unrelated_text_or_an_exact_repeat(self):
        detector = OverlapUpdateDetector(0.5)
        self.assertIsNone(self.store.detect_update("budget meeting moved", detector, viewers="alice"))
        self.assertIsNone(
            self.store.detect_update("my 5k personal best is 27 minutes", detector, viewers="alice")
        )

    def test_candidates_are_only_memories_the_viewers_can_see(self):
        seen = []

        def spy(text, candidates):
            seen.extend(memory.id for memory in candidates)
            return None

        self.store.detect_update("5k personal best", spy, viewers="alice")
        self.assertIn(self.time.id, seen)
        self.assertNotIn(self.hidden.id, seen)
        self.store.detect_update("5k personal best", spy, viewers="alice", channel="board")

    def test_embedding_detector_uses_its_threshold(self):
        embedder = NGramHashEmbedder()
        detector = EmbeddingUpdateDetector(embedder, threshold=0.5)
        found = self.store.detect_update("my 5k personal best is 26 minutes", detector, viewers="alice")
        self.assertEqual(found.id, self.time.id)
        strict = EmbeddingUpdateDetector(embedder, threshold=0.999)
        self.assertIsNone(self.store.detect_update("my 5k personal best is 26 minutes", strict, viewers="alice"))


if __name__ == "__main__":
    unittest.main()
