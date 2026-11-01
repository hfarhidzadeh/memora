import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

from dynmemory import (
    MemoryStore,
    MemoryType,
    NGramHashEmbedder,
    SentenceTransformerEmbedder,
)


class ConceptEmbedder:
    """Maps words to hand-picked concept axes, standing in for a real model."""

    CONCEPTS = {
        "sqlite": 0,
        "database": 0,
        "storage": 0,
        "concise": 1,
        "brief": 1,
        "ubuntu": 2,
        "linux": 2,
    }

    def __init__(self, name="test-concepts", min_similarity=0.5):
        self.name = name
        self.min_similarity = min_similarity
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0, 0.0, 0.0]
            for word in re.findall(r"\w+", text.lower()):
                if word in self.CONCEPTS:
                    vector[self.CONCEPTS[word]] += 1.0
            vectors.append(vector)
        return vectors


class VectorSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "memories.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def stored_models(self, store):
        with store._transaction() as connection:
            return sorted(
                row["model"] for row in connection.execute("SELECT model FROM embeddings")
            )

    def test_vector_search_finds_a_paraphrase_lexical_search_misses(self):
        lexical = MemoryStore(self.path)
        hybrid = MemoryStore(self.path, embedder=ConceptEmbedder())
        database = hybrid.add_memory("Prefer SQLite for local memory storage.", "user_preference")
        hybrid.add_memory("The user wants concise answers.", "user_preference")

        self.assertEqual(lexical.search("which database", record_access=False), [])
        results = hybrid.search("which database", record_access=False)
        self.assertEqual([result.memory.id for result in results], [database.id])
        self.assertEqual(results[0].matched_by, "vector")

    def test_matches_found_by_both_retrievers_rank_first(self):
        store = MemoryStore(self.path, embedder=ConceptEmbedder())
        both = store.add_memory("database notes", MemoryType.FACT)
        vector_only = store.add_memory("storage layer", MemoryType.FACT)
        lexical_only = store.add_memory("meeting notes", MemoryType.FACT)

        results = store.search("database notes", record_access=False)

        self.assertEqual(results[0].memory.id, both.id)
        by_id = {result.memory.id: result.matched_by for result in results}
        self.assertEqual(
            by_id,
            {
                both.id: "lexical+vector",
                vector_only.id: "vector",
                lexical_only.id: "lexical",
            },
        )

    def test_agreement_between_retrievers_outweighs_a_single_top_rank(self):
        store = MemoryStore(self.path, embedder=ConceptEmbedder())
        store.add_memory("notes notes", MemoryType.FACT)  # lexical rank 1, no vector
        store.add_memory("sqlite", MemoryType.FACT)  # vector only
        store.add_memory("storage sqlite", MemoryType.FACT)  # vector only
        both = store.add_memory("notes storage brief extra words", MemoryType.FACT)

        results = store.search("database notes", record_access=False)

        # Lexical rank 2 and vector rank 3: 1/62 + 1/63 beats 1/61.
        self.assertEqual(results[0].memory.id, both.id)
        self.assertEqual(results[0].matched_by, "lexical+vector")

    def test_memories_are_embedded_on_first_search_and_again_for_a_new_model(self):
        store = MemoryStore(self.path)
        memory = store.add_memory("Prefer SQLite for local memory storage.", "fact")
        self.assertEqual(self.stored_models(store), [])

        store.embedder = ConceptEmbedder()
        self.assertEqual(
            [result.memory.id for result in store.search("database", record_access=False)],
            [memory.id],
        )
        store.embedder = ConceptEmbedder(name="another-model")
        store.search("database", record_access=False)

        self.assertEqual(self.stored_models(store), ["another-model", "test-concepts"])
        self.assertEqual(store.embedder.calls[0], ["Prefer SQLite for local memory storage."])

    def test_min_similarity_filters_weak_vector_matches(self):
        content = "database storage brief"
        strict = MemoryStore(self.path, embedder=ConceptEmbedder(min_similarity=0.9))
        strict.add_memory(content, MemoryType.FACT)
        loose = MemoryStore(":memory:", embedder=ConceptEmbedder(min_similarity=0.5))
        loose.add_memory(content, MemoryType.FACT)

        # cosine(query, memory) = 2 / sqrt(5), about 0.894
        self.assertEqual(strict.search("sqlite", record_access=False), [])
        self.assertEqual(len(loose.search("sqlite", record_access=False)), 1)

    def test_results_report_each_matching_signal_separately(self):
        store = MemoryStore(":memory:", embedder=ConceptEmbedder(min_similarity=0.5))
        paraphrase = store.add_memory("storage backend", MemoryType.FACT)
        both = store.add_memory("sqlite storage", MemoryType.FACT)

        results = {
            result.memory.id: result for result in store.search("sqlite", record_access=False)
        }

        # "storage backend" shares no word with the query: vector only.
        self.assertEqual(results[paraphrase.id].matched_by, "vector")
        self.assertEqual(results[paraphrase.id].lexical_score, 0.0)
        self.assertEqual(results[paraphrase.id].coverage, 0.0)
        self.assertGreater(results[paraphrase.id].vector_score, 0.5)
        # "sqlite storage" shares the word and the concept: both signals.
        self.assertEqual(results[both.id].matched_by, "lexical+vector")
        self.assertGreater(results[both.id].lexical_score, 0.0)
        self.assertAlmostEqual(results[both.id].coverage, 1.0)
        self.assertAlmostEqual(results[both.id].vector_score, 1.0)

    def test_vector_score_is_zero_below_min_similarity(self):
        store = MemoryStore(":memory:", embedder=ConceptEmbedder(min_similarity=0.95))
        store.add_memory("sqlite database storage brief", MemoryType.FACT)

        # cosine(query, memory) = 3 / sqrt(10), about 0.949: under the 0.95 cutoff.
        (result,) = store.search("sqlite", record_access=False)

        self.assertEqual(result.matched_by, "lexical")
        self.assertEqual(result.vector_score, 0.0)
        self.assertGreater(result.lexical_score, 0.0)

    def test_archived_memories_are_not_vector_matches(self):
        store = MemoryStore(self.path, embedder=ConceptEmbedder())
        store.add_memory("database one", MemoryType.FACT, memory_key="k")
        store.add_memory("unrelated text", MemoryType.FACT, memory_key="k")

        self.assertEqual(store.search("storage", record_access=False), [])

    def test_embedder_returning_the_wrong_number_of_vectors_is_rejected(self):
        class NoVectors(ConceptEmbedder):
            def __call__(self, texts):
                return []

        store = MemoryStore(self.path, embedder=NoVectors())

        with self.assertRaises(ValueError):
            store.add_memory("anything", MemoryType.FACT)
        self.assertEqual(store.list_memories(), [])


class NGramHashEmbedderTests(unittest.TestCase):
    def test_vectors_are_stable_across_instances(self):
        self.assertEqual(
            NGramHashEmbedder()(["invoices"]), NGramHashEmbedder()(["invoices"])
        )

    def test_matches_word_forms_that_lexical_search_misses(self):
        lexical = MemoryStore(":memory:")
        hybrid = MemoryStore(":memory:", embedder=NGramHashEmbedder())
        for store in (lexical, hybrid):
            store.add_memory("Use the v2 billing endpoint for invoices.", MemoryType.FACT)
            store.add_memory("Lunch order was pasta.", MemoryType.EPISODE)

        self.assertEqual(lexical.search("invoice", record_access=False), [])
        results = hybrid.search("invoice", record_access=False)
        self.assertEqual(len(results), 1)
        self.assertIn("invoices", results[0].memory.content)
        self.assertEqual(results[0].matched_by, "vector")

    def test_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            NGramHashEmbedder(min_similarity=2)
        with self.assertRaises(ValueError):
            NGramHashEmbedder(dimensions=0)


class SentenceTransformerEmbedderTests(unittest.TestCase):
    @unittest.skipIf(
        importlib.util.find_spec("sentence_transformers") is not None,
        "sentence-transformers is installed",
    )
    def test_missing_package_explains_how_to_install_it(self):
        with self.assertRaisesRegex(ImportError, "pip install"):
            SentenceTransformerEmbedder()(["text"])


if __name__ == "__main__":
    unittest.main()
