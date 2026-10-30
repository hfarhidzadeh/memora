import copy
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from dynmemory import NGramHashEmbedder
from dynmemory.evaluation import (
    CachedEmbedder,
    PermutedEmbedder,
    load_eval_set,
    load_eval_suite,
    prune_report,
    retrieval_report,
    wrong_links,
)

EVAL_SET = Path(__file__).resolve().parents[1] / "eval" / "synthetic_set.json"
WORDS = re.compile(r"\w+")


def words(text):
    return set(WORDS.findall(text.lower()))


class ListEmbedder:
    name = "list"
    min_similarity = 0.0

    def __call__(self, texts):
        return [[float(ord(text[0]))] for text in texts]


class SyntheticSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_set = load_eval_set(EVAL_SET)
        report = retrieval_report(cls.eval_set, embedder=NGramHashEmbedder(), seeds=10)
        cls.metrics = {
            configuration["name"]: configuration["metrics"]
            for configuration in report["configurations"]
        }

    def recall(self, configuration, tag):
        return self.metrics[configuration][tag]["recall"]

    def test_hard_queries_cannot_be_answered_from_shared_words(self):
        # The tags only mean something if lexical search cannot answer alone.
        memories = self.eval_set["memories"]
        for query in self.eval_set["queries"]:
            tags = set(query.get("tags", []))
            overlaps = [
                bool(words(query["query"]) & words(memories[key]["content"]))
                for key in query["relevant"]
            ]
            if tags & {"word_form", "paraphrase"}:
                self.assertFalse(any(overlaps), query["query"])
            if "needs_link" in tags:
                self.assertFalse(all(overlaps), query["query"])

    def test_correct_links_help_and_wrong_links_do_not(self):
        lexical = self.recall("lexical", "needs_link")
        graph = self.recall("lexical + graph", "needs_link")
        control = self.recall("lexical + graph, wrong links", "needs_link")
        self.assertGreater(graph, lexical)
        self.assertLess(control - lexical, 0.25 * (graph - lexical))

    def test_vectors_help_on_word_forms_and_permuted_vectors_do_not(self):
        lexical = self.recall("lexical", "word_form")
        vector = self.recall("lexical + vector", "word_form")
        control = self.recall("lexical + vector, permuted vectors", "word_form")
        self.assertEqual(lexical, 0.0)
        self.assertGreater(vector, lexical)
        self.assertLess(control - lexical, 0.25 * (vector - lexical))

    def test_character_ngrams_do_not_answer_paraphrases(self):
        # If this fails, the paraphrase queries leak surface overlap.
        self.assertEqual(self.recall("lexical + vector", "paraphrase"), 0.0)

    def test_direct_queries_need_neither_links_nor_vectors(self):
        self.assertEqual(self.recall("lexical", "direct"), 1.0)

    def test_prune_decisions_match_labels(self):
        report = prune_report(self.eval_set)
        self.assertEqual(report["correct"], len(report["cases"]), report["cases"])
        self.assertEqual((report["precision"], report["recall"]), (1.0, 1.0))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.eval_set = load_eval_set(EVAL_SET)

    def test_wrong_links_never_reuse_a_true_neighbour(self):
        true_pairs = {
            frozenset((link["source"], link["target"])) for link in self.eval_set["links"]
        }
        for seed in range(25):
            moved = wrong_links(self.eval_set, seed)
            self.assertEqual(len(moved), len(self.eval_set["links"]))
            for link in moved:
                self.assertNotIn(frozenset((link["source"], link["target"])), true_pairs)

    def test_permuted_embedder_gives_every_text_another_texts_vector(self):
        base = ListEmbedder()
        contents = ["a", "b", "c", "d", "e"]
        for seed in range(25):
            permuted = PermutedEmbedder(base, contents, seed)
            vectors = permuted(contents)
            self.assertTrue(all(vector != base([text])[0] for text, vector in zip(contents, vectors)))
            self.assertEqual(sorted(vectors), sorted(base(contents)))
            self.assertEqual(permuted(["query"]), base(["query"]))

    def test_invalid_sets_name_the_problem(self):
        for change, message in (
            (lambda data: data["queries"][0].update(relevant=["nope"]), r"queries\[0\]\.relevant"),
            (lambda data: data["queries"][0].update(query=7), r"queries\[0\]\.query"),
            (lambda data: data["queries"][0].update(tags=[["x"]]), r"queries\[0\]\.tags"),
            (lambda data: data["queries"][0].update(tags=["all"]), r"queries\[0\]\.tags.*reserved"),
            (lambda data: data["links"][0].update(relation="causes"), r"links\[0\]\.relation"),
            (lambda data: data["links"][0].update(relation="supersedes"), r"links\[0\]\.relation"),
            (lambda data: data["links"][0].update(target=data["links"][0]["source"]), r"links\[0\]"),
            (lambda data: data["memories"]["lunch"].update(kind="meal"), r"memories\.lunch\.kind"),
            (lambda data: data["memories"]["lunch"].update(content=42), r"memories\.lunch\.content"),
            (lambda data: data["prune"]["cases"]["fresh"].update(expected="maybe"), r"prune\.cases\.fresh\.expected"),
            (lambda data: data["prune"]["settings"].update(speed=2), r"prune\.settings"),
            (lambda data: data["prune"]["settings"].update(threshold="0.1"), r"prune\.settings\.threshold"),
            (lambda data: data["prune"]["settings"].update(decay_days_by_kind={"episode": "14"}), r"prune\.settings\.decay_days_by_kind"),
            (lambda data: data["prune"]["settings"].update(protect_importance=2), r"prune\.settings\.protect_importance"),
        ):
            broken = copy.deepcopy(self.eval_set)
            change(broken)
            with self.assertRaisesRegex(ValueError, message):
                load_eval_set(broken)

    def test_tags_are_counted_once_per_query(self):
        eval_set = load_eval_set(
            {
                "memories": {
                    "a": {"content": "alpha", "kind": "fact"},
                    "b": {"content": "beta", "kind": "fact"},
                },
                "queries": [
                    {"query": "alpha", "relevant": ["a"], "tags": ["x", "x"]},
                    {"query": "gamma", "relevant": ["b"]},
                ],
            }
        )
        metrics = retrieval_report(eval_set, seeds=1)["configurations"][0]["metrics"]

        self.assertEqual(
            metrics["all"], {"queries": 2, "recall": 0.5, "mrr": 0.5, "stale_first": None}
        )
        self.assertEqual(metrics["x"]["queries"], 1)

    def test_updates_apply_in_every_configuration_and_stale_answers_are_scored(self):
        eval_set = load_eval_set(
            {
                "memories": {
                    "old": {"content": "I have three cats.", "kind": "fact"},
                    "context": {"content": "Luna loves tuna.", "kind": "episode"},
                    "new": {"content": "I now have four cats at home.", "kind": "fact"},
                },
                "links": [{"source": "context", "target": "old"}],
                "updates": [{"old": "old", "new": "new"}],
                "queries": [
                    {"query": "how many cats do I have", "relevant": ["new"], "stale": ["old"]}
                ],
            }
        )

        report = retrieval_report(eval_set, seeds=1, link_carryover="topic")
        metrics = {
            configuration["name"]: configuration["metrics"]["all"]
            for configuration in report["configurations"]
        }

        self.assertEqual(report["link_carryover"], "topic")
        self.assertEqual(metrics["lexical, updates not applied"]["stale_first"], 1.0)
        for name in (
            "lexical",
            "lexical + graph (topic)",
            "lexical + graph (topic), wrong links",
        ):
            self.assertEqual(metrics[name]["stale_first"], 0.0, name)
            self.assertEqual(metrics[name]["mrr"], 1.0, name)

    def test_update_and_stale_mistakes_name_the_problem(self):
        base = {
            "memories": {
                "a": {"content": "alpha", "kind": "fact"},
                "b": {"content": "bravo", "kind": "fact"},
                "c": {"content": "charlie", "kind": "fact"},
            },
            "queries": [{"query": "alpha", "relevant": ["a"]}],
        }
        for change, message in (
            ({"updates": [{"old": "b", "new": "a"}]}, r"updates\[0\]: the new memory"),
            ({"updates": [{"old": "a", "new": "b"}, {"old": "a", "new": "c"}]}, r"updates\[1\]: 'a' is already"),
            ({"updates": [{"old": "a", "new": "zz"}]}, r"updates\[0\]\.new"),
            ({"queries": [{"query": "alpha", "relevant": ["a"], "stale": ["a"]}]}, r"queries\[0\]\.stale"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                load_eval_set(dict(copy.deepcopy(base), **change))

    def test_suites_pool_queries_and_report_bad_lines(self):
        first = {
            "memories": {"a": {"content": "alpha", "kind": "fact"}},
            "queries": [{"query": "alpha", "relevant": ["a"]}],
        }
        second = {
            "memories": {"b": {"content": "bravo", "kind": "fact"}},
            "queries": [{"query": "zulu", "relevant": ["b"]}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = Path(temp_dir) / "suite.jsonl"
            suite.write_text(json.dumps(first) + "\n\n" + json.dumps(second) + "\n", encoding="utf-8")
            eval_sets = load_eval_suite(suite)
            broken = Path(temp_dir) / "broken.jsonl"
            broken.write_text(json.dumps(first) + "\n{not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_eval_suite(broken)
            empty = Path(temp_dir) / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no evaluation sets"):
                load_eval_suite(empty)

        report = retrieval_report(eval_sets, seeds=1)
        self.assertEqual(report["sets"], 2)
        self.assertEqual(
            report["configurations"][0]["metrics"]["all"],
            {"queries": 2, "recall": 0.5, "mrr": 0.5, "stale_first": None},
        )

    def test_cached_embedder_embeds_each_text_once(self):
        calls = []

        class Counting(ListEmbedder):
            def __call__(self, texts):
                calls.append(list(texts))
                return super().__call__(texts)

        cached = CachedEmbedder(Counting())
        first = cached(["a", "b", "a"])
        second = cached(["b", "c"])

        self.assertEqual(calls, [["a", "b"], ["c"]])
        self.assertEqual(first, [[97.0], [98.0], [97.0]])
        self.assertEqual(second, [[98.0], [99.0]])

    def test_prune_report_accepts_a_naive_date(self):
        report = prune_report(self.eval_set, as_of=datetime(2026, 1, 1))
        self.assertEqual(report["correct"], len(report["cases"]))

    def test_permuted_control_needs_two_distinct_texts(self):
        with self.assertRaises(ValueError):
            PermutedEmbedder(ListEmbedder(), ["same", "same"], seed=0)


if __name__ == "__main__":
    unittest.main()
