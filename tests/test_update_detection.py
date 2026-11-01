import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "longmemeval_update_detection.py"
spec = importlib.util.spec_from_file_location("update_detection_experiment", SCRIPT)
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)


def eval_set(size=50):
    """A stream: m10 states a value, m40 replaces it; the rest is unrelated filler."""
    memories = {
        "m{}".format(i): {"content": "common filler{} text".format(i), "kind": "episode"}
        for i in range(size)
    }
    memories["m10"]["content"] = "my personal best is twenty seven minutes"
    memories["m40"]["content"] = "my personal best is now twenty five minutes"
    return {
        "id": "q1",
        "memories": memories,
        "queries": [{"query": "personal best", "relevant": ["m40"], "tags": ["knowledge-update", "current"]}],
        "updates": [{"old": "m10", "new": "m40"}],
    }


class JudgePromptTests(unittest.TestCase):
    def test_answers_are_parsed_strictly(self):
        for raw, expected in (
            ("YES", "YES"), (" yes.", "YES"), ("Yes\n", "YES"), ("**NO**", "NO"),
            ("No, they differ", "NO"), ("Maybe", "INVALID"), ("", "INVALID"), ("Nope", "INVALID"),
        ):
            self.assertEqual(detection.parse_answer(raw), expected, raw)

    def test_long_messages_are_clipped_and_braces_survive(self):
        long_text = "word " * 400
        prompt = detection.build_prompt("{later} {earlier} " + long_text, "short")

        self.assertIn("{later} {earlier} word", prompt)
        self.assertLessEqual(len(detection.clip(long_text)), detection.MAX_CHARS + 4)
        self.assertTrue(detection.clip(long_text).endswith(" ..."))
        self.assertEqual(detection.clip("a  b\n c"), "a b c")
        self.assertTrue(prompt.rstrip().endswith("Answer with one word: YES or NO."))


class OllamaJudgeTests(unittest.TestCase):
    def test_answers_are_cached_in_memory_and_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            calls = []

            def post(prompt):
                calls.append(prompt)
                return "YES"

            judge = detection.OllamaJudge("test-model", path, post=post)
            self.assertEqual(judge("earlier", "later", label=["q", "k"]), "YES")
            self.assertEqual(judge("earlier", "later"), "YES")
            self.assertEqual((len(calls), judge.new_calls), (1, 1))

            # A new judge reads the file; a half-written last line is ignored.
            with path.open("a") as handle:
                handle.write('{"key": "cut')
            again = detection.OllamaJudge("test-model", path, post=post)
            self.assertEqual(again("earlier", "later"), "YES")
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(path.read_text().splitlines()[0])["label"], ["q", "k"])

    def test_the_model_name_is_part_of_the_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            first = detection.OllamaJudge("model-a", path, post=lambda prompt: "YES")
            first("a", "b")
            second = detection.OllamaJudge("model-b", path, post=lambda prompt: "NO")

            self.assertEqual(second("a", "b"), "NO")


class DesignTests(unittest.TestCase):
    def test_candidates_come_only_from_earlier_memories_best_first(self):
        candidates = detection.top_candidates(eval_set(), "m40", 5)

        self.assertEqual(candidates[0], "m10")
        self.assertEqual(len(candidates), 1)  # only m10 shares a word with m40
        self.assertTrue(all(int(key[1:]) < 40 for key in candidates))
        self.assertEqual(detection.top_candidates(eval_set(), "m0", 5), [])

    def test_control_turns_skip_evidence_and_the_start_of_the_stream(self):
        turns = detection.control_turns(eval_set(), {"m40", "m10", "m35"}, 4, "seed")

        self.assertEqual(len(turns), 4)
        self.assertEqual(turns, detection.control_turns(eval_set(), {"m40", "m10", "m35"}, 4, "seed"))
        for key in turns:
            self.assertGreaterEqual(int(key[1:]), detection.MIN_POSITION)
            self.assertNotIn(key, {"m40", "m10", "m35"})
        short = {"memories": {"m{}".format(i): {} for i in range(32)}}
        self.assertEqual(len(detection.control_turns(short, set(), 4, "seed")), 2)

    def test_valid_updates_keep_one_replacement_per_memory_in_stream_order(self):
        position = {"a": 0, "b": 1, "c": 2, "d": 3}

        updates = detection.valid_updates(
            [("d", "a"), ("c", "a"), ("d", "b"), ("b", "c"), ("c", "c")], position
        )

        # c replaces a before d would; b -> c and c -> c go backwards or nowhere.
        self.assertEqual(
            updates,
            [{"old": "a", "new": "c"}, {"old": "b", "new": "d"}],
        )

    def test_random_updates_replace_earlier_memories_with_the_same_newer_turns(self):
        keys = ["m{}".format(i) for i in range(50)]
        decisions = [("m40", "m10"), ("m45", "m12"), ("m0", "m0")]

        first = detection.random_updates(decisions, keys, "seed-a")
        again = detection.random_updates(decisions, keys, "seed-a")

        self.assertEqual(first, again)
        self.assertLessEqual({u["new"] for u in first}, {"m40", "m45"})  # position 0 has nothing earlier
        for update in first:
            self.assertLess(int(update["old"][1:]), int(update["new"][1:]))
        others = [detection.random_updates(decisions, keys, "seed-{}".format(i)) for i in range(10)]
        self.assertGreater(len({json.dumps(o) for o in others}), 1)


class JudgeStreamTests(unittest.TestCase):
    def test_a_stream_is_judged_at_the_evidence_turn_controls_and_random_pairs(self):
        asked = []

        def judge(earlier, later, label=None):
            asked.append(label)
            return "YES" if "twenty seven" in earlier and "twenty five" in later else "NO"

        result = detection.judge_stream(eval_set(), judge, "current")

        by_kind = {kind: [j for j in result["judged"] if j["kind"] == kind]
                   for kind in ("evidence", "control", "random")}
        self.assertEqual(result["candidates"], ["m10"])
        self.assertEqual([(j["old"], j["answer"], j["oracle"]) for j in by_kind["evidence"]],
                         [("m10", "YES", True)])
        self.assertEqual(len({j["new"] for j in by_kind["control"]}), detection.CONTROL_TURNS)
        for item in by_kind["control"]:
            self.assertNotIn(item["new"], {"m40", "m10"})
            self.assertGreaterEqual(int(item["new"][1:]), detection.MIN_POSITION)
            self.assertFalse(item["oracle"])
        self.assertEqual(len(by_kind["random"]), detection.RANDOM_PAIRS)
        for item in by_kind["random"]:
            self.assertEqual(item["new"], "m40")
            self.assertNotIn(item["old"], {"m10"})
            self.assertEqual(item["answer"], "NO")
        self.assertEqual(len(asked), len(result["judged"]))

    def test_several_newer_turns_in_the_oracle_are_rejected(self):
        broken = eval_set()
        broken["updates"].append({"old": "m11", "new": "m41"})

        with self.assertRaises(ValueError):
            detection.judge_stream(broken, lambda *args, **kwargs: "NO", "current")


class AnalysisTests(unittest.TestCase):
    def test_recovery_is_the_detectors_share_of_the_oracle_gain(self):
        result = detection.recovery([0.0] * 4, [1.0] * 4, [1.0, 0.0, 1.0, 0.0])

        self.assertEqual(result["ratio"], 0.5)
        self.assertLessEqual(result["ci95"][0], 0.5)
        self.assertGreaterEqual(result["ci95"][1], 0.5)
        self.assertIsNone(detection.recovery([0.5] * 3, [0.5] * 3, [0.7] * 3)["ratio"])

    def report(self, **changes):
        change = lambda mean, p: {"mean": mean, "p": p}
        report = {
            "pair_rates": {"random": {"rate": 0.0}, "control": {"rate": 0.02}},
            "candidate_recall": 0.9,
            "current": {
                "recovery": {"ratio": 0.7},
                "detector_vs_none": change(0.1, 0.01),
                "detector_vs_random": change(0.09, 0.01),
            },
            "costed": {"detector_vs_none": change(-0.2, 0.1), "oracle_vs_none": change(-0.25, 0.1)},
        }
        for path, value in changes.items():
            target = report
            *parents, last = path.split(".")
            for parent in parents:
                target = target[parent]
            target[last] = value
        return report

    def test_verdict_follows_the_decision_rule(self):
        self.assertEqual(detection.verdict(self.report()), "usable detector")
        self.assertTrue(detection.verdict(self.report(**{"pair_rates.random": {"rate": 0.5}})).startswith("inconclusive"))
        self.assertIn("candidate search", detection.verdict(self.report(candidate_recall=0.5)))
        self.assertIn("recovery criteria", detection.verdict(self.report(**{"current.recovery": {"ratio": 0.3}})))
        self.assertIn("recovery criteria", detection.verdict(
            self.report(**{"current.detector_vs_random": {"mean": 0.09, "p": 0.2}})))
        self.assertIn("false supersessions", detection.verdict(
            self.report(**{"pair_rates.control": {"rate": 0.2}})))
        self.assertIn("harms previous/both", detection.verdict(
            self.report(**{"costed.detector_vs_none": {"mean": -0.4, "p": 0.1}})))


if __name__ == "__main__":
    unittest.main()
