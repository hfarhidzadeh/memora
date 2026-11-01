import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "longmemeval_knowledge_update.py"
spec = importlib.util.spec_from_file_location("knowledge_update_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class SignFlipTests(unittest.TestCase):
    def test_exact_test_matches_hand_counts(self):
        # Six equal positive changes: only all-plus and all-minus reach |sum| = 6.
        self.assertEqual(experiment.sign_flip_p([0.5] * 6 + [0.0] * 40), 2 / 64)
        self.assertEqual(experiment.sign_flip_p([1.0, -1.0]), 1.0)
        self.assertEqual(experiment.sign_flip_p([0.0, 0.0]), 1.0)

    def test_random_flips_agree_with_the_exact_test(self):
        changes = [0.3, 0.2, -0.1, 0.25, 0.4, -0.05, 0.1, 0.15, 0.2, 0.3, -0.2, 0.1, 0.05, 0.3, 0.2, 0.1, 0.12]
        exact_limit = experiment.EXACT_FLIP_LIMIT
        try:
            experiment.EXACT_FLIP_LIMIT = len(changes)
            exact = experiment.sign_flip_p(changes)
            experiment.EXACT_FLIP_LIMIT = 0
            sampled = experiment.sign_flip_p(changes)
        finally:
            experiment.EXACT_FLIP_LIMIT = exact_limit
        self.assertAlmostEqual(sampled, exact, delta=0.003)

    def test_paired_changes_summarise_direction_and_interval(self):
        result = experiment.paired_changes([0.5, 0.5, 0.0, -0.25])

        self.assertEqual((result["better"], result["worse"], result["same"]), (2, 1, 1))
        self.assertAlmostEqual(result["mean"], 0.1875)
        self.assertLessEqual(result["ci95"][0], result["mean"])
        self.assertGreaterEqual(result["ci95"][1], result["mean"])

    def test_runs_over_different_questions_cannot_be_compared(self):
        with self.assertRaises(ValueError):
            experiment.changes_between([{"set": "a", "rr": 1.0}], [{"set": "b", "rr": 1.0}])


SILENCE_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "longmemeval_silence.py"
silence_spec = importlib.util.spec_from_file_location("silence_experiment", SILENCE_SCRIPT)
silence = importlib.util.module_from_spec(silence_spec)
silence_spec.loader.exec_module(silence)


class Result:
    def __init__(self, key, lexical=0.0, vector=0.0, coverage=0.0):
        self.memory = type("Memory", (), {"id": key})()
        self.lexical_score, self.vector_score, self.coverage = lexical, vector, coverage


class SilenceStatisticsTests(unittest.TestCase):
    def test_threshold_keeps_at_least_the_requested_share(self):
        values = [float(v) for v in range(1, 101)]
        # (1 - 0.9) * 100 is 9.999999999999998 in floating point; the cut must still be 10.
        for retention, threshold in ((0.9, 11.0), (0.8, 21.0), (0.95, 6.0)):
            self.assertEqual(silence.threshold_for(values, retention), threshold)
            self.assertEqual(
                sum(value >= threshold for value in values), round(retention * 100)
            )

    def test_threshold_with_tied_values_keeps_at_least_the_share(self):
        values = [0.0] * 30 + [1.0] * 70
        threshold = silence.threshold_for(values, 0.9)
        self.assertEqual(threshold, 0.0)
        self.assertGreaterEqual(sum(value >= threshold for value in values), 90)

    def test_threshold_picks_the_right_position_among_ties(self):
        values = [0.0] * 5 + [1.0] * 5 + [2.0] * 90

        # 10% of 100 values may fail: the cut sits at position 10, the first 2.0.
        self.assertEqual(silence.threshold_for(values, 0.9), 2.0)
        self.assertEqual(silence.threshold_for(values, 0.95), 1.0)
        self.assertEqual(silence.threshold_for(values, 1.0), 0.0)

    def test_auc_counts_ties_as_half(self):
        self.assertEqual(silence.auc([3, 2], [1, 2]), 0.875)
        self.assertEqual(silence.auc([1, 1], [1, 1]), 0.5)
        self.assertEqual(silence.auc([5], [1, 2, 3]), 1.0)

    def test_sign_test_matches_hand_counts(self):
        # 3 higher, 1 lower, 1 tie: P(X >= 3 of 4) = 5/16, two-sided 10/16.
        self.assertEqual(
            silence.sign_test([2, 2, 2, 1, 1], [1, 1, 1, 2, 1]), (3, 1, 1, 0.625)
        )
        # Six of six in one direction: 2 / 2**6.
        self.assertEqual(silence.sign_test([1] * 6, [0] * 6)[3], 2 / 64)
        self.assertEqual(silence.sign_test([1, 2], [1, 2]), (0, 0, 2, 1.0))

    def test_paired_auc_and_interval(self):
        self.assertEqual(silence.paired_auc([1, 2, 3], [0, 2, 4]), 0.5)
        low, high = silence.paired_auc_interval([1, 2, 3, 4], [0, 1, 2, 3])
        self.assertEqual((low, high), (1.0, 1.0))
        # One win and one loss: resampled pairs give 0, 0.5 or 1, so the interval is
        # the full range, not the point estimate 0.5.
        self.assertEqual(silence.paired_auc_interval([1, 0], [0, 1]), (0.0, 1.0))

    def test_wilson_interval(self):
        low, high = silence.wilson(15, 30)
        self.assertAlmostEqual(low, 0.3315, places=3)
        self.assertAlmostEqual(high, 0.6685, places=3)
        self.assertEqual(silence.wilson(0, 29)[0], 0.0)
        self.assertEqual(silence.wilson(0, 0), (0.0, 1.0))

    def test_gate_values_take_the_best_value_per_signal(self):
        results = [Result("a", 2.0, 0.1, 0.5), Result("b", 1.0, 0.7, 0.9)]
        self.assertEqual(
            silence.gate_values(results), {"lexical": 2.0, "coverage": 0.9, "vector": 0.7}
        )
        self.assertEqual(
            silence.gate_values([]), {"lexical": 0.0, "coverage": 0.0, "vector": 0.0}
        )

    def test_evidence_hit_looks_only_at_the_top_by_that_signal(self):
        key_of = {str(i): "turn{}".format(i) for i in range(8)}
        results = [Result(str(i), lexical=10.0 - i, vector=float(i)) for i in range(8)]

        # Lexical ranking is 0..7; the top five are turn0..turn4.
        self.assertTrue(silence.evidence_hit(results, key_of, {"turn4"}, "lexical_score"))
        self.assertFalse(silence.evidence_hit(results, key_of, {"turn5"}, "lexical_score"))
        # Vector ranking is the reverse: turn7..turn3.
        self.assertTrue(silence.evidence_hit(results, key_of, {"turn3"}, "vector_score"))
        self.assertFalse(silence.evidence_hit(results, key_of, {"turn2"}, "vector_score"))
        self.assertIsNone(silence.evidence_hit(results, key_of, set(), "lexical_score"))
        # A memory the signal did not match is never a hit.
        self.assertFalse(
            silence.evidence_hit([Result("0", lexical=0.0)], {"0": "turn0"}, {"turn0"}, "lexical_score")
        )


def toy_question(question_id, question_type, *, evidence=True, sessions=("s",), answers=("a",)):
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "question " + question_id,
        "haystack_session_ids": list(sessions),
        "haystack_dates": ["2023/05/25 (Thu) 20:21"] * len(sessions),
        "haystack_sessions": [
            [{"role": "user", "content": "turn " + question_id, "has_answer": evidence}]
            for _ in sessions
        ],
        "answer_session_ids": list(answers),
    }


class SilenceDesignTests(unittest.TestCase):
    def test_calibration_sample_is_type_matched_and_excludes_twins_and_unanswerable(self):
        questions = [
            toy_question("{}-{}".format(kind, index), kind)
            for kind in silence.TYPES
            for index in range(4)
        ]
        questions += [
            toy_question("multi-session-0_abs", "multi-session"),
            toy_question("multi-session-blank", "multi-session", evidence=False),
            toy_question("other-0", "single-session-preference"),
        ]
        original = dict(silence.CALIBRATION_PER_TYPE)
        try:
            silence.CALIBRATION_PER_TYPE.update({kind: 2 for kind in silence.TYPES})
            sample, pool = silence.calibration_sample(questions, {"multi-session-0"}, seed=0)
            repeat, _ = silence.calibration_sample(questions, {"multi-session-0"}, seed=0)
        finally:
            silence.CALIBRATION_PER_TYPE.update(original)

        ids = [q["question_id"] for q in sample]
        self.assertEqual(ids, [q["question_id"] for q in repeat])
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(
            [q["question_type"] for q in sample],
            [kind for kind in silence.TYPES for _ in range(2)],
        )
        self.assertNotIn("multi-session-0", {q["question_id"] for q in pool})
        self.assertEqual(len(pool), 15)  # 16 typed questions, minus the twin; blanks and _abs left out

    def test_foreign_queries_never_use_a_question_whose_evidence_is_in_the_store(self):
        twins = [
            toy_question("q1", "multi-session", sessions=("a1", "shared"), answers=("a1",)),
            toy_question("q2", "multi-session", sessions=("a2",), answers=("a2",)),
            toy_question("q3", "multi-session", sessions=("a3",), answers=("shared",)),
        ]

        for seed in range(20):
            queries = silence.foreign_queries(twins, seed=seed)
            self.assertNotEqual(queries["q1"], "question q1")
            self.assertEqual(queries["q1"], "question q2")  # q3's evidence is in q1's store
            self.assertNotEqual(queries["q2"], "question q2")
            self.assertNotEqual(queries["q3"], "question q3")

    def test_random_word_queries_match_the_question_length(self):
        twins = [toy_question("q1", "multi-session")]
        twins[0]["question"] = "What is my name today"
        vocabulary = ["w{}".format(i) for i in range(50)]

        queries = silence.random_word_queries(twins, vocabulary, seed=0)

        words = queries["q1"].split()
        self.assertEqual(len(words), 5)
        self.assertEqual(len(set(words)), 5)  # drawn without repeats
        self.assertLessEqual(set(words), set(vocabulary))
        self.assertEqual(queries, silence.random_word_queries(twins, vocabulary, seed=0))


class SilenceAnalysisTests(unittest.TestCase):
    def groups(self):
        return {
            "calibration": {"c{}".format(i): float(i) for i in range(1, 11)},
            "twin": {"a": 5.0, "b": 6.0, "c": 7.0},
            "near_miss": {"a_abs": 1.0, "b_abs": 3.0, "c_abs": 8.0},
            "foreign": {"a": 0.0, "b": 0.0, "c": 9.0},
            "negative": [([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])],
        }

    def test_gates_use_only_the_calibration_distribution(self):
        analysis = silence.analyse("lexical", self.groups(), ["c{}".format(i) for i in range(1, 11)])

        gate = analysis["gates"]["0.9"]
        self.assertEqual(gate["threshold"], 2.0)  # keeps 9 of the 10 calibration values
        self.assertEqual(gate["calibration_kept"], 0.9)
        self.assertEqual(gate["twin_kept"], 1.0)
        self.assertAlmostEqual(gate["near_miss_silenced"], 1 / 3)  # only 1.0 is below 2.0
        self.assertAlmostEqual(gate["foreign_silenced"], 2 / 3)
        self.assertEqual(analysis["gates"]["0.8"]["threshold"], 3.0)
        self.assertEqual(
            (analysis["twin_higher"], analysis["near_miss_higher"], analysis["ties"]), (2, 1, 0)
        )
        self.assertAlmostEqual(analysis["paired_auc"], 2 / 3)
        # Each twin (5, 6, 7) beats the foreign values 0 and 0 and loses to 9.
        self.assertAlmostEqual(analysis["foreign_auc"], 6 / 9.0)
        self.assertEqual(analysis["negative_control"][0]["p"], 1.0)

    def test_pairs_need_both_a_twin_and_its_near_miss(self):
        groups = self.groups()
        del groups["near_miss"]["b_abs"]
        groups["near_miss"]["lonely_abs"] = 0.0

        analysis = silence.analyse("lexical", groups, ["c{}".format(i) for i in range(1, 11)])

        self.assertEqual(analysis["pairs"], 2)
        # The unpaired near-miss question still counts toward how many are silenced:
        # threshold 2.0 silences a_abs (1.0) and lonely_abs (0.0), not c_abs (8.0).
        self.assertAlmostEqual(analysis["gates"]["0.9"]["near_miss_silenced"], 2 / 3)

    def analysis(self, **changes):
        base = {
            "foreign_auc": 0.95,
            "negative_control": [{"p": 0.5}],
            "sign_p": 0.001,
            "twin_higher": 25,
            "near_miss_higher": 4,
            "gates": {"0.9": {"near_miss_silenced": 0.6}},
        }
        base.update(changes)
        return base

    def test_verdict_follows_the_decision_rule(self):
        self.assertEqual(silence.verdict(self.analysis()), "useful gate")
        self.assertEqual(
            silence.verdict(self.analysis(foreign_auc=0.7)), "inconclusive (a control failed)"
        )
        self.assertEqual(
            silence.verdict(self.analysis(negative_control=[{"p": 0.5}, {"p": 0.01}])),
            "inconclusive (a control failed)",
        )
        self.assertEqual(
            silence.verdict(self.analysis(sign_p=0.04)), "no signal for near-miss questions"
        )
        self.assertEqual(
            silence.verdict(self.analysis(twin_higher=4, near_miss_higher=25)),
            "no signal for near-miss questions",
        )
        self.assertEqual(
            silence.verdict(self.analysis(gates={"0.9": {"near_miss_silenced": 0.4}})),
            "separates near-miss questions but silences too few",
        )

    def test_verdict_boundaries(self):
        alpha = silence.ALPHA
        # Controls: an AUC of exactly 0.80 and a control p of exactly alpha pass.
        self.assertEqual(silence.verdict(self.analysis(foreign_auc=0.80)), "useful gate")
        self.assertEqual(
            silence.verdict(self.analysis(negative_control=[{"p": alpha}])), "useful gate"
        )
        self.assertEqual(
            silence.verdict(self.analysis(foreign_auc=0.7999)), "inconclusive (a control failed)"
        )
        # Separation needs p strictly below alpha; 0.03 is significant at 0.05 but not here.
        self.assertEqual(
            silence.verdict(self.analysis(sign_p=alpha)), "no signal for near-miss questions"
        )
        self.assertEqual(
            silence.verdict(self.analysis(sign_p=0.03)), "no signal for near-miss questions"
        )
        self.assertEqual(silence.verdict(self.analysis(sign_p=alpha - 1e-9)), "useful gate")
        # Useful needs at least half of the near-miss questions silenced.
        self.assertEqual(
            silence.verdict(self.analysis(gates={"0.9": {"near_miss_silenced": 0.5}})),
            "useful gate",
        )
        self.assertEqual(
            silence.verdict(self.analysis(gates={"0.9": {"near_miss_silenced": 0.4999}})),
            "separates near-miss questions but silences too few",
        )


if __name__ == "__main__":
    unittest.main()
