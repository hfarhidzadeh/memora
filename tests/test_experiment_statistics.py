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


if __name__ == "__main__":
    unittest.main()
