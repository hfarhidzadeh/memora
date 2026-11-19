import importlib.util
import json
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_disputes_who_knows.py"
spec = importlib.util.spec_from_file_location("audience_disputes_who_knows_experiment", SCRIPT)
e5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e5
spec.loader.exec_module(e5)


def summary(trials, failed):
    return {
        "checks": {
            name: {
                "held": trials - len(failed.get(name, ())),
                "not_held_seeds": sorted(failed.get(name, ())),
            }
            for name in e5.CHECKS
        }
    }


def verdict(trials=10, **failed):
    return e5.decide(summary(trials, failed), trials)["verdict"]


class DecisionRuleTests(unittest.TestCase):
    def test_all_rules_holding_passes(self):
        self.assertEqual(verdict(), "disputes and who-knows pass")

    def test_any_failed_control_is_inconclusive(self):
        self.assertEqual(verdict(acknowledgement=[1]), "inconclusive (control 3 failed)")
        self.assertEqual(verdict(presence=[1], noninterference=[2]), "inconclusive (control 4 failed)")
        self.assertEqual(verdict(power=range(6)), "inconclusive (control 2 failed)")

    def test_interference_names_the_first_seed(self):
        self.assertEqual(verdict(noninterference=[8, 3]), "they fail rule 5 (first failing seed 3)")


class ScriptTests(unittest.TestCase):
    def test_labels_do_not_depend_on_interleaving(self):
        # The bug found in the first smoke run: labels came from the world's sequence.
        for seed in range(10):
            pair = e5.make_pair(seed)
            w, h = e5.observe(pair, "W"), e5.observe(pair, "H")
            self.assertEqual(sorted(w["alice"]["who_knows"]), sorted(h["alice"]["who_knows"]))

    def test_hidden_steps_are_by_people_in_the_dm_and_come_after_the_planted_step(self):
        for seed in range(30):
            pair = e5.make_pair(seed)
            self.assertEqual(pair["visible"][0]["op"], "planted")
            for claim in pair["hidden"]:
                self.assertIn(claim["writer"], ("bob", "dana"))
                self.assertGreaterEqual(claim["position"], 1)
            for step in pair["visible"][1:]:
                self.assertIn(step.get("writer", step.get("member")), e5.VISIBLE_WRITERS)

    def test_a_trial_is_a_function_of_its_seed(self):
        self.assertEqual(e5.run_trial(4), e5.run_trial(4))
        json.dumps(e5.run_trial(4))


class FaultTests(unittest.TestCase):
    def test_the_current_store_passes_and_the_fault_is_detected(self):
        results = [e5.run_trial(seed) for seed in range(6)]
        self.assertTrue(all(all(result["checks"].values()) for result in results))
        disputes, who_knows = e5.MemoryStore.disputes, e5.MemoryStore.who_knows
        try:
            e5.install_fault("administrative")
            faulty = [e5.run_trial(seed) for seed in range(6)]
        finally:
            e5.MemoryStore.disputes, e5.MemoryStore.who_knows = disputes, who_knows
        self.assertTrue(any(not result["checks"]["noninterference"] for result in faulty))
        self.assertTrue(all(result["checks"]["acknowledgement"] for result in faulty))
        with self.assertRaises(ValueError):
            e5.install_fault("unknown")


if __name__ == "__main__":
    unittest.main()
