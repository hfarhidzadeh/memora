import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_side_channel.py"
spec = importlib.util.spec_from_file_location("audience_side_channel_experiment", SCRIPT)
side_channel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = side_channel
spec.loader.exec_module(side_channel)


def trial(values):
    """A trial whose D is ``values[world][arm]`` and whose raw scores never move."""
    return {
        world: {arm: {"d": d, "lexical": 1.0} for arm, d in arms.items()}
        for world, arms in values.items()
    }


def worlds(present_after, present_before=0.0, shared=0.2, absent=0.0, present_built=0.0):
    values = {
        "none": {"after": 0.0, "before": 0.0, "built": 0.0},
        "absent": {"after": absent, "before": 0.0, "built": 0.0},
        "shared": {"after": shared, "before": shared, "built": shared},
    }
    for count in side_channel.PRESENT_COUNTS:
        values["present-{}".format(count)] = {
            "after": present_after,
            "before": present_before,
            "built": present_built,
        }
    return values


def decision(values):
    trials = [trial(values)]
    return side_channel.decide(side_channel.summarise(trials), len(trials))


def verdict(values):
    return decision(values)["verdict"]


class DecisionRuleTests(unittest.TestCase):
    def test_leak_in_after_arm_only_supports_the_hypothesis(self):
        self.assertEqual(verdict(worlds(present_after=0.1)), "H2 supported")

    def test_no_leak_does_not_support_it(self):
        self.assertEqual(verdict(worlds(present_after=0.0)), "H2 not supported")

    def test_leak_in_before_arm_does_not_support_it(self):
        self.assertEqual(
            verdict(worlds(present_after=0.1, present_before=0.1)), "H2 not supported"
        )

    def test_dead_probe_is_inconclusive(self):
        self.assertTrue(
            verdict(worlds(present_after=0.1, shared=0.0)).startswith("inconclusive")
        )

    def test_failed_negative_control_is_inconclusive(self):
        self.assertTrue(
            verdict(worlds(present_after=0.1, absent=0.1)).startswith("inconclusive")
        )

    def test_the_fix_is_judged_separately_from_the_hypothesis(self):
        good = decision(worlds(present_after=0.1))
        self.assertEqual(good["verdict_on_the_fix"], "the built fix closes the channel")

        leaky = decision(worlds(present_after=0.1, present_built=0.1))
        self.assertEqual(leaky["verdict"], "H2 supported")
        self.assertEqual(
            leaky["verdict_on_the_fix"], "the built fix does not close the channel"
        )

    def test_a_built_arm_that_differs_from_filter_before_fails_rule_4(self):
        values = worlds(present_after=0.1)
        values["shared"]["built"] = 0.3
        result = decision(values)
        self.assertFalse(result["rule_4_built_matches_before"])
        self.assertEqual(
            result["verdict_on_the_fix"], "the built fix does not close the channel"
        )


class TrialTests(unittest.TestCase):
    def test_a_trial_is_reproducible_and_covers_every_world(self):
        first = side_channel.run_trial(3)
        self.assertEqual(first, side_channel.run_trial(3))
        self.assertEqual(
            set(first),
            {"none", "absent", "shared"}
            | {"present-{}".format(count) for count in side_channel.PRESENT_COUNTS},
        )

    def test_before_arm_does_not_depend_on_hidden_memories(self):
        outcome = side_channel.run_trial(0)
        for world in ("absent", "present-1", "present-5", "present-20"):
            self.assertEqual(outcome[world]["before"], outcome["none"]["before"])
            self.assertEqual(outcome[world]["built"], outcome["none"]["built"])


if __name__ == "__main__":
    unittest.main()
