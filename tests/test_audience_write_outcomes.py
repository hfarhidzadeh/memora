import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_write_outcomes.py"
spec = importlib.util.spec_from_file_location("audience_write_outcomes_experiment", SCRIPT)
write_outcomes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = write_outcomes
spec.loader.exec_module(write_outcomes)


def summary(trials, failed):
    return {
        "checks": {
            name: {
                "held": trials - len(failed.get(name, ())),
                "not_held_seeds": sorted(failed.get(name, ())),
            }
            for name in write_outcomes.CHECKS
        }
    }


def verdict(trials=10, **failed):
    return write_outcomes.decide(summary(trials, failed), trials)["verdict"]


class DecisionRuleTests(unittest.TestCase):
    def test_every_rule_holding_passes(self):
        self.assertEqual(verdict(), "the write path passes")

    def test_a_rebuild_that_differs_is_inconclusive(self):
        self.assertEqual(verdict(determinism=[3]), "inconclusive (control 1 failed)")

    def test_power_needs_at_least_half_of_the_pairs(self):
        self.assertEqual(verdict(power=range(6)), "inconclusive (control 2 failed)")
        self.assertEqual(verdict(power=range(5)), "the write path passes")

    def test_failed_controls_outrank_interference(self):
        self.assertEqual(
            verdict(determinism=[1], power=range(10), noninterference=[2]),
            "inconclusive (control 1, 2 failed)",
        )

    def test_interference_fails_rule_3_with_the_first_failing_seed(self):
        self.assertEqual(
            verdict(noninterference=[7, 3]), "the write path fails rule 3 (first failing seed 3)"
        )


class ScriptTests(unittest.TestCase):
    def test_visible_steps_are_by_people_who_cannot_see_the_hidden_claims(self):
        for seed in range(60):
            pair = write_outcomes.make_pair(seed)
            for step in pair["visible"]:
                if step["op"] == "claim":
                    self.assertIn(step["writer"], write_outcomes.VISIBLE_WRITERS)
            for claim in pair["hidden"]:
                self.assertIn(claim["writer"], ("bob", "dana"))
                self.assertGreaterEqual(claim["position"], 1)
                if claim["writer"] == "dana":
                    self.assertIsNone(claim["subject"])

    def test_contents_are_unique_labels(self):
        for seed in range(60):
            pair = write_outcomes.make_pair(seed)
            contents = [
                step["content"] for step in pair["visible"] if step["op"] == "claim"
            ] + [claim["content"] for claim in pair["hidden"]]
            self.assertEqual(len(contents), len(set(contents)), seed)

    def test_worlds_run_the_visible_script_in_order_and_h_adds_the_hidden_claims(self):
        for seed in range(30):
            pair = write_outcomes.make_pair(seed)
            for world in write_outcomes.WORLDS:
                steps = write_outcomes.sequence(pair, world)
                self.assertEqual(
                    [step for script, step in steps if script == "visible"], pair["visible"]
                )
                hidden = [step for script, step in steps if script == "hidden"]
                self.assertEqual(len(hidden), 0 if world == "W" else len(pair["hidden"]))
                # No hidden claim runs before the first visible step (the grant).
                self.assertEqual(steps[0][0], "visible")

    def test_a_delegation_is_granted_before_any_hidden_claim_and_revoked_after(self):
        seen_revoke = False
        for seed in range(80):
            steps = write_outcomes.sequence(write_outcomes.make_pair(seed), "H")
            ops = [step["op"] for _, step in steps]
            if "grant" in ops:
                self.assertEqual(ops.index("grant"), 0)
            if "revoke" in ops:
                seen_revoke = True
                self.assertGreater(ops.index("revoke"), ops.index("grant"))
        self.assertTrue(seen_revoke)


class TrialTests(unittest.TestCase):
    def test_a_trial_is_a_function_of_its_seed_and_json_serialisable(self):
        first = write_outcomes.run_trial(5)
        self.assertEqual(write_outcomes.run_trial(5), first)
        json.dumps(first)

    def test_the_controls_hold_on_a_small_set_of_pairs(self):
        trials = [write_outcomes.run_trial(seed) for seed in range(12)]
        self.assertTrue(all(trial["checks"]["determinism"] for trial in trials))
        self.assertGreaterEqual(sum(trial["checks"]["power"] for trial in trials), 6)

    def seeds_where(self, kind, count=60):
        """Seeds whose visible script is refused for this kind of reason."""
        found = []
        for seed in range(count):
            trial = write_outcomes.run_trial(seed)
            if kind in trial["refusal_kinds"]:
                found.append(seed)
        return found

    def test_the_current_store_passes_the_pairs_that_exercise_each_leak(self):
        for kind in ("scope", "lapsed authority"):
            seeds = self.seeds_where(kind)
            self.assertTrue(seeds, kind)
            for seed in seeds[:3]:
                self.assertTrue(
                    write_outcomes.run_trial(seed)["checks"]["noninterference"], (kind, seed)
                )

    def test_a_write_check_that_sees_every_claim_is_detected(self):
        # Emulates the earlier store: bound writers' writes use the trusted view.
        store_module = sys.modules[write_outcomes.MemoryStore.__module__]
        original = write_outcomes.MemoryStore._add_memory

        def administrative(store, *args, **kwargs):
            kwargs["check_membership"] = False
            return original(store, *args, **kwargs)

        with mock.patch.object(write_outcomes.MemoryStore, "_add_memory", administrative):
            failing = [
                seed
                for seed in range(40)
                if not write_outcomes.run_trial(seed)["checks"]["noninterference"]
            ]
        self.assertGreaterEqual(len(failing), 5)
        self.assertTrue(store_module)


class MainTests(unittest.TestCase):
    def test_main_writes_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                write_outcomes.main(["--trials", "2", "--output", str(output)])
            report = json.loads(output.with_suffix(".json").read_text())
            text = output.with_suffix(".md").read_text()
        self.assertEqual(report["trials"], 2)
        self.assertIn(report["decision"]["verdict"], text)
        self.assertEqual(printed.getvalue().strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
