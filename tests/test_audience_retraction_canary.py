import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_retraction_canary.py"
spec = importlib.util.spec_from_file_location("audience_retraction_canary_experiment", SCRIPT)
canary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = canary
spec.loader.exec_module(canary)


def summary(trials, failed):
    return {
        "checks": {
            name: {
                "held": trials - len(failed.get(name, ())),
                "not_held_seeds": sorted(failed.get(name, ())),
            }
            for name in canary.CHECKS
        }
    }


def verdict(trials=10, **failed):
    return canary.decide(summary(trials, failed), trials)["verdict"]


class DecisionRuleTests(unittest.TestCase):
    def test_every_rule_holding_means_retraction_erases(self):
        self.assertEqual(verdict(), "retraction erases")

    def test_a_failed_control_is_inconclusive(self):
        self.assertEqual(verdict(determinism=[2]), "inconclusive (control 1 failed)")
        self.assertEqual(verdict(probe=[4]), "inconclusive (control 2 failed)")
        self.assertEqual(
            verdict(probe=[4], erasure=[1]), "inconclusive (control 2 failed)"
        )

    def test_erasure_and_specificity_failures_name_the_rules_and_first_seed(self):
        self.assertEqual(
            verdict(erasure=[7, 3]),
            "retraction does not erase (rules 4 failed, first failing seed 3)",
        )
        self.assertEqual(
            verdict(specificity=[9], erasure=[5]),
            "retraction does not erase (rules 3, 4 failed, first failing seed 5)",
        )


class TrialTests(unittest.TestCase):
    def test_a_trial_is_a_function_of_its_seed(self):
        self.assertEqual(canary.make_trial(11), canary.make_trial(11))
        self.assertEqual(canary.run_trial(11), canary.run_trial(11))
        json.dumps(canary.run_trial(11))

    def test_every_trial_retracts_a_message_that_has_a_dependent_memory(self):
        for seed in range(80):
            trial = canary.make_trial(seed)
            self.assertTrue(trial["retract"])
            for index in trial["retract"]:
                self.assertTrue(
                    any(index in memory["sources"] for memory in trial["memories"]), seed
                )
            for memory in trial["memories"]:
                for index in memory["sources"]:
                    self.assertIn(trial["messages"][index]["canary"], memory["text"])

    def test_canaries_are_unique_tokens(self):
        for seed in range(40):
            canaries = [m["canary"] for m in canary.make_trial(seed)["messages"]]
            self.assertEqual(len(canaries), len(set(canaries)))
            # No canary is a substring of another, so a scan cannot confuse them.
            for one in canaries:
                self.assertEqual(sum(one in other for other in canaries), 1, (seed, one))

    def test_the_current_store_passes_a_small_run(self):
        for seed in range(10):
            result = canary.run_trial(seed)
            for name in canary.CHECKS:
                self.assertTrue(result[name], (seed, name))
            self.assertGreater(result["dependents"], 0)

    def test_a_store_that_keeps_the_derived_text_is_detected(self):
        # Emulates the earlier store: retraction stales and archives but leaves the text.
        store_class = canary.MemoryStore
        original = store_class.retract_message

        def keep_text(store, message_id, *, actor_id):
            with store._transaction(write=False) as connection:
                kept = {
                    row["id"]: row["content"]
                    for row in connection.execute(
                        "SELECT memories.id, memories.content FROM memories "
                        "JOIN memory_sources ON memory_sources.memory_id = memories.id "
                        "WHERE memory_sources.message_id = ?",
                        (message_id,),
                    )
                }
            original(store, message_id, actor_id=actor_id)
            with store._transaction() as connection:
                for memory_id, content in kept.items():
                    connection.execute(
                        "UPDATE memories SET content = ? WHERE id = ?", (content, memory_id)
                    )

        with mock.patch.object(store_class, "retract_message", keep_text):
            results = [canary.run_trial(seed) for seed in range(10)]
        self.assertTrue(all(result["probe"] for result in results))
        self.assertEqual([r["erasure"] for r in results], [False] * 10)

    def test_a_store_that_erases_too_much_is_detected(self):
        store_class = canary.MemoryStore
        original = store_class.retract_message

        def wipe_everything(store, message_id, *, actor_id):
            original(store, message_id, actor_id=actor_id)
            with store._transaction() as connection:
                connection.execute(
                    "UPDATE memories SET content = ? WHERE stale_at IS NULL",
                    (store.RETRACTED_CONTENT,),
                )

        with mock.patch.object(store_class, "retract_message", wipe_everything):
            results = [canary.run_trial(seed) for seed in range(10)]
        self.assertTrue(any(not result["specificity"] for result in results))
        self.assertTrue(all(result["erasure"] for result in results))


class MainTests(unittest.TestCase):
    def test_main_writes_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                canary.main(["--trials", "2", "--output", str(output)])
            report = json.loads(output.with_suffix(".json").read_text())
            text = output.with_suffix(".md").read_text()
        self.assertEqual(report["trials"], 2)
        self.assertIn(report["decision"]["verdict"], text)
        self.assertEqual(printed.getvalue().strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
