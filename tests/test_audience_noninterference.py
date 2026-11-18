import contextlib
import copy
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_noninterference.py"
spec = importlib.util.spec_from_file_location("audience_noninterference_experiment", SCRIPT)
noninterference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = noninterference
spec.loader.exec_module(noninterference)

# A small pair (topic carry-over, no embedder, the viewer's churn, keyed
# rewrites, an accepted supersedes and hidden relates that name visible
# labels), so a whole trial takes well under a second.
FAST_SEED = 125


def summary(trials, failed):
    """A synthetic summary: every check held in every pair but the seeds in ``failed``."""
    return {
        "checks": {
            name: {
                "held": trials - len(failed.get(name, ())),
                "not_held_seeds": sorted(failed.get(name, ())),
            }
            for name in noninterference.CHECKS
        }
    }


def decision(trials=10, **failed):
    return noninterference.decide(summary(trials, failed), trials)


def verdict(trials=10, **failed):
    return decision(trials, **failed)["verdict"]


def rule_values(result):
    return [
        result["rule_{}_{}".format(number, name)]
        for number, name in noninterference.RULES.items()
    ]


class DecisionRuleTests(unittest.TestCase):
    def test_every_rule_holding_passes(self):
        result = decision()
        self.assertEqual(result["verdict"], "the built read path passes")
        self.assertEqual(rule_values(result), [True] * 6)

    def test_a_rebuild_that_differs_is_inconclusive(self):
        self.assertEqual(verdict(determinism=[4]), "inconclusive (control 1 failed)")

    def test_a_comparison_blind_to_an_extra_memory_is_inconclusive(self):
        self.assertEqual(verdict(sensitivity=[0]), "inconclusive (control 2 failed)")

    def test_power_needs_at_least_half_of_the_pairs(self):
        self.assertEqual(verdict(power=range(6)), "inconclusive (control 3 failed)")
        self.assertEqual(verdict(power=range(5)), "the built read path passes")

    def test_failed_controls_are_all_named_and_outrank_the_built_rules(self):
        result = decision(determinism=[1], power=range(10), noninterference=[2])
        self.assertEqual(result["verdict"], "inconclusive (control 1, 3 failed)")
        self.assertEqual(rule_values(result), [False, True, False, False, True, True])

    def test_interference_fails_rule_4_with_the_first_failing_seed(self):
        self.assertEqual(
            verdict(noninterference=[7, 3]),
            "the built read path fails rules 4 (first failing seed 3)",
        )

    def test_inexactness_fails_rule_5(self):
        self.assertEqual(
            verdict(exactness=[9]), "the built read path fails rules 5 (first failing seed 9)"
        )

    def test_unsoundness_fails_rule_6(self):
        self.assertEqual(
            verdict(soundness=[0]), "the built read path fails rules 6 (first failing seed 0)"
        )

    def test_several_failing_rules_are_listed_with_the_earliest_seed(self):
        result = decision(noninterference=[8], soundness=[5, 9])
        self.assertEqual(
            result["verdict"], "the built read path fails rules 4, 6 (first failing seed 5)"
        )
        self.assertEqual(rule_values(result), [True, True, True, False, True, False])


class ScriptTests(unittest.TestCase):
    def test_the_visible_script_is_identical_whatever_the_hidden_script_is(self):
        original = noninterference.hidden_script

        def elsewhere(seed, world, viewers, churn, labels):
            return original(seed + 1000, world, viewers, churn, labels)

        for seed in range(6):
            pair = noninterference.make_pair(seed)
            self.assertNotEqual(pair["hidden"]["A"], pair["hidden"]["B"])
            noninterference.hidden_script = elsewhere
            try:
                other = noninterference.make_pair(seed)
            finally:
                noninterference.hidden_script = original
            self.assertNotEqual(other["hidden"], pair["hidden"])
            for part in ("visible", "reads", "extra", "viewers", "churn", "carryover"):
                self.assertEqual(other[part], pair[part], (seed, part))
            # Every world runs exactly the visible script, in its own order.
            for world in noninterference.WORLDS:
                executed = [
                    operation
                    for source, operation in noninterference.world_sequence(pair, world)
                    if source == "visible"
                ]
                extra = [pair["extra"]] if world == "A+" else []
                self.assertEqual(executed, pair["visible"] + extra, (seed, world))

    def test_hidden_operations_stay_out_of_sight(self):
        for seed in range(30):
            pair = noninterference.make_pair(seed)
            shown = set(noninterference.visible_channels(pair["viewers"]))
            for part in ("A", "B"):
                hidden = pair["hidden"][part]
                self.assertTrue(
                    all(join["person"] in noninterference.OTHERS for join in hidden["joins"])
                )
                channels = {}
                for operation in hidden["operations"]:
                    if operation["op"] == "add":
                        self.assertNotIn(operation["channel"], shown)
                        self.assertTrue(operation["label"].startswith("h" + part))
                        channels[operation["label"]] = operation["channel"]
                    elif operation["op"] == "member":
                        self.assertIn(operation["person"], noninterference.OTHERS)
                    else:
                        ends = (operation["source"], operation["target"])
                        self.assertTrue(any(end in channels for end in ends))
                        if operation["relation"] == noninterference.SUPERSEDES:
                            same = all(end in channels for end in ends) and (
                                channels[ends[0]] == channels[ends[1]]
                            )
                            self.assertNotEqual(same, operation["deliberate"])

    def test_labels_are_stable_and_do_not_depend_on_ids(self):
        pair = noninterference.make_pair(FAST_SEED)
        builds = []
        for _ in range(2):
            with noninterference.open_store(pair) as store:
                record = noninterference.build_world(store, pair, "A")
                memories = {}
                for label, memory_id in record["ids"].items():
                    memory = store.get_memory(memory_id)
                    memories[label] = (
                        memory.content,
                        memory.kind,
                        memory.channel,
                        memory.speaker,
                        memory.memory_key,
                        memory.archived_at is not None,
                        record["labels"].get(memory.superseded_by),
                    )
                builds.append((record, memories))
        (first, first_memories), (second, second_memories) = builds
        self.assertEqual(first_memories, second_memories)
        self.assertTrue(
            all(first["ids"][label] != second["ids"][label] for label in first["ids"])
        )
        visible = [op["label"] for op in pair["visible"] if op["op"] == "add"]
        self.assertEqual(visible, ["v{}".format(index) for index in range(len(visible))])
        hidden = [label for label in first["ids"] if not label.startswith("v")]
        self.assertTrue(hidden)
        self.assertTrue(all(label.startswith("hA") for label in hidden))
        self.assertTrue(any(memory[5] for memory in first_memories.values()))


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = noninterference.run_trial(FAST_SEED)

    def test_the_determinism_control_passes_on_one_pair(self):
        self.assertTrue(self.trial["checks"]["determinism"])
        self.assertNotIn("determinism", self.trial["details"])

    def test_a_trial_is_a_function_of_its_seed(self):
        self.assertEqual(noninterference.run_trial(FAST_SEED), self.trial)

    def test_a_trial_reports_every_check_and_counts_skips(self):
        self.assertEqual(set(self.trial["checks"]), set(noninterference.CHECKS))
        self.assertTrue(self.trial["model_predicts_visible_outcomes"])
        outcomes = self.trial["operations"]["hidden_outcomes"]
        self.assertGreater(outcomes.get("relate: skipped: not written yet", 0), 0)
        self.assertGreater(self.trial["operations"]["cross_channel_attempted"], 0)
        json.dumps(self.trial)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = noninterference.make_pair(FAST_SEED)
        cls.seen = noninterference.observe(cls.pair, "A")
        cls.reads = cls.seen["arms"]["built"]

    def changed(self, edit):
        reads = copy.deepcopy(self.reads)
        edit(reads)
        self.assertNotEqual(reads, self.reads)
        return noninterference.first_difference(self.reads, reads)

    def search_with(self, test):
        return next(
            index for index, rows in enumerate(self.reads["search"]) if test(rows)
        )

    def test_a_one_ulp_change_in_a_score_is_detected(self):
        index = self.search_with(lambda rows: rows)

        def edit(reads):
            row = list(reads["search"][index][0])
            row[8] = math.nextafter(row[8], math.inf)
            reads["search"][index][0] = tuple(row)

        self.assertEqual(
            self.changed(edit), "search {}: row 0 differs in lexical_score".format(index)
        )

    def test_a_changed_order_or_link_is_detected(self):
        index = self.search_with(lambda rows: len(rows) > 1)

        def swap(reads):
            rows = reads["search"][index]
            rows[0], rows[1] = rows[1], rows[0]

        self.assertEqual(self.changed(swap), "search {}: row 0 differs in label".format(index))

        linked = self.search_with(lambda rows: any(row[11] for row in rows))
        position = next(
            number for number, row in enumerate(self.reads["search"][linked]) if row[11]
        )

        def drop_link(reads):
            row = reads["search"][linked][position]
            reads["search"][linked][position] = row[:11] + (row[11][1:],)

        self.assertEqual(
            self.changed(drop_link),
            "search {}: row {} differs in links".format(linked, position),
        )

    def test_changed_memories_lists_and_histories_are_detected(self):
        def unarchive(reads):
            row = reads["list all"][0]
            reads["list all"][0] = row[:8] + (not row[8],) + row[9:]

        self.assertEqual(self.changed(unarchive), "list all: row 0 differs in archived")

        def hide(reads):
            reads["get"]["v0"] = None

        self.assertEqual(self.changed(hide), "get v0: row 0: a memory against nothing")

        def forget(reads):
            reads["history"]["v0"] = []

        self.assertEqual(
            self.changed(forget),
            "history v0: {} rows against 0".format(len(self.reads["history"]["v0"])),
        )

    def test_the_lists_are_compared_sorted_by_label(self):
        for name in ("list active", "list all"):
            labels = [row[0] for row in self.reads[name]]
            self.assertEqual(labels, sorted(labels))

    def test_the_after_arms_show_only_visible_results(self):
        for rows in self.seen["arms"]["after-naive"]["search"]:
            self.assertTrue(all(row[0].startswith("v") for row in rows))
        for rows in self.seen["arms"]["after-careful"]["search"]:
            for row in rows:
                self.assertTrue(row[0].startswith("v"))
                for source, target, _ in row[11]:
                    self.assertTrue(source.startswith("v") and target.startswith("v"))

    def test_an_injected_leak_is_detected_end_to_end(self):
        pair = copy.deepcopy(self.pair)
        label = "hB{}".format(
            sum(op["op"] == "add" for op in pair["hidden"]["B"]["operations"])
        )
        # A "hidden" add that lands in a channel the viewers can see.
        pair["hidden"]["B"]["operations"].append(
            {
                "op": "add",
                "label": label,
                "channel": noninterference.visible_channels(pair["viewers"])[0],
                "kind": "fact",
                "content": "w00 w01",
                "key": None,
                "speaker": "ana",
                "expect": "ok",
            }
        )
        pair["order"]["B"].append("hidden")
        seen = {world: noninterference.observe(pair, world) for world in noninterference.WORLDS}
        trial = noninterference.evaluate(pair, seen)

        self.assertTrue(trial["checks"]["exactness"])
        self.assertFalse(trial["checks"]["noninterference"])
        self.assertFalse(trial["checks"]["soundness"])
        self.assertIn(label, trial["details"]["soundness"])

    def test_one_extra_visible_memory_is_detected(self):
        plus = noninterference.observe(self.pair, "A+")["arms"]["built"]
        self.assertNotEqual(plus, self.reads)
        self.assertIsNotNone(noninterference.first_difference(self.reads, plus))

    def test_attribution_names_the_parts_that_differ(self):
        row = ("v0", "fact", "w01", None, "both", "ana", 1.0, "lexical", 2.0, 0.0, 0.5, ())
        other = ("v1", "fact", "w02", None, "both", "ben", 0.5, "graph", 0.0, 0.0, 0.0, ())

        def parts(left, right):
            found = noninterference.attribute({"search": [left]}, {"search": [right]})
            return {part for part, differs in found.items() if differs}

        self.assertEqual(parts([row, other], [row, other]), set())
        self.assertEqual(parts([row], [row[:8] + (2.5,) + row[9:]]), {"any", "lexical_score"})
        self.assertEqual(parts([row, other], [other, row]), {"any", "order"})
        self.assertEqual(parts([row, other], [row]), {"any", "result set", "count"})
        linked = row[:11] + ((("v0", "hA3", "supports"),),)
        self.assertEqual(parts([row], [linked]), {"any", "links"})


class MainTests(unittest.TestCase):
    def test_main_writes_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                noninterference.main(["--trials", "1", "--output", str(output)])
            report = json.loads(output.with_suffix(".json").read_text())
            text = output.with_suffix(".md").read_text()
        self.assertEqual(report["trials"], 1)
        self.assertIn(report["decision"]["verdict"], text)
        self.assertEqual(printed.getvalue().strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
