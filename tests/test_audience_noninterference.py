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
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_noninterference.py"
spec = importlib.util.spec_from_file_location("audience_noninterference_experiment", SCRIPT)
noninterference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = noninterference
spec.loader.exec_module(noninterference)

# A small pair (topic carry-over, no embedder, the viewer's churn, keyed
# rewrites, an accepted supersedes and hidden relates that name visible
# labels), so a whole trial takes well under a second.
FAST_SEED = 125
# A small pair in which a visible retraction removes a memory that replaced
# another, and hidden activity replaces a visible memory in the
# administrative view; every check holds on it.
EXTENSION_SEED = 20


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
        self.assertEqual(rule_values(result), [True] * 7)

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
        self.assertEqual(rule_values(result), [False, True, False, False, True, True, True])

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

    def test_a_retraction_that_leaves_traces_fails_rule_7(self):
        self.assertEqual(
            verdict(retraction=[2]), "the built read path fails rules 7 (first failing seed 2)"
        )

    def test_several_failing_rules_are_listed_with_the_earliest_seed(self):
        result = decision(noninterference=[8], soundness=[5, 9], retraction=[6])
        self.assertEqual(
            result["verdict"], "the built read path fails rules 4, 6, 7 (first failing seed 5)"
        )
        self.assertEqual(rule_values(result), [True, True, True, False, True, False, False])


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
            # Every world but N runs exactly the visible script, in its own order.
            for world in noninterference.WORLDS:
                if world == "N":
                    continue
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
            visible = {
                operation["label"]: operation["channel"]
                for operation in pair["visible"]
                if operation["op"] == "add"
            }
            for part in ("A", "B"):
                hidden = pair["hidden"][part]
                self.assertTrue(
                    all(join["person"] in noninterference.OTHERS for join in hidden["joins"])
                )
                channels, sourced = {}, set()
                for operation in hidden["operations"]:
                    if operation["op"] == "add":
                        if operation.get("narrow"):
                            # In a channel the viewers see, hidden by its source.
                            self.assertIn(operation["channel"], shown)
                            self.assertIsNotNone(operation["source"])
                            self.assertNotIn(operation["source"], shown)
                        else:
                            self.assertNotIn(operation["channel"], shown)
                            self.assertIn(operation["source"], (None, operation["channel"]))
                        self.assertTrue(operation["label"].startswith("h" + part))
                        channels[operation["label"]] = operation["channel"]
                        if operation["source"] is not None:
                            sourced.add(operation["label"])
                    elif operation["op"] == "member":
                        self.assertIn(operation["person"], noninterference.OTHERS)
                    elif operation["op"] == "retract":
                        self.assertIn(operation["label"], sourced)
                        sourced.discard(operation["label"])
                    else:
                        ends = (operation["source"], operation["target"])
                        self.assertTrue(any(end in channels for end in ends))
                        if operation["relation"] == noninterference.SUPERSEDES:
                            where = {**visible, **channels}
                            same = where[ends[0]] == where[ends[1]]
                            if operation.get("narrow"):
                                self.assertTrue(same)
                                self.assertIn(operation["target"], visible)
                            else:
                                self.assertNotEqual(same, operation["deliberate"])

    def test_visible_retractions_end_the_script(self):
        for seed in range(30):
            visible = noninterference.make_pair(seed)["visible"]
            ops = [operation["op"] for operation in visible]
            count = ops.count("retract")
            low, high = noninterference.VISIBLE_RETRACTIONS
            self.assertLessEqual(count, high)
            self.assertEqual(ops[len(ops) - count:], ["retract"] * count)
            sourced = {
                operation["label"]: operation["speaker"]
                for operation in visible
                if operation["op"] == "add" and operation["source"] is not None
            }
            retracted = [op["label"] for op in visible if op["op"] == "retract"]
            self.assertEqual(len(set(retracted)), len(retracted))
            for operation in visible[len(ops) - count:]:
                self.assertEqual(operation["speaker"], sourced[operation["label"]])
            if len(sourced) >= low:
                self.assertGreaterEqual(count, low)

    def test_world_n_leaves_out_retracted_writes_and_refused_operations(self):
        pair = noninterference.make_pair(EXTENSION_SEED)
        retracted = {op["label"] for op in pair["visible"] if op["op"] == "retract"}
        self.assertTrue(retracted)
        outcomes = ["ok"] * len(pair["visible"])
        refused = next(
            index for index, op in enumerate(pair["visible"])
            if op["op"] == "relate" and not ({op["source"], op["target"]} & retracted)
        )
        outcomes[refused] = "refused: older"
        executed = [
            operation for _, operation in noninterference.world_sequence(pair, "N", outcomes)
        ]
        self.assertNotIn(pair["visible"][refused], executed)
        for operation in executed:
            self.assertNotEqual(operation["op"], "retract")
            self.assertFalse(noninterference._names(operation, retracted))
        kept = [
            op for index, op in enumerate(pair["visible"])
            if index != refused and not noninterference._names(op, retracted)
        ]
        self.assertEqual(executed, kept)

    def test_labels_are_stable_and_do_not_depend_on_ids(self):
        pair = noninterference.make_pair(FAST_SEED)
        builds = []
        for _ in range(2):
            with noninterference.open_store(pair) as store:
                record = noninterference.build_world(store, pair, "A")
                memories = {}
                for label, memory_id in record["ids"].items():
                    memory = store.get_memory(memory_id)
                    # A retracted memory is stale, which even the administrative
                    # read withholds.
                    memories[label] = None if memory is None else (
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
        self.assertTrue(any(memory and memory[5] for memory in first_memories.values()))
        self.assertIn(None, first_memories.values())  # the pair retracts something


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
            reads["search"][linked][position] = row[:11] + (row[11][1:],) + row[12:]

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
        trial = noninterference.evaluate(pair, noninterference.observe_all(pair))

        self.assertTrue(trial["checks"]["exactness"])
        self.assertFalse(trial["checks"]["noninterference"])
        self.assertFalse(trial["checks"]["soundness"])
        self.assertIn(label, trial["details"]["soundness"])

    def test_one_extra_visible_memory_is_detected(self):
        plus = noninterference.observe(self.pair, "A+")["arms"]["built"]
        self.assertNotEqual(plus, self.reads)
        self.assertIsNotNone(noninterference.first_difference(self.reads, plus))

    def test_attribution_names_the_parts_that_differ(self):
        row = (
            "v0", "fact", "w01", None, "both", "ana", 1.0, "lexical", 2.0, 0.0, 0.5, (),
            "current",
        )
        other = (
            "v1", "fact", "w02", None, "both", "ben", 0.5, "graph", 0.0, 0.0, 0.0, (),
            "current",
        )

        def parts(left, right):
            found = noninterference.attribute({"search": [left]}, {"search": [right]})
            return {part for part, differs in found.items() if differs}

        self.assertEqual(parts([row, other], [row, other]), set())
        self.assertEqual(parts([row], [row[:8] + (2.5,) + row[9:]]), {"any", "lexical_score"})
        self.assertEqual(parts([row, other], [other, row]), {"any", "order"})
        self.assertEqual(parts([row, other], [row]), {"any", "result set", "count"})
        linked = row[:11] + ((("v0", "hA3", "supports"),),) + row[12:]
        self.assertEqual(parts([row], [linked]), {"any", "links"})
        self.assertEqual(parts([row], [row[:12] + ("disputed",)]), {"any", "claim_status"})


class VersionsModelTests(unittest.TestCase):
    """The model resolves channel memories as reads without viewers do (schema 6)."""

    @staticmethod
    def add(model, label, *, channel="both", key="k0", speaker="ana", source="both"):
        model.add(
            {"label": label, "channel": channel, "key": key, "speaker": speaker, "source": source}
        )

    def test_channel_claims_replay_rule_2(self):
        model = noninterference.Versions()
        self.add(model, "a1")
        self.add(model, "b1", speaker="ben")
        self.add(model, "a2")
        self.assertEqual(model.replaced(), {"a1": "a2"})
        self.assertEqual(set(model.active()), {"b1", "a2"})
        self.add(model, "c1", speaker="cy")  # the admin resolves both
        self.assertEqual(model.active(), ["c1"])
        self.add(model, "b2", speaker="ben")  # a later claim stands beside it
        self.assertEqual(set(model.active()), {"c1", "b2"})

    def test_retraction_reinstates_what_a_claim_replaced(self):
        model = noninterference.Versions()
        self.add(model, "a1")
        self.add(model, "b1", speaker="ben")
        self.add(model, "a2")
        self.add(model, "c1", speaker="cy")
        model.retract("c1")
        self.assertEqual(set(model.active()), {"b1", "a2"})
        model.retract("a2")
        self.assertEqual(set(model.active()), {"a1", "b1"})

    def test_explicit_supersedes_counts_while_its_source_is_not_retracted(self):
        model = noninterference.Versions()
        self.add(model, "n1", key=None)
        self.add(model, "n2", key=None)
        self.assertEqual(model.relate("n2", "n1", noninterference.SUPERSEDES), "ok")
        self.assertFalse(model.is_active("n1"))
        # Replacement is decided at read time, so a replaced memory can be linked...
        self.assertEqual(model.relate("n1", "n2", "related_to"), "ok")
        model.retract("n2")
        self.assertTrue(model.is_active("n1"))
        # ...but a retracted one cannot.
        self.assertEqual(model.relate("n1", "n2", "related_to"), "refused: not active")

    def test_relate_predictions_follow_the_store_order(self):
        model = noninterference.Versions()
        self.add(model, "claim")
        self.add(model, "ana_note", key=None)
        self.add(model, "ben_note", key=None, speaker="ben")
        self.add(model, "cy_note", key=None, speaker="cy")
        self.add(model, "elsewhere", key=None, channel="ana-only")
        supersedes = noninterference.SUPERSEDES
        self.assertEqual(model.relate("ana_note", "claim", supersedes), "refused: keyed claim")
        self.assertEqual(model.relate("ben_note", "ana_note", supersedes), "refused: other speaker")
        self.assertEqual(model.relate("elsewhere", "ana_note", supersedes), "refused: other channel")
        self.assertEqual(model.relate("ana_note", "cy_note", supersedes), "refused: other speaker")
        self.assertEqual(model.relate("cy_note", "ana_note", supersedes), "ok")
        self.add(model, "later", key=None)
        self.assertEqual(model.relate("claim", "later", supersedes), "refused: older")

    def test_memories_without_a_channel_are_replaced_when_written(self):
        model = noninterference.Versions()
        self.add(model, "old", channel=None, source=None)
        self.add(model, "new", channel=None, source=None)
        self.assertEqual(model.active(), ["new"])
        self.add(model, "plain", channel=None, key=None, source=None)
        self.assertEqual(model.relate("plain", "new", noninterference.SUPERSEDES), "ok")
        self.assertEqual(model.memories["plain"]["key"], "k0")  # the key hand-over


class ExtensionTrialTests(unittest.TestCase):
    """Version 2's checks on one pair, and each fault they were added to catch."""

    @classmethod
    def setUpClass(cls):
        cls.pair = noninterference.make_pair(EXTENSION_SEED)
        cls.trial = noninterference.evaluate(cls.pair, noninterference.observe_all(cls.pair))

    def rerun(self, **patches):
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(
                    mock.patch.object(noninterference.MemoryStore, name, value)
                )
            return noninterference.evaluate(self.pair, noninterference.observe_all(self.pair))

    def test_the_pair_passes_and_exercises_the_new_paths(self):
        self.assertTrue(all(self.trial["checks"].values()), self.trial["details"])
        operations = self.trial["operations"]
        self.assertGreater(operations["narrow_adds"], 0)
        self.assertGreater(operations["retractions_of_replacers"], 0)
        self.assertGreater(operations["replaced_in_admin_view"], 0)
        self.assertEqual(operations["hidden_unpredicted"], 0)
        self.assertTrue(self.trial["model_predicts_visible_outcomes"])

    def test_a_retraction_that_leaves_its_memory_is_detected(self):
        trial = self.rerun(retract_message=lambda self, message_id, *, actor_id: None)
        self.assertFalse(trial["checks"]["retraction"])
        # R retracts nothing either, so only rule 7 can see it.
        self.assertTrue(trial["checks"]["exactness"])

    def test_resolving_claims_from_every_memory_is_detected(self):
        # Emulates the store before read-time resolution, at read time: every
        # viewer's claims are resolved against all memories, hidden ones included.
        original = noninterference.MemoryStore._claim_view

        def administrative(store, connection, viewers, *, only=None):
            return original(store, connection, None, only=only)

        trial = self.rerun(_claim_view=administrative)
        # The hidden replacement is in B, so A against B and B's reads show it.
        self.assertFalse(trial["checks"]["noninterference"])
        self.assertFalse(trial["checks"]["soundness"])
        self.assertIn("superseded_by", trial["details"]["soundness"])

    def test_ignoring_source_channels_is_detected(self):
        def channel_only(viewers, column="memories.channel"):
            if viewers is None:
                return " AND memories.stale_at IS NULL", []
            return (
                " AND memories.stale_at IS NULL AND {} IN (SELECT channel_members.channel"
                " FROM channel_members WHERE channel_members.left_at IS NULL AND"
                " channel_members.member_id IN ({}) GROUP BY channel_members.channel"
                " HAVING COUNT(*) = ?)".format(column, ",".join("?" for _ in viewers)),
                [*viewers, len(viewers)],
            )

        trial = self.rerun(_visible=staticmethod(channel_only))
        self.assertFalse(trial["checks"]["soundness"])
        self.assertFalse(trial["checks"]["noninterference"])


class RelateTrialTests(unittest.TestCase):
    """Run 1 failed seed 3 through ``relate``'s administrative check; it is gone."""

    SEED = 3

    def test_the_seed_passes_with_relate_checking_only_its_ends(self):
        trial = noninterference.run_trial(self.SEED)
        self.assertTrue(all(trial["checks"].values()), trial["details"])
        self.assertEqual(trial["operations"]["visible_dependent"], 0)

    def test_restoring_the_administrative_check_is_detected(self):
        store_class = noninterference.MemoryStore
        original = store_class.relate

        def administrative(store, source_id, target_id, relation="related_to"):
            with store._transaction(write=False) as connection:
                replaced = store._claim_view(
                    connection, None, only=[source_id, target_id]
                ).replaced
            if source_id in replaced or target_id in replaced:
                raise ValueError("both memories must exist and be active")
            return original(store, source_id, target_id, relation)

        with mock.patch.object(store_class, "relate", administrative):
            trial = noninterference.run_trial(self.SEED)
        self.assertFalse(trial["checks"]["noninterference"])
        self.assertIn("got refused: not active", trial["details"]["noninterference"])


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
