import importlib.util
import json
import random
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_utility.py"
spec = importlib.util.spec_from_file_location("audience_utility_experiment", SCRIPT)
utility = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utility
spec.loader.exec_module(utility)

SEED = 1
ARMS = utility.ARMS
ORIGINAL_SEARCH = utility.MemoryStore.search
CONTENT = re.compile(r"^The (\w+) of (\w+) is (\w+)\.((?: \w+)*)$")


def summary(pooled=0.9, owner=1.0, missing=0, leaks=None, ci=(0.0, 0.0), mean=0.0):
    """A synthetic summary holding the fields ``decide`` reads (1,000 items each)."""
    counts = dict({"after-k": 0, "after-all": 0, "built": 0}, **(leaks or {}))
    counts["pooled"] = round(pooled * 1000)
    return {
        "leakage": {
            "items": 1000,
            "by_arm": {
                arm: {"leaks": count, "first_seed": 7 if count else None}
                for arm, count in counts.items()
            },
        },
        "owner": {"items": 1000, "by_arm": {"built": {"hits": round(owner * 1000)}}},
        "over_blocking": {"items": 1000, "missing": missing},
        "utility": {
            "differences": {"built_minus_after_all": {"mean": mean, "ci95": list(ci)}}
        },
    }


def decision(**options):
    return utility.decide(summary(**options), 200)


def scores(rr=None, leaks=None, owner_hits=15, returned=30):
    """Per-trial scores as ``score_items`` returns them."""
    return {
        "utility": {
            "items": 120,
            "rr": dict({"pooled": 0.9, "after-k": 1.0, "after-all": 1.0, "built": 1.0}, **(rr or {})),
            "hit": {arm: 1.0 for arm in ARMS},
            "shown": {arm: 10.0 for arm in ARMS},
        },
        "leakage": {
            "items": 50,
            "leaks": dict({"pooled": 40, "after-k": 0, "after-all": 0, "built": 0}, **(leaks or {})),
        },
        "owner": {"items": 15, "hits": {arm: owner_hits for arm in ARMS}},
        "over_blocking": {"items": 30, "returned": returned, "hit_at_k": returned},
    }


def trial(seed, **options):
    return {
        "seed": seed,
        "members": {},
        "memories": 150,
        "timestamps_increase": True,
        "items": [],
        "scores": scores(**options),
    }


def search_ignoring_viewers(self, query, **options):
    """A broken store: reads with viewers see everything."""
    options.pop("viewers", None)
    return ORIGINAL_SEARCH(self, query, **options)


def search_hiding_everything(self, query, **options):
    """A broken store: reads with viewers see nothing."""
    if options.get("viewers") is not None:
        return []
    return ORIGINAL_SEARCH(self, query, **options)


def run_subset(seed, search):
    """Ask one viewer's utility items and every canary item, against a patched store."""
    workspace = utility.build_workspace(seed)
    asked = [
        item
        for item in utility.questions(workspace)
        if item["type"] != "utility" or item["viewer"] == utility.MEMBERS[0]
    ]
    with mock.patch.object(utility.MemoryStore, "search", search):
        with utility.MemoryStore(":memory:") as store:
            labels, _ = utility.populate(store, workspace)
            items = utility.answer(store, labels, workspace, asked)
    return dict(trial(seed), items=items, scores=utility.score_items(items))


class DecisionRuleTests(unittest.TestCase):
    def test_clean_filtered_arms_at_no_cost_support_h1(self):
        result = decision()
        self.assertEqual(result["verdict"], "H1 supported")
        self.assertTrue(result["rule_1_positive_controls_hold"])
        self.assertTrue(result["rule_2_no_over_blocking"])
        self.assertTrue(result["rule_3_zero_leakage"])
        self.assertEqual(result["rule_4_utility"], "no utility cost")

    def test_a_pooled_arm_that_leaks_too_little_is_inconclusive(self):
        result = decision(pooled=0.49)
        self.assertFalse(result["rule_1_pooled_leaks"])
        self.assertEqual(result["verdict"], "inconclusive (a control failed)")
        self.assertEqual(decision(pooled=0.5)["verdict"], "H1 supported")

    def test_owners_who_miss_their_canaries_make_it_inconclusive(self):
        result = decision(owner=0.94)
        self.assertFalse(result["rule_1_owner_finds_canary"])
        self.assertEqual(result["verdict"], "inconclusive (a control failed)")
        self.assertEqual(decision(owner=0.95)["verdict"], "H1 supported")

    def test_one_over_blocked_item_makes_it_inconclusive(self):
        result = decision(missing=1)
        self.assertFalse(result["rule_2_no_over_blocking"])
        self.assertEqual(result["verdict"], "inconclusive (a control failed)")

    def test_a_failed_control_comes_before_a_leak(self):
        result = decision(missing=1, leaks={"built": 3})
        self.assertEqual(result["verdict"], "inconclusive (a control failed)")
        self.assertFalse(result["rule_3_zero_leakage"])

    def test_one_leak_in_any_filtered_arm_rejects_h1(self):
        for arm in utility.FILTERED_ARMS:
            result = decision(leaks={arm: 1})
            self.assertFalse(result["rule_3_zero_leakage"])
            self.assertFalse(result["rule_3_zero_leakage_by_arm"][arm])
            self.assertTrue(result["verdict"].startswith("H1 not supported: rule 3"), arm)
            self.assertIn("{} (1 of 1000 items, first failing seed 7)".format(arm), result["verdict"])

    def test_utility_readings_at_the_margin(self):
        cases = {
            (0.0, 0.0): "no utility cost",
            (-0.01, 0.02): "no utility cost",
            (-0.0100001, 0.02): "unclear",
            (-0.02, -0.01): "unclear",
            (-0.05, -0.0100001): "utility cost",
        }
        for ci, reading in cases.items():
            self.assertEqual(utility.utility_reading(list(ci)), reading, ci)

    def test_a_utility_cost_rejects_h1(self):
        result = decision(ci=(-0.05, -0.02), mean=-0.035)
        self.assertEqual(result["rule_4_utility"], "utility cost")
        self.assertEqual(
            result["verdict"],
            "H1 not supported: rule 4, utility cost "
            "(built minus after-all -0.0350, 95% CI -0.0500 to -0.0200)",
        )

    def test_an_unclear_utility_rejects_h1(self):
        result = decision(ci=(-0.03, 0.01), mean=-0.01)
        self.assertEqual(result["rule_4_utility"], "unclear")
        self.assertTrue(result["verdict"].startswith("H1 not supported: rule 4, utility unclear"))

    def test_every_reason_is_listed(self):
        result = decision(leaks={"built": 2}, ci=(-0.05, -0.02))
        self.assertEqual(len(result["reasons"]), 2)
        self.assertTrue(result["verdict"].startswith("H1 not supported: rule 3"))
        self.assertIn("; rule 4, utility cost", result["verdict"])


class SummaryTests(unittest.TestCase):
    def test_counts_and_first_failing_seeds(self):
        result = utility.summarise(
            [trial(4), trial(6, leaks={"built": 1}, returned=29), trial(5, leaks={"built": 2})]
        )
        built = result["leakage"]["by_arm"]["built"]
        self.assertEqual((built["leaks"], built["trials_with_leak"], built["first_seed"]), (3, 2, 5))
        self.assertEqual(result["leakage"]["items"], 150)
        self.assertEqual(result["leakage"]["by_arm"]["pooled"]["leaks"], 120)
        self.assertIsNone(result["leakage"]["by_arm"]["after-all"]["first_seed"])
        over = result["over_blocking"]
        self.assertEqual((over["missing"], over["first_seed"]), (1, 6))
        self.assertIsNone(result["owner"]["built_first_miss_seed"])

    def test_d_is_built_minus_after_all_per_trial(self):
        trials = [trial(seed, rr={"after-all": 0.75}) for seed in range(3)]
        difference = utility.summarise(trials)["utility"]["differences"]["built_minus_after_all"]
        self.assertEqual(difference["values"], [0.25, 0.25, 0.25])
        self.assertEqual(difference["mean"], 0.25)
        self.assertEqual(difference["ci95"], [0.25, 0.25])
        self.assertEqual((difference["higher"], difference["equal"], difference["lower"]), (3, 0, 0))

    def test_the_bootstrap_uses_random_0_and_the_51st_and_1950th_means(self):
        values = [0.0, 0.1, -0.2, 0.05, 0.3]
        rng = random.Random(0)
        means = sorted(
            sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000)
        )
        result = utility.bootstrap(values)
        self.assertEqual(result["ci95"], [means[50], means[1949]])
        self.assertLess(result["ci95"][0], result["mean"])
        self.assertLess(result["mean"], result["ci95"][1])


class ScoringTests(unittest.TestCase):
    def test_reciprocal_rank_counts_the_first_gold_within_k(self):
        shown = ["x{}".format(index) for index in range(12)]
        self.assertEqual(utility.reciprocal_rank(shown, ["x2", "x5"]), 1 / 3)
        self.assertEqual(utility.reciprocal_rank(shown, ["x10"]), 0.0)
        self.assertEqual(utility.reciprocal_rank([], ["x0"]), 0.0)

    def test_items_are_scored_per_arm(self):
        shown = {"pooled": ["h", "g"], "after-k": ["g"], "after-all": ["g", "v"], "built": []}
        items = [
            {"type": "utility", "gold": ["g"], "shown": shown},
            {"type": "leakage", "canary": "h", "shown": shown},
            {"type": "owner", "canary": "g", "shown": shown},
            {
                "type": "over_blocking",
                "canary": "c",
                "shown": {"built": ["x{}".format(index) for index in range(11)] + ["c"]},
            },
        ]
        result = utility.score_items(items)
        self.assertEqual(
            result["utility"]["rr"], {"pooled": 0.5, "after-k": 1.0, "after-all": 1.0, "built": 0.0}
        )
        self.assertEqual(result["utility"]["shown"]["after-all"], 2.0)
        self.assertEqual(
            result["leakage"]["leaks"], {"pooled": 1, "after-k": 0, "after-all": 0, "built": 0}
        )
        self.assertEqual(result["owner"]["hits"]["built"], 0)
        self.assertEqual(result["over_blocking"], {"items": 1, "returned": 1, "hit_at_k": 0})


class DetectorTests(unittest.TestCase):
    """The checks fire when the store is broken on purpose."""

    def test_a_store_that_ignores_viewers_fails_rule_3(self):
        broken = run_subset(SEED, search_ignoring_viewers)
        leaks = broken["scores"]["leakage"]["leaks"]
        self.assertGreater(leaks["built"], 0)
        self.assertEqual((leaks["after-k"], leaks["after-all"]), (0, 0))
        result = utility.decide(utility.summarise([broken]), 1)
        self.assertEqual(result["rule_3_zero_leakage_by_arm"]["built"], False)
        self.assertTrue(result["verdict"].startswith("H1 not supported: rule 3, leaks in built"))

    def test_a_store_that_shows_viewers_nothing_fails_the_controls(self):
        broken = run_subset(SEED, search_hiding_everything)
        self.assertEqual(broken["scores"]["leakage"]["leaks"]["built"], 0)
        self.assertEqual(broken["scores"]["over_blocking"]["returned"], 0)
        self.assertEqual(broken["scores"]["owner"]["hits"]["built"], 0)
        result = utility.decide(utility.summarise([broken]), 1)
        self.assertFalse(result["rule_2_no_over_blocking"])
        self.assertEqual(result["verdict"], "inconclusive (a control failed)")


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = utility.build_workspace(SEED)
        cls.trial = utility.run_trial(SEED)
        cls.channels = utility.visible_channels(cls.workspace)
        cls.memory = {memory["label"]: memory for memory in cls.workspace["memories"]}

    def visible(self, viewer, label):
        return self.memory[label]["channel"] in self.channels[viewer]

    def items(self, kind):
        return [item for item in self.trial["items"] if item["type"] == kind]

    def test_the_workspace_follows_the_spec(self):
        members = self.workspace["members"]
        self.assertEqual(members["general"], list(utility.MEMBERS))
        for channel in utility.PRIVATE_CHANNELS:
            self.assertIn(len(members[channel]), (2, 3, 4))
            self.assertLessEqual(set(members[channel]), set(utility.MEMBERS))
        memories = self.workspace["memories"]
        self.assertEqual(len(memories), 150)
        for channel, facts in (("general", 40), ("p1", 30), ("p2", 30), ("p3", 30)):
            in_channel = [memory for memory in memories if memory["channel"] == channel]
            self.assertEqual(sum(not memory["canary"] for memory in in_channel), facts)
            self.assertEqual(sum(memory["canary"] for memory in in_channel), 5)
        self.assertEqual(
            {memory["value"] for memory in memories if memory["canary"]},
            {"canary{}x{}".format(SEED, k) for k in range(20)},
        )
        for memory in memories:
            self.assertIn(memory["speaker"], members[memory["channel"]])
            attribute, subject, value, filler = CONTENT.match(memory["content"]).groups()
            self.assertEqual(
                (attribute, subject, value), (memory["attribute"], memory["subject"], memory["value"])
            )
            self.assertIn(subject, utility.SUBJECTS)
            self.assertIn(attribute, utility.ATTRIBUTES)
            self.assertLessEqual(len(filler.split()), 6)
            self.assertLessEqual(set(filler.split()), set(utility.FILLER))
            if not memory["canary"]:
                self.assertIn(value, utility.VALUES)
        vocabularies = [
            set(utility.SUBJECTS),
            set(utility.ATTRIBUTES),
            set(utility.VALUES),
            set(utility.FILLER),
            {"the", "of", "is"},
        ]
        words = [word for vocabulary in vocabularies for word in vocabulary]
        self.assertEqual(len(words), len(set(words)))

    def test_the_store_holds_the_recorded_memberships(self):
        with utility.MemoryStore(":memory:") as store:
            labels, stamps = utility.populate(store, self.workspace)
            for member in utility.MEMBERS:
                self.assertEqual(store.channels_of(member), sorted(self.channels[member]))
            stored = store.list_memories(limit=1000)
        self.assertEqual(len(stored), 150)
        self.assertEqual({labels[memory.id] for memory in stored}, set(self.memory))
        self.assertEqual(stamps, sorted(stamps))

    def test_questions_follow_the_spec(self):
        utility_items = self.items("utility")
        for viewer in utility.MEMBERS:
            asked = [item for item in utility_items if item["viewer"] == viewer]
            self.assertEqual(len(asked), 20)
            self.assertEqual(len({item["query"] for item in asked}), 20)
            for item in asked:
                attribute, subject = item["query"].split()
                expected = sorted(
                    label
                    for label, memory in self.memory.items()
                    if (memory["subject"], memory["attribute"]) == (subject, attribute)
                    and self.visible(viewer, label)
                )
                self.assertTrue(expected)
                self.assertEqual(item["gold"], expected)
        canaries = [label for label, memory in self.memory.items() if memory["canary"]]
        private = [label for label in canaries if self.memory[label]["channel"] != "general"]
        self.assertEqual(
            sorted((item["viewer"], item["canary"]) for item in self.items("leakage")),
            sorted(
                (viewer, label)
                for viewer in utility.MEMBERS
                for label in private
                if not self.visible(viewer, label)
            ),
        )
        owners = self.items("owner")
        self.assertEqual(sorted(item["canary"] for item in owners), sorted(private))
        for item in owners:
            self.assertEqual(item["viewer"], self.memory[item["canary"]]["speaker"])
        self.assertEqual(len(self.items("over_blocking")), 5 * 6)
        for item in self.trial["items"]:
            if item["type"] != "utility":
                canary = self.memory[item["canary"]]
                self.assertEqual(item["query"], "{} {}".format(canary["attribute"], canary["subject"]))

    def test_each_arm_shows_what_it_should(self):
        for item in self.trial["items"]:
            viewer = item["viewer"]
            if item["type"] == "over_blocking":
                for label in item["shown"]["built"]:
                    self.assertTrue(self.visible(viewer, label), label)
                continue
            shown = item["shown"]
            for arm in ARMS:
                self.assertLessEqual(len(shown[arm]), utility.K)
            for arm in utility.FILTERED_ARMS:
                for label in shown[arm]:
                    self.assertTrue(self.visible(viewer, label), (arm, label))
            words = set(item["query"].split())
            matches = sum(
                bool(words & set(re.findall(r"\w+", memory["content"])))
                for memory in self.memory.values()
            )
            self.assertEqual(len(shown["pooled"]), min(utility.K, matches))
            # search(q, limit=k) is the first k of the full ranking, filtered.
            self.assertEqual(
                shown["after-k"],
                [label for label in shown["pooled"] if self.visible(viewer, label)],
            )

    def test_the_positive_controls_can_fire(self):
        scores = self.trial["scores"]
        self.assertGreater(scores["leakage"]["leaks"]["pooled"], 0)
        self.assertGreater(scores["owner"]["hits"]["built"], 0)
        self.assertTrue(self.trial["timestamps_increase"])
        self.assertEqual(self.trial["memories"], 150)

    def test_a_trial_is_reproducible(self):
        self.assertEqual(utility.run_trial(SEED), self.trial)

    def test_a_report_renders_and_serialises(self):
        report = utility.report_for([self.trial])
        json.dumps(report)
        text = utility.render(report)
        self.assertIn("**Verdict: {}**".format(report["decision"]["verdict"]), text)
        for arm in ARMS:
            self.assertIn(utility.ARM_LABELS[arm], text)


if __name__ == "__main__":
    unittest.main()
