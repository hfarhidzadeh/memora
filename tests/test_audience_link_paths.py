import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "audience_link_paths.py"
spec = importlib.util.spec_from_file_location("audience_link_paths_experiment", SCRIPT)
link_paths = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = link_paths
spec.loader.exec_module(link_paths)

from dynmemory import Link, Memory, MemoryType, Relationship, SearchResult  # noqa: E402

PATHS = link_paths.PATHS
ARMS = link_paths.ARMS
AFTER_ARMS = ("after-naive", "after-careful")
TARGET = link_paths.TARGET
INCONCLUSIVE = "inconclusive (a control failed)"
# The real trial's seed. Its dispute path searches with kind="fact" and its
# disputant is a correction, so the flag has to come from outside the filter.
SEED = 5
# Joins, background memories and background links come before planted writes.
BACKGROUND_OPERATIONS = (
    len(link_paths.MEMBERSHIPS)
    + link_paths.VISIBLE_BACKGROUND
    + link_paths.HIDDEN_BACKGROUND
    + link_paths.BACKGROUND_LINKS
)


def expected(path, world, arm):
    """(P, L) as the design predicts them, with a sound built path."""
    if world == "shared":
        return True, path != "dispute"
    if world in ("present", "present-copy") and arm in AFTER_ARMS:
        if path == "dispute":
            leaks = arm == "after-naive"
            return leaks, leaks
        return True, True
    return False, False


def synthetic_trial(seed=0, changes=None, differ=()):
    """A trial whose statistics are ``expected`` except where ``changes`` says.

    ``changes`` maps (path, world, arm) to (P, L). In the (path, world) pairs
    of ``differ``, the built arm's observation differs from the before arm's.
    """
    changes = changes or {}
    trial = {"seed": seed}
    for path in PATHS:
        worlds = {}
        for world in link_paths.WORLDS[path]:
            arms = {}
            for arm in ARMS:
                probe, leak = changes.get((path, world, arm), expected(path, world, arm))
                observation = []
                if arm == "built" and (path, world) in differ:
                    observation = [("V",)]
                arms[arm] = {"p": probe, "l": leak, "observation": observation}
            worlds[world] = {
                "mode": "keep",
                "kind": None,
                "full_hidden": 51,
                "before_hidden": 0,
                "arms": arms,
            }
        trial[path] = {
            "settings": {
                "mode": "keep",
                "kind_filter": None,
                "disputant_kind": "fact",
                "relation": "related_to",
            },
            "worlds": worlds,
        }
    return trial


def trials_with(changes=None, differ=(), affected=range(20), count=20):
    """``count`` synthetic trials; ``changes`` and ``differ`` apply to the seeds in ``affected``."""
    return [
        synthetic_trial(seed, changes, differ) if seed in affected else synthetic_trial(seed)
        for seed in range(count)
    ]


def decision(trials):
    return link_paths.decide(link_paths.summarise(trials), len(trials))


def row(label, content, links=()):
    """One row of a canonical observation."""
    return (label, "fact", content, None, "visible", "insider", 1.0, "lexical", 1.0, 0.0, 1.0, tuple(links))


class DecisionRuleTests(unittest.TestCase):
    def test_the_predicted_pattern_leaks_in_the_baseline_and_the_built_path_closes_it(self):
        result = decision(trials_with())
        for path in PATHS:
            self.assertEqual(result[path]["verdict"], "the baseline leaks through {}".format(path))
            self.assertEqual(
                result[path]["verdict_on_the_built_path"], "the built path closes {}".format(path)
            )
            for rule in (
                "rule_1_probe_works",
                "rule_2_negative_controls_hold",
                "rule_3_baseline_leaks",
                "rule_4_built_never_leaks_where_before_does_not",
                "rule_4_built_equals_before",
                "rule_4_built_closes",
            ):
                self.assertIs(result[path][rule], True, (path, rule))
            self.assertTrue(all(prediction["held"] for prediction in result[path]["predictions"]))
        self.assertEqual(len(result["topic"]["predictions"]), 3)

    def test_a_baseline_that_shows_nothing_does_not_leak(self):
        result = decision(trials_with({("graph", "present", "after-naive"): (False, False)}))
        self.assertFalse(result["graph"]["rule_3_baseline_leaks"])
        self.assertEqual(result["graph"]["verdict"], "the baseline does not leak through graph")
        self.assertEqual(result["dispute"]["verdict"], "the baseline leaks through dispute")

    def test_rule_3_needs_95_percent_of_trials(self):
        silent = {("topic", "present", "after-naive"): (False, False)}
        self.assertTrue(decision(trials_with(silent, affected={0}))["topic"]["rule_3_baseline_leaks"])
        result = decision(trials_with(silent, affected={0, 1}))
        self.assertFalse(result["topic"]["rule_3_baseline_leaks"])
        self.assertEqual(result["topic"]["verdict"], "the baseline does not leak through topic")

    def test_a_dead_probe_in_any_arm_is_inconclusive(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                result = decision(
                    trials_with({("dispute", "shared", arm): (False, False)}, affected={0, 1})
                )
                self.assertFalse(result["dispute"]["rule_1_probe_works"])
                self.assertEqual(result["dispute"]["verdict"], INCONCLUSIVE)
                self.assertEqual(result["graph"]["verdict"], "the baseline leaks through graph")
        # One miss in 20 trials is still 95%.
        result = decision(trials_with({("graph", "shared", "before"): (False, False)}, affected={0}))
        self.assertTrue(result["graph"]["rule_1_probe_works"])

    def test_one_leak_in_any_negative_control_is_inconclusive(self):
        for path in PATHS:
            for world in link_paths.NEGATIVE_WORLDS[path]:
                for arm in ARMS:
                    with self.subTest(path=path, world=world, arm=arm):
                        result = decision(
                            trials_with({(path, world, arm): (False, True)}, affected={7})
                        )
                        self.assertFalse(result[path]["rule_2_negative_controls_hold"])
                        self.assertEqual(result[path]["verdict"], INCONCLUSIVE)
        self.assertEqual(link_paths.NEGATIVE_WORLDS["topic"], ("absent", "unlinked", "present-keep"))

    def test_present_copy_is_informative_not_a_control(self):
        result = decision(trials_with({("topic", "present-copy", "after-naive"): (False, False)}))
        self.assertTrue(result["topic"]["rule_2_negative_controls_hold"])
        self.assertEqual(result["topic"]["verdict"], "the baseline leaks through topic")
        held = {p["statement"]: p["held"] for p in result["topic"]["predictions"]}
        self.assertFalse(held["after-naive leaks in at least 95% of `present-copy` trials"])
        self.assertTrue(held["after-careful leaks in at least 95% of `present-copy` trials"])

    def test_a_built_leak_where_before_is_clean_fails_rule_4(self):
        trials = trials_with({("graph", "present", "built"): (True, True)}, affected={3})
        summary = link_paths.summarise(trials)
        result = link_paths.decide(summary, len(trials))
        self.assertFalse(result["graph"]["rule_4_built_never_leaks_where_before_does_not"])
        self.assertTrue(result["graph"]["rule_4_built_equals_before"])
        self.assertEqual(
            result["graph"]["verdict_on_the_built_path"], "the built path does not close graph"
        )
        self.assertEqual(result["graph"]["verdict"], "the baseline leaks through graph")
        self.assertEqual(summary["graph"]["first_failing_seed"], 3)
        self.assertEqual(
            result["dispute"]["verdict_on_the_built_path"], "the built path closes dispute"
        )

    def test_a_built_leak_where_before_leaks_too_passes_the_first_part_of_rule_4(self):
        both = {
            ("dispute", "present", "built"): (True, True),
            ("dispute", "present", "before"): (True, True),
        }
        result = decision(trials_with(both))
        self.assertTrue(result["dispute"]["rule_4_built_never_leaks_where_before_does_not"])
        self.assertEqual(
            result["dispute"]["verdict_on_the_built_path"], "the built path closes dispute"
        )

    def test_a_built_observation_that_differs_from_before_fails_rule_4(self):
        trials = trials_with(differ={("topic", "absent")}, affected={5})
        summary = link_paths.summarise(trials)
        result = link_paths.decide(summary, len(trials))
        self.assertTrue(result["topic"]["rule_4_built_never_leaks_where_before_does_not"])
        self.assertFalse(result["topic"]["rule_4_built_equals_before"])
        self.assertEqual(
            result["topic"]["verdict_on_the_built_path"], "the built path does not close topic"
        )
        self.assertEqual(summary["topic"]["built_differs_from_before"], 1)
        self.assertEqual(
            summary["topic"]["failures"],
            [
                {
                    "seed": 5,
                    "world": "absent",
                    "observation_differs": True,
                    "built_leaks_where_before_does_not": False,
                }
            ],
        )
        self.assertTrue(result["graph"]["rule_4_built_closes"])

    def test_the_built_verdict_follows_rule_4_alone(self):
        # As the spec words it, a failed control makes the baseline verdict
        # inconclusive but leaves the verdict on the built path to rule 4.
        result = decision(trials_with({("graph", "shared", "built"): (False, False)}))
        self.assertEqual(result["graph"]["verdict"], INCONCLUSIVE)
        self.assertEqual(result["graph"]["verdict_on_the_built_path"], "the built path closes graph")

    def test_predictions_are_reported_but_do_not_decide(self):
        changes = {
            ("graph", "present", "after-careful"): (False, False),
            ("dispute", "present", "after-careful"): (True, True),
            ("topic", "present-copy", "after-careful"): (False, False),
        }
        result = decision(trials_with(changes))
        for path in PATHS:
            self.assertFalse(all(p["held"] for p in result[path]["predictions"]), path)
            self.assertEqual(result[path]["verdict"], "the baseline leaks through {}".format(path))
            self.assertEqual(
                result[path]["verdict_on_the_built_path"], "the built path closes {}".format(path)
            )


class StatisticTests(unittest.TestCase):
    def test_graph_and_topic_fire_when_a_memory_without_the_target_is_shown(self):
        visible = frozenset({"H", "V"})
        planted = row("H", "w0001 qtargetword w0002")
        for path in ("graph", "topic"):
            self.assertEqual(link_paths.statistics(path, [], visible), (False, False))
            self.assertEqual(link_paths.statistics(path, [planted], visible), (False, False))
            self.assertEqual(
                link_paths.statistics(path, [planted, row("V", "w0001 w0002")], visible),
                (True, True),
            )
            # A longer word that contains the target is not the target.
            self.assertEqual(
                link_paths.statistics(path, [row("V", "xqtargetword")], visible), (True, True)
            )

    def test_dispute_probe_and_leak(self):
        flag = ("D", "C", "contradicts")
        shown = [row("C", "qtargetword w0001", [flag])]
        self.assertEqual(link_paths.statistics("dispute", shown, frozenset({"C"})), (True, True))
        self.assertEqual(
            link_paths.statistics("dispute", shown, frozenset({"C", "D"})), (True, False)
        )
        self.assertEqual(
            link_paths.statistics(
                "dispute", [row("C", "qtargetword", [("C", "D", "supports")])], frozenset({"C"})
            ),
            (False, True),
        )
        self.assertEqual(
            link_paths.statistics("dispute", [row("C", "qtargetword")], frozenset({"C"})),
            (False, False),
        )
        # An ID the experiment never labelled counts as hidden.
        self.assertEqual(
            link_paths.statistics(
                "dispute", [row("C", "qtargetword", [("C", "?abc", "contradicts")])], frozenset({"C"})
            ),
            (True, True),
        )

    def test_the_canonical_observation_uses_labels_and_leaves_out_ids_and_times(self):
        memory = Memory(
            id="id1",
            kind=MemoryType.FACT,
            content="w0001 qtargetword",
            memory_key=None,
            source="conversation",
            confidence=0.8,
            importance=0.5,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            last_accessed_at=None,
            access_count=3,
            archived_at=None,
            superseded_by=None,
            channel="visible",
            speaker="insider",
        )
        result = SearchResult(
            memory=memory,
            score=1.0,
            matched_by="lexical",
            links=(
                Link("id2", "id1", Relationship.SUPPORTS),
                Link("id1", "zz", Relationship.CONTRADICTS),
            ),
            lexical_score=2.5,
            coverage=1.0,
        )
        self.assertEqual(
            link_paths.observe([result], {"id1": "C", "id2": "D"}),
            [
                (
                    "C", "fact", "w0001 qtargetword", None, "visible", "insider",
                    1.0, "lexical", 2.5, 0.0, 1.0,
                    (("C", "?zz", "contradicts"), ("D", "C", "supports")),
                )
            ],
        )


class ScriptTests(unittest.TestCase):
    def test_draws_are_reproducible_and_shared_by_the_worlds(self):
        chosen = link_paths.draw(SEED, "topic")
        self.assertEqual(chosen, link_paths.draw(SEED, "topic"))
        self.assertNotEqual(chosen, link_paths.draw(SEED + 1, "topic"))
        plans = {
            world: link_paths.script(chosen, "topic", world) for world in link_paths.WORLDS["topic"]
        }
        shared_part = plans["absent"]["operations"][:BACKGROUND_OPERATIONS]
        for plan in plans.values():
            self.assertEqual(plan["operations"][:BACKGROUND_OPERATIONS], shared_part)
        self.assertEqual(
            {world: plan["mode"] for world, plan in plans.items()},
            {
                "absent": "topic",
                "present": "topic",
                "shared": "topic",
                "unlinked": "topic",
                "present-keep": "keep",
                "present-copy": "copy",
            },
        )

    def test_the_background_is_as_specified(self):
        chosen = link_paths.draw(SEED, "graph")
        channels = Counter(channel for _, channel, _ in chosen["background"])
        self.assertEqual(channels, {"visible": 150, "hidden": 50})
        labels = {label for label, _, _ in chosen["background"]}
        links = chosen["background_links"]
        self.assertEqual(len(links), 40)
        self.assertEqual(len({frozenset(link[:2]) for link in links}), 40)
        for source, target, relation in links:
            self.assertIn(source, labels)
            self.assertIn(target, labels)
            self.assertIn(relation, ("related_to", "supports", "contradicts", "depends_on"))

    def test_only_planted_memories_contain_the_target(self):
        carriers = {"graph": {"H"}, "dispute": {"C"}, "topic": {"H"}}
        for path in PATHS:
            chosen = link_paths.draw(SEED, path)
            contents = [content for _, _, content in chosen["background"]]
            contents += list(chosen["contents"].values())
            for content in contents:
                self.assertEqual(len(content.split()), 12)
            for _, _, content in chosen["background"]:
                self.assertNotIn(TARGET, content.split())
            self.assertEqual(
                {label for label, text in chosen["contents"].items() if TARGET in text.split()},
                carriers[path],
            )
            for label in carriers[path]:
                self.assertEqual(chosen["contents"][label].split().count(TARGET), 1)

    def test_the_planted_writes_follow_the_spec(self):
        def planted(path, world):
            """The planted writes as (add, label, kind, key, channel) or (relate, source, target)."""
            plan = link_paths.script(link_paths.draw(SEED, path), path, world)
            writes = []
            for operation in plan["operations"][BACKGROUND_OPERATIONS:]:
                if operation[0] == "add":
                    _, label, _, kind, key, channel = operation
                    writes.append(("add", label, kind, key, channel))
                else:
                    writes.append(tuple(operation[:3]))
            return writes

        topic_link = link_paths.draw(SEED, "topic")["link"]
        self.assertEqual(
            planted("topic", "present"),
            [
                ("add", "O_old", "fact", "planted", "visible"),
                ("add", "H", "fact", None, "hidden"),
                ("relate",) + tuple(topic_link[:2]),
                ("add", "O_new", "fact", "planted", "visible"),
            ],
        )
        self.assertEqual(set(topic_link[:2]), {"H", "O_old"})
        self.assertEqual(
            planted("topic", "absent"),
            [("add", "O_old", "fact", "planted", "visible"), ("add", "O_new", "fact", "planted", "visible")],
        )
        graph_link = link_paths.draw(SEED, "graph")["link"]
        self.assertEqual(
            planted("graph", "shared"),
            [
                ("add", "H", "fact", None, "visible"),
                ("add", "V", "fact", None, "visible"),
                ("relate",) + tuple(graph_link[:2]),
            ],
        )
        self.assertEqual(set(graph_link[:2]), {"H", "V"})
        self.assertEqual(planted("graph", "absent"), [("add", "V", "fact", None, "visible")])
        self.assertEqual(
            planted("dispute", "unlinked"),
            [("add", "C", "fact", None, "visible"), ("add", "D", "correction", None, "hidden")],
        )
        dispute_link = link_paths.draw(SEED, "dispute")["link"]
        self.assertEqual(dispute_link[2], "contradicts")
        self.assertEqual(set(dispute_link[:2]), {"C", "D"})

    def test_visibility_comes_from_the_scripts_own_record(self):
        operations = [
            ("join", "visible", "viewer"),
            ("join", "hidden", "insider"),
            ("join", "visible", "insider"),
            ("add", "a", "x", "fact", None, "visible"),
            ("add", "b", "x", "fact", None, "hidden"),
            ("add", "c", "x", "fact", None, None),
            ("add", "d", "x", "fact", None, "nobody"),
            ("add", "e", "x", "fact", None, "visible"),
            ("relate", "a", "b", "related_to"),
            ("relate", "e", "a", "supports"),
        ]
        visible = link_paths.visible_labels(operations)
        self.assertEqual(visible, frozenset({"a", "e"}))
        self.assertEqual(
            link_paths.visible_part(operations, visible),
            [
                ("join", "visible", "viewer"),
                ("add", "a", "x", "fact", None, "visible"),
                ("add", "e", "x", "fact", None, "visible"),
                ("relate", "e", "a", "supports"),
            ],
        )

    def test_the_kind_filter_is_used_in_about_half_of_trials(self):
        filters = [link_paths.draw(seed, "dispute")["kind_filter"] for seed in range(100)]
        self.assertEqual(set(filters), {None, "fact"})
        self.assertTrue(35 <= filters.count("fact") <= 65, filters.count("fact"))


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = link_paths.run_trial(SEED)

    def arm(self, path, world, arm):
        return self.trial[path]["worlds"][world]["arms"][arm]

    def statistics(self, path, world, arm):
        outcome = self.arm(path, world, arm)
        return outcome["p"], outcome["l"]

    def shown(self, path, world, arm):
        return [row[0] for row in self.arm(path, world, arm)["observation"]]

    def links(self, path, world, arm):
        return [row[11] for row in self.arm(path, world, arm)["observation"]]

    def test_the_trial_covers_every_path_world_and_arm(self):
        self.assertEqual(self.trial["seed"], SEED)
        self.assertEqual(set(self.trial), {"seed"} | set(PATHS))
        for path in PATHS:
            self.assertEqual(tuple(self.trial[path]["worlds"]), link_paths.WORLDS[path])
            for world in link_paths.WORLDS[path]:
                self.assertEqual(tuple(self.trial[path]["worlds"][world]["arms"]), ARMS)
        self.assertEqual(self.trial["dispute"]["settings"]["kind_filter"], "fact")
        self.assertEqual(self.trial["dispute"]["settings"]["disputant_kind"], "correction")

    def test_graph_present_fires_after_naive_and_after_careful(self):
        for arm in AFTER_ARMS:
            self.assertEqual(self.shown("graph", "present", arm), ["V"])
            self.assertEqual(self.statistics("graph", "present", arm), (True, True))
        self.assertEqual(self.links("graph", "present", "after-careful"), [()])
        for arm in ARMS:
            self.assertEqual(self.shown("graph", "shared", arm), ["H", "V"])
            self.assertTrue(self.statistics("graph", "shared", arm)[0])
            for world in ("absent", "unlinked"):
                self.assertEqual(self.statistics("graph", world, arm), (False, False))

    def test_dispute_present_fires_after_naive_only(self):
        # The disputant is a correction outside the kind filter, yet flagged.
        self.assertEqual(self.shown("dispute", "present", "after-naive"), ["C"])
        (flags,) = self.links("dispute", "present", "after-naive")
        self.assertEqual([link[2] for link in flags], ["contradicts"])
        self.assertEqual(set(flags[0][:2]), {"C", "D"})
        self.assertEqual(self.statistics("dispute", "present", "after-naive"), (True, True))
        self.assertEqual(self.links("dispute", "present", "after-careful"), [()])
        self.assertEqual(self.statistics("dispute", "present", "after-careful"), (False, False))
        for world in link_paths.WORLDS["dispute"]:
            self.assertEqual(self.trial["dispute"]["worlds"][world]["kind"], "fact")
        for arm in ARMS:
            # In shared, the kind filter keeps the visible correction out of the
            # results, while C still shows the flag.
            self.assertEqual(self.shown("dispute", "shared", arm), ["C"])
            self.assertEqual(self.links("dispute", "shared", arm), [flags])
            self.assertEqual(self.statistics("dispute", "shared", arm), (True, False))
            for world in ("absent", "unlinked"):
                self.assertEqual(self.shown("dispute", world, arm), ["C"])
                self.assertEqual(self.statistics("dispute", world, arm), (False, False))

    def test_topic_present_fires_unless_the_store_keeps_links_on_the_old_version(self):
        for world in ("present", "present-copy"):
            for arm in AFTER_ARMS:
                self.assertEqual(self.shown("topic", world, arm), ["O_new"])
                self.assertEqual(self.statistics("topic", world, arm), (True, True))
        for arm in ARMS:
            self.assertEqual(self.shown("topic", "shared", arm), ["H", "O_new"])
            for world in ("absent", "unlinked", "present-keep"):
                self.assertEqual(self.statistics("topic", world, arm), (False, False))

    def test_the_before_arm_never_holds_or_shows_a_hidden_memory(self):
        for path in PATHS:
            chosen = link_paths.draw(SEED, path)
            for world, result in self.trial[path]["worlds"].items():
                with self.subTest(path=path, world=world):
                    visible = link_paths.visible_labels(
                        link_paths.script(chosen, path, world)["operations"]
                    )
                    self.assertEqual(result["before_hidden"], 0)
                    # The other arms read a store that does hold the hidden memories.
                    self.assertGreaterEqual(result["full_hidden"], link_paths.HIDDEN_BACKGROUND)
                    for row_ in self.arm(path, world, "before")["observation"]:
                        self.assertIn(row_[0], visible)
                        for link in row_[11]:
                            self.assertIn(link[0], visible)
                            self.assertIn(link[1], visible)

    def test_the_built_arm_equals_the_before_arm_on_this_trial(self):
        for path in PATHS:
            for world in link_paths.WORLDS[path]:
                with self.subTest(path=path, world=world):
                    self.assertEqual(
                        self.arm(path, world, "built")["observation"],
                        self.arm(path, world, "before")["observation"],
                    )

    def test_every_shown_memory_and_link_end_has_a_label(self):
        for path in PATHS:
            for world in link_paths.WORLDS[path]:
                for arm in ARMS:
                    for row_ in self.arm(path, world, arm)["observation"]:
                        ends = [row_[0]] + [end for link in row_[11] for end in link[:2]]
                        self.assertFalse(any(end.startswith("?") for end in ends))


class ComparisonPowerTests(unittest.TestCase):
    def test_a_built_path_that_ignores_viewers_is_caught(self):
        # Positive control for rule 4: a deliberately broken built arm must fail it.
        search = link_paths.MemoryStore.search

        def ignore_viewers(store, query, **options):
            options.pop("viewers", None)
            return search(store, query, **options)

        with mock.patch.object(link_paths.MemoryStore, "search", ignore_viewers):
            outcome = link_paths.run_path(SEED, "graph")
        self.assertEqual(self.shown(outcome, "present"), ["H", "V"])
        self.assertTrue(outcome["worlds"]["present"]["arms"]["built"]["l"])

        trial = synthetic_trial(SEED)
        trial["graph"] = outcome
        summary = link_paths.summarise([trial])
        result = link_paths.decide(summary, 1)
        self.assertFalse(result["graph"]["rule_4_built_never_leaks_where_before_does_not"])
        self.assertFalse(result["graph"]["rule_4_built_equals_before"])
        self.assertEqual(
            result["graph"]["verdict_on_the_built_path"], "the built path does not close graph"
        )
        self.assertEqual(summary["graph"]["first_failing_seed"], SEED)

    @staticmethod
    def shown(outcome, world):
        return [row_[0] for row_ in outcome["worlds"][world]["arms"]["built"]["observation"]]


class ReportTests(unittest.TestCase):
    def test_main_writes_json_and_markdown_and_prints_the_markdown(self):
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with mock.patch.object(link_paths, "run_trial", synthetic_trial):
                with contextlib.redirect_stdout(printed):
                    self.assertEqual(
                        link_paths.main(["--trials", "3", "--output", str(output)]), 0
                    )
            report = json.loads(output.with_suffix(".json").read_text())
            markdown = output.with_suffix(".md").read_text()
        self.assertEqual(printed.getvalue().strip(), markdown.strip())
        self.assertEqual(report["trials"], 3)
        self.assertEqual(set(report["decision"]), set(PATHS))
        for path in PATHS:
            self.assertIn(
                "**Verdict: the baseline leaks through {0}; the built path closes {0}**".format(path),
                markdown,
            )

    def test_trials_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                link_paths.main(["--trials", "0", "--output", "unused"])


if __name__ == "__main__":
    unittest.main()
