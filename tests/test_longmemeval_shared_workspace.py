import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "longmemeval_shared_workspace.py"
spec = importlib.util.spec_from_file_location("longmemeval_shared_workspace_experiment", SCRIPT)
sw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sw
spec.loader.exec_module(sw)


def setUpModule():
    sw.install_clock()


def tearDownModule():
    sw.restore_clock()


COMMON = "project plan weekend dinner garden music trip budget".split()


def fixture_questions(count=24):
    rng = random.Random("fixture")
    questions = []
    for number in range(count):
        sessions, ids, dates = [], [], []
        for session in range(5):
            turns = []
            for turn in range(2):
                words = rng.sample(COMMON, 4) + ["w{}x{}".format(number, rng.randint(0, 9))]
                text = "I talked about " + " ".join(words)
                if session == 1 and turn == 0:
                    text = "My ev{} answer is blue and I mention {} too".format(number, COMMON[number % 8])
                turns.append({"role": "user", "content": text,
                              "has_answer": session == 1 and turn == 0})
                turns.append({"role": "assistant", "content": "ok", "has_answer": False})
            ids.append("s{}_{}".format(number, session))
            dates.append("2023/05/{:02d} (Sat) 10:00".format(session + 1))
            sessions.append(turns)
        # a session shared verbatim by every question, and one identical turn
        sessions.append([{"role": "user", "content": "I went hiking last weekend with my brother.",
                          "has_answer": False}])
        ids.append("shared_hike")
        dates.append("2023/05/09 (Tue) 10:00")
        questions.append({
            "question_id": "q{:02d}".format(number),
            "question": "What was ev{} about {}?".format(number, COMMON[number % 8]),
            "answer": "blue", "question_type": "single-session-user",
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_session_ids": ids, "haystack_dates": dates, "haystack_sessions": sessions,
        })
    questions.append(dict(questions[0], question_id="q00_abs"))
    return questions


QUESTIONS = fixture_questions()
PARSED = sw.parse_questions(QUESTIONS)
ELIGIBLE = sorted(PARSED)
POOL = sw.pool_of(ELIGIBLE)
SPECS = sw.workspaces_of(POOL, ELIGIBLE, 0, 3)


def summary(**overrides):
    base = {
        "totals": {"leak": 0, "link_leak": 0, "list_in": 0, "get_in": 0, "list_out": 0,
                   "get_out": 0, "canary_missed": 0},
        "classes": {"exact": 5},
        "controls": {1: True, 2: True, 3: True, 4: True, 5: True},
        "utility": {"differs_share": 0.3, "built_minus_after_all": {"mean": 0.0, "low": -0.005, "high": 0.005}},
        "first_failure": {1: {"family": "full", "seed": 0, "workspace": 3, "item": "x"},
                          2: {"family": "full", "seed": 0, "workspace": 4, "item": "y"},
                          3: {"family": "full", "seed": 1, "workspace": 5, "item": "z"}},
    }
    for key, value in overrides.items():
        if key in base["totals"]:
            base["totals"][key] = value
        else:
            base[key] = value
    return base


class DecisionRuleTests(unittest.TestCase):
    def test_all_rules_holding_with_a_flat_utility_passes(self):
        verdict = sw.decide(summary())["verdict"]
        self.assertEqual(
            verdict, "zero leakage, exactness and no over-blocking supported; utility no cost")

    def test_a_confirmed_leak_is_a_failure_whatever_the_controls_say(self):
        decision = sw.decide(summary(leak=2, controls={1: False, 2: True, 3: True, 4: True, 5: True}))
        self.assertTrue(decision["verdict"].startswith("fails (rules 1 failed, first failing full seed 0 workspace 3"))

    def test_a_failed_control_makes_a_clean_run_inconclusive(self):
        decision = sw.decide(summary(controls={1: True, 2: False, 3: True, 4: False, 5: True}))
        self.assertEqual(decision["verdict"], "inconclusive (control 2, 4 failed)")

    def test_exactness_and_overblocking_name_their_rules(self):
        decision = sw.decide(summary(classes={"exact": 3, "order": 1}, list_out=1))
        self.assertIn("rules 2, 3 failed", decision["verdict"])

    def test_utility_readings(self):
        flat = {"differs_share": 0.3}

        def reading(share, low, high):
            return sw.utility_reading({
                "differs_share": share,
                "built_minus_after_all": {"mean": 0, "low": low, "high": high}})

        self.assertEqual(reading(0.3, -0.01, 0.01), "no cost")
        self.assertEqual(reading(0.3, -0.05, -0.03), "cost")
        self.assertEqual(reading(0.3, -0.05, 0.0), "unclear")
        self.assertTrue(reading(0.01, -0.01, 0.01).startswith("insensitive"))
        self.assertEqual(sw.utility_reading(None), "not measured")
        self.assertIsNotNone(flat)

    def test_skipped_controls_are_named(self):
        decision = sw.decide(summary(controls={1: None, 2: True, 3: True, 4: None, 5: True}))
        self.assertTrue(decision["verdict"].endswith("(controls not run: 1, 4)"))


class PlanTests(unittest.TestCase):
    def test_a_plan_is_a_function_of_the_data_and_the_seed(self):
        one = sw.plan_workspace(PARSED, SPECS[0], "full")
        two = sw.plan_workspace(PARSED, SPECS[0], "full")
        self.assertEqual(one["writes"], two["writes"])
        self.assertEqual(one["links"], two["links"])
        self.assertEqual(one["post_items"], two["post_items"])

    def test_placement_is_intended(self):
        for spec_ in SPECS:
            plan = sw.plan_workspace(PARSED, spec_, "full")
            self.assertEqual(sw.check_plan(plan), [])
            for member in plan["members"]:
                self.assertEqual(
                    {plan["channel_of"][label] for label in plan["gold"][member]},
                    {sw.private(member)})
            self.assertEqual({plan["channel_of"][l] for l in plan["gold"]["S"]}, {sw.GENERAL})
            self.assertEqual({plan["channel_of"][l] for l in plan["gold"]["T"]}, {sw.TEAM})

    def test_extras_are_not_owned_in_the_workspace_and_roles_differ(self):
        for spec_ in SPECS:
            self.assertFalse(set(spec_["extras"]) & set(spec_["owners"]))
            plan = sw.plan_workspace(PARSED, spec_, "full")
            self.assertNotEqual(plan["joiner"], plan["leaver"])

    def test_the_outsider_memory_is_spoken_by_a_non_member_of_its_channel(self):
        plan = sw.plan_workspace(PARSED, SPECS[0], "full")
        write = next(w for w in plan["writes"] if w["label"] == "outsider")
        self.assertNotIn(write["speaker"], plan["post"][write["channel"]])

    def test_writers_of_shared_channels_are_current_members(self):
        for spec_ in SPECS:
            plan = sw.plan_workspace(PARSED, spec_, "full")
            for write in plan["writes"]:
                if write["channel"] in (sw.GENERAL,):
                    self.assertNotEqual(write["speaker"], plan["joiner"])
                if write["channel"] == sw.TEAM:
                    self.assertIn(write["speaker"], sw.TEAM_MEMBERS)

    def test_canary_words_are_absent_from_every_question(self):
        self.assertEqual(set(sw.INVENTED) & sw.vocabulary(PARSED), set())
        self.assertFalse(any(word.startswith("zq") for word in sw.vocabulary(PARSED)))

    def test_links_cross_visibility_boundaries(self):
        plan = sw.plan_workspace(PARSED, SPECS[0], "full")
        crossing = [
            (a, b) for a, b, _ in plan["links"] if plan["channel_of"][a] != plan["channel_of"][b]
        ]
        self.assertGreaterEqual(len(crossing), len(plan["members"]))

    def test_pool_is_a_multiple_of_eight_and_pairs_with_the_data(self):
        self.assertEqual(len(POOL) % 8, 0)
        self.assertEqual(len(ELIGIBLE), 24)
        self.assertNotIn("q00_abs", PARSED)

    def test_visibility_follows_the_record(self):
        plan = sw.plan_workspace(PARSED, SPECS[0], "full")
        leaver, joiner = plan["leaver"], plan["joiner"]
        before = sw.visible_labels(plan, plan["pre"], (joiner,))
        after = sw.visible_labels(plan, plan["post"], (joiner,))
        general = {l for l, c in plan["channel_of"].items() if c == sw.GENERAL}
        self.assertFalse(general & before)
        self.assertTrue(general <= after)
        self.assertFalse(general & sw.visible_labels(plan, plan["post"], (leaver,)))


class RunTests(unittest.TestCase):
    def test_the_current_store_passes_a_full_workspace(self):
        for spec_ in SPECS:
            result = sw.run_workspace(PARSED, spec_, "full")
            self.assertEqual(result["leak"] + result["link_leak"], 0, result["failures"])
            self.assertEqual(result["list_in"] + result["get_in"], 0)
            self.assertEqual(result["list_out"] + result["get_out"] + result["canary_missed"], 0)
            self.assertEqual(set(result["classes"]), {"exact"})
            self.assertEqual(result["placement_errors"] + result["plan_errors"], 0)
            self.assertGreater(result["canary_items"], 0)
            self.assertEqual(result["pooled_canary"], result["pooled_canary_rank1"])

    def test_the_leak_probe_has_power_on_the_pooled_arm(self):
        results = [sw.run_workspace(PARSED, spec_, "full") for spec_ in SPECS]
        other = sum(r["pooled_other"] for r in results)
        self.assertGreater(other, 0)
        self.assertGreaterEqual(sum(r["pooled_other_hidden"] for r in results), 0.95 * other)

    def test_hybrid_and_bare_runs_pass(self):
        hybrid = sw.run_workspace(PARSED, SPECS[0], "full", family="hybrid", hybrid=True)
        self.assertEqual(hybrid["leak"] + hybrid["list_in"] + hybrid["get_in"], 0)
        self.assertEqual(set(hybrid["classes"]), {"exact"})
        bare = sw.run_workspace(PARSED, SPECS[0], "bare", family="bare8")
        self.assertEqual(set(bare["classes"]), {"exact", "solo-exact"})
        single = sw.run_workspace(
            PARSED, {"seed": 0, "index": 0, "owners": ["q05"], "extras": []}, "bare", family="k1")
        self.assertEqual(set(single["classes"]), {"exact", "solo-exact"})

    def test_a_run_is_a_function_of_its_inputs(self):
        one = sw.run_workspace(PARSED, SPECS[1], "full")
        two = sw.run_workspace(PARSED, SPECS[1], "full")
        self.assertEqual(sw.digest(one), sw.digest(two))
        json.dumps(one)

    def test_clock_is_monotone(self):
        stamps = [sw.MemoryStore._now() for _ in range(5)]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(set(stamps)), 5)


class FaultTests(unittest.TestCase):
    def run_fault(self, name, workspaces=2):
        results = [sw.run_workspace(PARSED, spec_, "full", flaw=name) for spec_ in SPECS[:workspaces]]
        return results, sw.validate_fault(name, results)

    def test_each_fault_shows_its_pattern(self):
        for name in sw.FAULTS:
            results, validation = self.run_fault(name)
            self.assertTrue(validation["holds"], (name, validation, [r["failures"] for r in results]))

    def test_faults_trip_only_the_predicted_rule_where_stated(self):
        results, _ = self.run_fault("overblock-joiner")
        self.assertTrue(all(r["leak"] == 0 and r["list_in"] == 0 for r in results))
        results, _ = self.run_fault("speaker-sees")
        self.assertTrue(all(r["leak_foreign"] == 0 for r in results))
        self.assertTrue(all("outsider-self" in r["leak_kinds"] for r in results))
        results, _ = self.run_fault("union")
        for result in results:
            self.assertTrue(set(result["exact_kinds"]) <= sw.MIXED_KINDS)

    def test_a_fault_is_removed_afterwards_and_an_unknown_one_refused(self):
        before = sw.MemoryStore.__dict__["_visible"]
        with sw.fault("union"):
            self.assertIsNot(sw.MemoryStore.__dict__["_visible"], before)
        self.assertIs(sw.MemoryStore.__dict__["_visible"], before)
        with self.assertRaises(ValueError):
            with sw.fault("no-such-fault"):
                pass


class MainTests(unittest.TestCase):
    def test_digest_is_the_same_under_different_hash_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            data.write_text(json.dumps(QUESTIONS))
            self.assertTrue(sw.determinism_check(data))

    def test_main_writes_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            data.write_text(json.dumps(QUESTIONS))
            output = Path(directory) / "report"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                sw.main([
                    "--data", str(data), "--seeds", "1", "--workspaces", "2", "--bare", "1",
                    "--hybrid", "1", "--k1", "3", "--output", str(output),
                ])
            report = json.loads(output.with_suffix(".json").read_text())
            text = output.with_suffix(".md").read_text()
        self.assertIn(report["decision"]["verdict"], text)
        self.assertEqual(printed.getvalue().strip(), text.strip())
        self.assertEqual(report["summary"]["families"],
                         {"full": 2, "bare8": 1, "hybrid": 1, "k1": 3})
        self.assertTrue(report["decision"]["verdict"].startswith("zero leakage"))
        self.assertTrue(report["decision"]["controls"]["4"])

    def test_a_faulty_run_is_reported_as_a_failure_with_its_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            data.write_text(json.dumps(QUESTIONS))
            output = Path(directory) / "fault"
            with contextlib.redirect_stdout(io.StringIO()):
                sw.main([
                    "--data", str(data), "--seeds", "1", "--workspaces", "2", "--bare", "0",
                    "--hybrid", "0", "--k1", "0", "--no-determinism", "--fault", "union",
                    "--output", str(output),
                ])
            report = json.loads(output.with_suffix(".json").read_text())
        self.assertTrue(report["decision"]["verdict"].startswith("fails (rules 1"))
        self.assertTrue(report["validation"]["holds"])


if __name__ == "__main__":
    unittest.main()
