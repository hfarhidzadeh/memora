import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from dynmemory.evaluation import load_eval_suite, retrieval_report
from dynmemory.longmemeval import convert, convert_question, load_labels, main

# A made-up question in LongMemEval's format; sessions are listed out of date order.
QUESTION = {
    "question_id": "q1",
    "question_type": "knowledge-update",
    "question": "What is my personal best time?",
    "answer": "25:50",
    "question_date": "2023/06/01 (Thu) 00:58",
    "haystack_session_ids": ["late", "early", "filler"],
    "haystack_dates": [
        "2023/05/27 (Sat) 10:20",
        "2023/05/25 (Thu) 20:21",
        "2023/05/26 (Fri) 09:00",
    ],
    "haystack_sessions": [
        [
            {"role": "user", "content": "Training again; hoping to beat my 25:50.", "has_answer": True},
            {"role": "assistant", "content": "Good luck!"},
        ],
        [
            {"role": "user", "content": "I ran a charity race."},
            {"role": "assistant", "content": "Nice."},
            {"role": "user", "content": "My personal best time is 27:12.", "has_answer": True},
        ],
        [
            {"role": "user", "content": "Recommend a book."},
            {"role": "assistant", "content": "Try Dune."},
            {"role": "user", "content": "   "},
        ],
    ],
    "answer_session_ids": ["late", "early"],
}


def question(**changes):
    changed = copy.deepcopy(QUESTION)
    changed.update(changes)
    return changed


class ConvertQuestionTests(unittest.TestCase):
    def test_user_turns_become_memories_in_date_order(self):
        eval_set, outcome = convert_question(QUESTION)

        self.assertEqual(outcome, "converted")
        self.assertEqual(list(eval_set["memories"]), ["early:0", "early:2", "filler:0", "late:0"])
        self.assertEqual(
            eval_set["links"],
            [{"source": "early:0", "target": "early:2", "relation": "related_to"}],
        )

    def test_knowledge_update_splits_current_and_stale_evidence(self):
        eval_set, _ = convert_question(QUESTION)

        query = eval_set["queries"][0]
        self.assertEqual(
            (query["relevant"], query["stale"], query["tags"]),
            (["late:0"], ["early:2"], ["knowledge-update"]),
        )
        self.assertEqual(eval_set["updates"], [{"old": "early:2", "new": "late:0"}])
        self.assertNotIn("updates", convert_question(QUESTION, updates=False)[0])

    def test_other_types_count_all_evidence_as_relevant(self):
        eval_set, _ = convert_question(question(question_type="multi-session"))

        query = eval_set["queries"][0]
        self.assertEqual(query["relevant"], ["early:2", "late:0"])
        self.assertNotIn("stale", query)
        self.assertNotIn("updates", eval_set)

    def test_unscorable_questions_are_skipped_with_a_reason(self):
        self.assertEqual(convert_question(question(question_id="q1_abs")), (None, "abstention question"))
        assistant_only = question()
        for session in assistant_only["haystack_sessions"]:
            for turn in session:
                turn.pop("has_answer", None)
        assistant_only["haystack_sessions"][0][1]["has_answer"] = True
        self.assertEqual(convert_question(assistant_only), (None, "no evidence in user turns"))

    def test_convert_filters_by_type_and_counts_outcomes(self):
        questions = [QUESTION, question(question_id="q2_abs"), question(question_type="multi-session")]

        eval_sets, outcomes = convert(questions, question_types=["knowledge-update"])

        self.assertEqual([eval_set["id"] for eval_set in eval_sets], ["q1"])
        self.assertEqual(dict(outcomes), {"converted": 1, "abstention question": 1})


class LabelTests(unittest.TestCase):
    def converted(self, category):
        return convert_question(QUESTION, label={"category": category})

    def test_categories_decide_relevance_and_updates(self):
        update = [{"old": "early:2", "new": "late:0"}]
        expected = {
            "current": (["late:0"], ["early:2"], update),
            "latest_event": (["late:0"], ["early:2"], update),
            "previous": (["early:2"], None, update),
            "both": (["early:2", "late:0"], None, update),
            "context": (["late:0"], None, None),
        }
        for category, (relevant, stale, updates) in expected.items():
            eval_set, outcome = self.converted(category)
            query = eval_set["queries"][0]
            self.assertEqual(outcome, "converted", category)
            self.assertEqual(
                (query["relevant"], query.get("stale"), eval_set.get("updates")),
                (relevant, stale, updates),
                category,
            )
            self.assertEqual(query["tags"], ["knowledge-update", category])
        self.assertEqual(self.converted("exclude"), (None, "excluded by label"))

    def test_update_questions_need_a_label_when_labels_are_given(self):
        with self.assertRaisesRegex(ValueError, "q1: has an older and a newer version but no label"):
            convert([QUESTION], question_types=None, labels={})
        eval_sets, _ = convert(
            [QUESTION], question_types=None, labels={"q1": {"category": "previous"}}
        )
        self.assertEqual(eval_sets[0]["queries"][0]["relevant"], ["early:2"])

    def test_labels_file_rejects_unknown_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(json.dumps({"labels": {"q1": {"category": "maybe"}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "q1: category must be one of"):
                load_labels(str(path))


class ConvertedSuiteTests(unittest.TestCase):
    def test_command_writes_a_suite_that_memora_can_evaluate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "questions.json"
            source.write_text(json.dumps([QUESTION, question(question_id="q3")]), encoding="utf-8")
            suite = Path(temp_dir) / "suite.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), "--out", str(suite)]), 0)

            eval_sets = load_eval_suite(suite)

        self.assertEqual(len(eval_sets), 2)
        report = retrieval_report(eval_sets, seeds=1)
        metrics = {
            configuration["name"]: configuration["metrics"]["all"]
            for configuration in report["configurations"]
        }
        self.assertEqual(report["sets"], 2)
        self.assertEqual(metrics["lexical"]["queries"], 2)
        # Without the update, the old "personal best time" turn outranks the new one.
        self.assertEqual(metrics["lexical, updates not applied"]["stale_first"], 1.0)
        self.assertEqual(metrics["lexical"]["stale_first"], 0.0)


if __name__ == "__main__":
    unittest.main()
