"""Convert LongMemEval questions into Memora evaluation sets.

LongMemEval (Wu et al., ICLR 2025; MIT licence) gives every question its own
timestamped chat history. Each question becomes one evaluation set:

- memories: the user's turns, oldest first. Assistant turns are left out, so
  questions whose evidence is only in assistant turns are skipped.
- links: each user turn is ``related_to`` the user's next turn in the same
  session. They come from the conversation's structure, not from any label.
- queries: the question. For knowledge-update questions, ``relevant`` lists
  the evidence turns of the most recent evidence session and ``stale`` lists
  the evidence turns of earlier sessions, which hold the outdated answer. For
  other question types every evidence turn is relevant.
- updates (knowledge-update questions only): the first current evidence turn
  replaces each stale one. These come from the benchmark's evidence labels,
  so they are an oracle: perfect update detection, which a real system would
  have to achieve on its own.

Abstention questions (ids ending in ``_abs``) have no evidence and are skipped.

The rule "newest evidence is current" does not fit every knowledge-update
question: some ask for the previous value, some need both versions. A labels
file (``eval/longmemeval_knowledge_update_labels.json``) records a category for
each question with an older and a newer version, and the conversion follows it:

- current, latest_event: as above;
- previous: the older evidence is relevant; the update still applies;
- both: all evidence is relevant; the update still applies;
- context: the newer evidence is relevant; no update (the older turn is not an
  older version of the fact);
- exclude: skipped.

The category is added as a second tag, so reports break results down by it.

Usage::

    python -m dynmemory.longmemeval data/longmemeval/longmemeval_s_cleaned.json \\
        --labels eval/longmemeval_knowledge_update_labels.json \\
        --out data/longmemeval/knowledge_update.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .evaluation import load_eval_set

KNOWLEDGE_UPDATE = "knowledge-update"
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
CATEGORIES = ("current", "previous", "both", "latest_event", "context", "exclude")


def load_labels(path: str) -> dict:
    """Read a labels file: ``{"labels": {question_id: {"category": ...}}}``."""
    labels = json.loads(Path(path).read_text(encoding="utf-8")).get("labels")
    if not isinstance(labels, dict):
        raise ValueError("{}: expected a 'labels' object".format(path))
    for question_id, label in labels.items():
        if not isinstance(label, dict) or label.get("category") not in CATEGORIES:
            raise ValueError(
                "{}: {}: category must be one of {}".format(path, question_id, ", ".join(CATEGORIES))
            )
    return labels


def convert_question(
    question: dict, *, updates: bool = True, label: Optional[dict] = None
) -> tuple[Optional[dict], str]:
    """Return ``(eval_set, "converted")`` or ``(None, reason it was skipped)``."""
    if question["question_id"].endswith("_abs"):
        return None, "abstention question"
    category = label["category"] if label else None
    if category == "exclude":
        return None, "excluded by label"
    order = sorted(
        range(len(question["haystack_sessions"])),
        key=lambda index: (
            datetime.strptime(question["haystack_dates"][index], DATE_FORMAT),
            index,
        ),
    )
    memories, links, evidence = {}, [], []
    for session_rank, session_index in enumerate(order):
        session_id = question["haystack_session_ids"][session_index]
        previous_key = None
        for turn_index, turn in enumerate(question["haystack_sessions"][session_index]):
            content = turn["content"].strip()
            if turn["role"] != "user" or not content:
                continue
            key = "{}:{}".format(session_id, turn_index)
            if key in memories:
                key = "{}#{}".format(key, session_rank)
            memories[key] = {"content": content, "kind": "episode"}
            if previous_key is not None:
                links.append({"source": previous_key, "target": key, "relation": "related_to"})
            previous_key = key
            if turn.get("has_answer"):
                evidence.append((session_rank, key))
    if not evidence:
        return None, "no evidence in user turns"

    query = {"query": question["question"], "tags": [question["question_type"]]}
    eval_set = {
        "id": question["question_id"],
        "source": {
            "benchmark": "LongMemEval",
            "question_type": question["question_type"],
            "question_date": question["question_date"],
            "answer": question["answer"],
        },
        "memories": memories,
        "links": links,
        "queries": [query],
    }
    if question["question_type"] == KNOWLEDGE_UPDATE:
        latest = max(session_rank for session_rank, _ in evidence)
        current = [key for session_rank, key in evidence if session_rank == latest]
        older = [key for session_rank, key in evidence if session_rank != latest]
        if category in (None, "current", "latest_event"):
            query["relevant"], query["stale"] = current, older
        elif category == "previous":
            query["relevant"] = older
        elif category == "both":
            query["relevant"] = older + current
        else:
            query["relevant"] = current
        if category is not None:
            query["tags"].append(category)
        if updates and older and category != "context":
            eval_set["updates"] = [{"old": key, "new": current[0]} for key in older]
    else:
        query["relevant"] = [key for _, key in evidence]
    load_eval_set(eval_set)
    return eval_set, "converted"


def convert(
    questions: Sequence[dict],
    *,
    question_types: Optional[Sequence[str]],
    updates: bool = True,
    labels: Optional[dict] = None,
) -> tuple[list[dict], Counter]:
    """Convert the questions of the given types; also count why others were skipped.

    With ``labels``, every knowledge-update question that has an older and a
    newer version must have a label.
    """
    eval_sets, outcomes = [], Counter()
    for question in questions:
        if question_types and question["question_type"] not in question_types:
            continue
        label = labels.get(question["question_id"]) if labels is not None else None
        try:
            eval_set, outcome = convert_question(question, updates=updates, label=label)
        except ValueError as error:
            raise ValueError("{}: {}".format(question["question_id"], error)) from error
        if labels is not None and label is None and eval_set is not None and any(
            eval_set["queries"][0].get("stale", [])
        ):
            raise ValueError(
                "{}: has an older and a newer version but no label".format(
                    question["question_id"]
                )
            )
        outcomes[outcome] += 1
        if eval_set is not None:
            eval_sets.append(eval_set)
    return eval_sets, outcomes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dynmemory.longmemeval",
        description="Convert LongMemEval questions into a Memora evaluation suite (.jsonl).",
    )
    parser.add_argument("input", help="a LongMemEval JSON file, e.g. longmemeval_s_cleaned.json")
    parser.add_argument("--out", required=True, help="output .jsonl file")
    parser.add_argument(
        "--types",
        default=KNOWLEDGE_UPDATE,
        help="comma-separated question types, or 'all' (default: knowledge-update)",
    )
    parser.add_argument(
        "--no-updates",
        action="store_true",
        help="leave out the oracle updates for knowledge-update questions",
    )
    parser.add_argument(
        "--labels",
        help="labels file with a category for each question that has an older and a newer version",
    )
    args = parser.parse_args(argv)

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    question_types = None if args.types == "all" else args.types.split(",")
    eval_sets, outcomes = convert(
        questions,
        question_types=question_types,
        updates=not args.no_updates,
        labels=load_labels(args.labels) if args.labels else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for eval_set in eval_sets:
            handle.write(json.dumps(eval_set, ensure_ascii=False) + "\n")
    for outcome, count in sorted(outcomes.items()):
        print("{}: {}".format(outcome, count))
    print("wrote {} evaluation sets to {}".format(len(eval_sets), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
