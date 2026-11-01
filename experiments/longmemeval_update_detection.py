"""Update detection on LongMemEval: can a small local model find updates itself?

Question. The knowledge-update experiment showed that knowing which memory
replaces which is the biggest lever (current-answer reciprocal rank 0.563 ->
0.720), but it used the benchmark's evidence labels as an oracle. When a new
memory arrives, can a small local language model decide which earlier memory
(if any) it replaces, and how much of the oracle's gain does that recover?

Data. The 65 knowledge-update questions of ``data/longmemeval/knowledge_update.jsonl``
that the labels file classes as ``current`` (50, the primary group), ``previous``
(9) or ``both`` (6). One store per question, its memories the user's turns (about
260), no links. The oracle's update is "the first current evidence turn replaces
each older evidence turn".

Detector. When a memory N is ingested the system sees only the memories before it.
The five best BM25 matches for N's text are the candidates, and a local model
(Ollama, temperature 0) answers YES or NO for each (earlier, later) pair: "does the
later message give a new value for the same fact, so the earlier value is out of
date?". A YES supersedes the candidate. Messages are cut to 600 characters. The
prompt was written once and checked only on made-up pairs, never on benchmark data.

Which turns are judged. Judging every turn of every stream is too slow on this
machine, so the detector runs on N (the oracle's newer evidence turn) and on
four random control turns per question (not evidence turns, at least 30 memories
into the stream), each against its three best candidates. Control turns measure
false supersessions in ordinary traffic; their decisions are applied too, so
their harm shows in the retrieval numbers.

Configurations, all without links:
- none: no updates (lower bound);
- oracle: the benchmark's updates (upper bound);
- detector (evidence turn only): the YES decisions for N;
- detector (all judged turns): also the YES decisions for the control turns;
- random (negative control): the same new turns as the detector's YES decisions,
  each replacing a random earlier memory instead, averaged over 10 seeds.
  Archiving random memories removes distractors, so it can help a little by
  itself; the detector has to beat it.

Other controls. Positive: the oracle. Negative: two random (earlier, N) pairs per
question must almost never get YES, or the model says YES to everything.

Decision rule, fixed before running:
  1. Validity. The YES rate on random pairs must be at most 10%. Otherwise the
     result is inconclusive. If fewer than 70% of the oracle's older turns are
     among the five candidates, candidate search, not the model, is the limit;
     the numbers are still reported.
  2. Recovery. On the 50 "current" questions the detector (all judged turns) must
     recover at least half of the oracle's reciprocal-rank gain over none, improve
     on none with two-sided sign-flip p < 0.05, and improve on random archiving
     with p < 0.05.
  3. Safety. At most 5% of the candidates of control turns may get YES, and on the
     15 "previous"/"both" questions the detector may lower reciprocal rank by no
     more than 0.05 more than the oracle does.
  The detector is "usable" only if it meets 1 to 3.

Limits. Controls apply the detector to 4 of about 260 turns per stream; the
expected number of false supersessions per stream is the control rate times 3
candidates times 260 turns, reported separately. Unlabelled genuine updates among
the other candidates count as false positives, so false-positive rates are upper
bounds. Updates come from a model the benchmark does not know, so some YES
decisions on non-oracle pairs may be correct.

Status: the full run has not been completed, so no results are published yet.

Run from the project root (Ollama must be serving the model):

    python3 -I experiments/longmemeval_update_detection.py --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore  # noqa: E402
from dynmemory.evaluation import load_eval_set, load_eval_suite, query_rows  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


knowledge_update = _load_script("longmemeval_knowledge_update")
silence = _load_script("longmemeval_silence")

CATEGORIES = ("current", "previous", "both")
PRIMARY = "current"
CANDIDATES = 5
CONTROL_CANDIDATES = 3
CONTROL_TURNS = 4
RANDOM_PAIRS = 2
MIN_POSITION = 30
MAX_CHARS = 600
SEEDS = 10
SEED = 0
STREAM_LENGTH = 260
RANDOM_YES_LIMIT = 0.10
CANDIDATE_RECALL_MINIMUM = 0.70
RECOVERY_MINIMUM = 0.5
CONTROL_YES_LIMIT = 0.05
HARM_MARGIN = 0.05
ALPHA = 0.05
DEPTH = knowledge_update.DEPTH

PROMPT = """You are checking a user's memory notes for outdated facts.

EARLIER message from the user:
\"\"\"{earlier}\"\"\"

LATER message from the user:
\"\"\"{later}\"\"\"

Does the LATER message give a new value for the same fact that the EARLIER message stated, so that the earlier value is now out of date? A new value can be a changed number or count, a new date or time, a new place or a changed status. Answer NO if the messages only share a topic, describe two different events that can both be true, or the later message only adds to the earlier one.

Answer with one word: YES or NO."""


# --- the judge -------------------------------------------------------------------


def clip(text):
    text = " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rstrip() + " ..."


def build_prompt(earlier, later):
    return PROMPT.format(earlier=clip(earlier), later=clip(later))


def parse_answer(raw):
    words = raw.strip().lower().lstrip("\"'*`").split()
    first = words[0].rstrip(".,!:;\"'*`") if words else ""
    return "YES" if first == "yes" else "NO" if first == "no" else "INVALID"


class OllamaJudge:
    """Ask a local model YES/NO for (earlier, later) pairs; answers are cached on disk."""

    def __init__(self, model, cache_path, *, host="http://127.0.0.1:11434", post=None):
        self.model = model
        self.cache_path = Path(cache_path)
        self.host = host
        self._post = post or self._http_post
        self.entries = {}
        self.new_calls = 0
        self.new_seconds = 0.0
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:  # a line cut short by an interrupted run
                    continue
                self.entries[entry["key"]] = entry

    def _http_post(self, prompt):
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": 4, "seed": SEED},
        }).encode()
        request = urllib.request.Request(
            self.host + "/api/generate", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["response"]

    def __call__(self, earlier, later, label=None):
        prompt = build_prompt(earlier, later)
        key = hashlib.sha256((self.model + "\0" + prompt).encode()).hexdigest()
        if key not in self.entries:
            started = time.time()
            raw = self._post(prompt)
            seconds = time.time() - started
            entry = {"key": key, "model": self.model, "label": label, "raw": raw,
                     "answer": parse_answer(raw), "seconds": round(seconds, 2)}
            self.entries[key] = entry
            self.new_calls += 1
            self.new_seconds += seconds
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        return self.entries[key]["answer"]


# --- design ----------------------------------------------------------------------


def top_candidates(eval_set, new_key, count):
    """The ``count`` best BM25 matches for a memory among the memories before it."""
    keys = list(eval_set["memories"])
    earlier = keys[: keys.index(new_key)]
    if not earlier:
        return []
    store = MemoryStore(":memory:")
    key_of = {}
    for key in earlier:
        memory = eval_set["memories"][key]
        key_of[store.add_memory(memory["content"], memory["kind"]).id] = key
    results = store.search(
        eval_set["memories"][new_key]["content"], limit=count, record_access=False
    )
    store.close()
    return [key_of[result.memory.id] for result in results]


def control_turns(eval_set, exclude, count, seed):
    """Random turns that are not evidence and sit at least MIN_POSITION into the stream."""
    keys = [
        key for position, key in enumerate(eval_set["memories"])
        if position >= MIN_POSITION and key not in exclude
    ]
    return random.Random(seed).sample(keys, min(count, len(keys)))


def valid_updates(decisions, position):
    """One update per replaced memory (the earliest newer turn wins), in stream order."""
    chosen = {}
    for new, old in decisions:
        if position[new] <= position[old]:
            continue
        if old not in chosen or position[new] < position[chosen[old]]:
            chosen[old] = new
    return [{"old": old, "new": new}
            for old, new in sorted(chosen.items(), key=lambda pair: (position[pair[1]], position[pair[0]]))]


def random_updates(decisions, keys, seed):
    """The same newer turns, each replacing a random earlier memory (negative control)."""
    rng = random.Random(seed)
    position = {key: index for index, key in enumerate(keys)}
    return valid_updates(
        [(new, rng.choice(keys[: position[new]])) for new, _ in decisions if position[new] > 0],
        position,
    )


def judge_stream(eval_set, judge, category):
    """Judge N's candidates, the control turns' candidates and random pairs for one question."""
    question_id = eval_set["id"]
    memories = eval_set["memories"]
    keys = list(memories)
    updates = eval_set["updates"]
    new_key = updates[0]["new"]
    olds = [update["old"] for update in updates]
    if any(update["new"] != new_key for update in updates):
        raise ValueError("{}: the oracle has several newer turns".format(question_id))
    judged = []

    def ask(kind, new, old):
        answer = judge(memories[old]["content"], memories[new]["content"],
                       label=[question_id, kind, old, new])
        judged.append({"kind": kind, "new": new, "old": old, "answer": answer,
                       "oracle": new == new_key and old in olds,
                       "clipped": max(len(memories[old]["content"]), len(memories[new]["content"])) > MAX_CHARS})

    candidates = top_candidates(eval_set, new_key, CANDIDATES)
    for old in candidates:
        ask("evidence", new_key, old)
    for turn in control_turns(eval_set, {new_key, *olds}, CONTROL_TURNS, "{}:{}".format(SEED, question_id)):
        for old in top_candidates(eval_set, turn, CONTROL_CANDIDATES):
            ask("control", turn, old)
    rng = random.Random("{}:{}:random".format(SEED, question_id))
    earlier = [k for k in keys[: keys.index(new_key)] if k not in olds and k not in candidates]
    for old in rng.sample(earlier, min(RANDOM_PAIRS, len(earlier))):
        ask("random", new_key, old)
    return {"id": question_id, "category": category, "oracle_olds": olds, "new": new_key,
            "candidates": candidates, "judged": judged}


# --- analysis --------------------------------------------------------------------


def report_rate(judged, kind):
    selection = [j for j in judged if j["kind"] == kind]
    return sum(j["answer"] == "YES" for j in selection) / len(selection) if selection else None


def category_of(eval_set):
    tags = eval_set["queries"][0]["tags"]
    return tags[1] if len(tags) > 1 else "unlabelled"


def recovery(none, oracle, detector, seed=0):
    """Share of the oracle's mean gain over none that the detector recovers, with a bootstrap CI."""
    gains = [(o - n, d - n) for n, o, d in zip(none, oracle, detector)]
    ratio = lambda rows: (sum(d for _, d in rows) / sum(o for o, _ in rows)
                          if abs(sum(o for o, _ in rows)) > 1e-12 else None)
    rng = random.Random(seed)
    draws = sorted(r for r in (ratio([rng.choice(gains) for _ in gains])
                               for _ in range(knowledge_update.BOOTSTRAP_SAMPLES)) if r is not None)
    return {"ratio": ratio(gains),
            "ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]] if draws else None}


def rates(judged):
    def share(selection):
        yes = sum(item["answer"] == "YES" for item in selection)
        return {"yes": yes, "n": len(selection),
                "rate": yes / len(selection) if selection else None,
                "ci95": silence.wilson(yes, len(selection))}

    return {
        "evidence_non_oracle": share([j for j in judged if j["kind"] == "evidence" and not j["oracle"]]),
        "control": share([j for j in judged if j["kind"] == "control"]),
        "random": share([j for j in judged if j["kind"] == "random"]),
        "oracle_pairs": share([j for j in judged if j["oracle"]]),
        "invalid": sum(j["answer"] == "INVALID" for j in judged),
        "clipped_oracle_pairs": sum(j["clipped"] for j in judged if j["oracle"]),
    }


def verdict(report):
    if report["pair_rates"]["random"]["rate"] is None or report["pair_rates"]["random"]["rate"] > RANDOM_YES_LIMIT:
        return "inconclusive: the model says YES to random pairs"
    problems = []
    if report["candidate_recall"] < CANDIDATE_RECALL_MINIMUM:
        problems.append("candidate search finds too few of the oracle's older turns")
    current = report["current"]
    recovered = current["recovery"]["ratio"]
    if not (recovered is not None and recovered >= RECOVERY_MINIMUM
            and current["detector_vs_none"]["p"] < ALPHA and current["detector_vs_none"]["mean"] > 0
            and current["detector_vs_random"]["p"] < ALPHA and current["detector_vs_random"]["mean"] > 0):
        problems.append("recovery criteria not met")
    control_rate = report["pair_rates"]["control"]["rate"]
    if control_rate is None or control_rate > CONTROL_YES_LIMIT:
        problems.append("too many false supersessions on control turns")
    costed = report["costed"]
    if costed["detector_vs_none"]["mean"] < costed["oracle_vs_none"]["mean"] - HARM_MARGIN:
        problems.append("harms previous/both questions beyond the oracle")
    return "usable detector" if not problems else "not usable: " + "; ".join(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--suite", default=str(ROOT / "data/longmemeval/knowledge_update.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--out", default=str(ROOT / "experiments/results"))
    args = parser.parse_args()
    started = time.time()

    suite = [s for s in load_eval_suite(args.suite) if category_of(s) in CATEGORIES]
    if args.limit:
        suite = suite[: args.limit]
    safe_model = args.model.replace(":", "-").replace("/", "-")
    out = Path(args.out)
    judge = OllamaJudge(args.model, out / "update_detection_judgments_{}.jsonl".format(safe_model))
    print("{} questions: {}".format(len(suite), {c: sum(category_of(s) == c for s in suite) for c in CATEGORIES}))

    streams = []
    for index, eval_set in enumerate(suite, start=1):
        streams.append(judge_stream(eval_set, judge, category_of(eval_set)))
        print("  {}/{} {} judged={} new model calls={} ({:.0f}s)".format(
            index, len(suite), eval_set["id"], len(streams[-1]["judged"]), judge.new_calls,
            time.time() - started), flush=True)

    judged = [item for stream in streams for item in stream["judged"]]
    pairs = [(stream, old) for stream in streams for old in stream["oracle_olds"]]
    candidate_recall = sum(old in stream["candidates"] for stream, old in pairs) / len(pairs)
    pair_recall = sum(
        any(j["oracle"] and j["old"] == old and j["answer"] == "YES" for j in stream["judged"])
        for stream, old in pairs
    ) / len(pairs)

    def decisions(stream, kinds):
        return [(j["new"], j["old"]) for j in stream["judged"]
                if j["kind"] in kinds and j["answer"] == "YES"]

    def run(updates_for):
        sets = []
        for stream, eval_set in zip(streams, suite):
            modified = dict(eval_set, updates=updates_for(stream, eval_set))
            load_eval_set(modified)
            sets.append(modified)
        rows = query_rows(sets, k=DEPTH, links_for=lambda s: [], embedder_for=lambda s: None)
        return knowledge_update.per_query(rows)

    def position_of(eval_set):
        return {key: index for index, key in enumerate(eval_set["memories"])}

    configurations = {
        "none": run(lambda stream, s: []),
        "oracle": run(lambda stream, s: s["updates"]),
        "detector_evidence_only": run(
            lambda stream, s: valid_updates(decisions(stream, {"evidence"}), position_of(s))),
        "detector_all": run(
            lambda stream, s: valid_updates(decisions(stream, {"evidence", "control"}), position_of(s))),
    }
    configurations["random"] = knowledge_update.average_runs([
        run(lambda stream, s, seed=seed: random_updates(
            decisions(stream, {"evidence", "control"}), list(s["memories"]), "{}:{}:{}".format(seed, SEED, s["id"])))
        for seed in range(SEEDS)
    ])

    group = {c: [i for i, stream in enumerate(streams) if stream["category"] in c]
             for c in (("current",), ("previous", "both"))}

    def subset(name, indexes):
        return [configurations[name][i] for i in indexes]

    def change(first, second, indexes):
        return knowledge_update.paired_changes(
            knowledge_update.changes_between(subset(first, indexes), subset(second, indexes)))

    current, costed = group[("current",)], group[("previous", "both")]
    report = {
        "model": args.model,
        "questions": {"current": len(current), "previous_both": len(costed)},
        "candidate_recall": candidate_recall,
        "pair_recall_overall": pair_recall,
        "oracle_pairs": len(pairs),
        "pair_rates": rates(judged),
        "summary": {name: {"current": knowledge_update.summary(subset(name, current)),
                           "previous_both": knowledge_update.summary(subset(name, costed))}
                    for name in configurations},
        "current": {
            "oracle_vs_none": change("none", "oracle", current),
            "detector_vs_none": change("none", "detector_all", current),
            "detector_evidence_only_vs_none": change("none", "detector_evidence_only", current),
            "detector_vs_random": change("random", "detector_all", current),
            "recovery": recovery([e["rr"] for e in subset("none", current)],
                                 [e["rr"] for e in subset("oracle", current)],
                                 [e["rr"] for e in subset("detector_all", current)]),
        },
        "costed": {
            "oracle_vs_none": change("none", "oracle", costed),
            "detector_vs_none": change("none", "detector_all", costed),
        },
        "expected_false_supersessions_per_stream": (
            None if report_rate(judged, "control") is None
            else report_rate(judged, "control") * CONTROL_CANDIDATES * STREAM_LENGTH),
        "model_calls": {"new": judge.new_calls, "new_seconds": round(judge.new_seconds, 1),
                        "judged_pairs": len(judged)},
    }
    report["verdict"] = verdict(report)
    print_report(report)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "update_detection_{}.json".format(safe_model)
    path.write_text(json.dumps({"report": report, "streams": streams,
                                "per_question": configurations}, indent=1), encoding="utf-8")
    print("Saved {} ({:.0f}s)".format(path, time.time() - started))


def print_report(report):
    r = report["pair_rates"]
    fmt = lambda x: "n/a" if x["rate"] is None else "{:.1%} ({}/{}; 95% CI {:.1%} to {:.1%})".format(
        x["rate"], x["yes"], x["n"], *x["ci95"])
    print("\nmodel {}: {} model calls this run ({:.0f}s)".format(
        report["model"], report["model_calls"]["new"], report["model_calls"]["new_seconds"]))
    print("candidate recall: {:.0%} of {} oracle older turns are among the {} candidates".format(
        report["candidate_recall"], report["oracle_pairs"], CANDIDATES))
    print("oracle older turns found as a candidate and judged YES: {:.0%}".format(report["pair_recall_overall"]))
    print("YES rate, oracle pairs among the candidates: " + fmt(r["oracle_pairs"]))
    print("YES rate, other candidates of the evidence turn: " + fmt(r["evidence_non_oracle"]))
    print("YES rate, control-turn candidates: " + fmt(r["control"]))
    print("YES rate, random pairs (negative control): " + fmt(r["random"]))
    print("invalid answers: {}; oracle pairs with a clipped message: {}".format(
        r["invalid"], r["clipped_oracle_pairs"]))
    print("expected false supersessions per {}-turn stream (control rate x {} x {}): {}".format(
        STREAM_LENGTH, CONTROL_CANDIDATES, STREAM_LENGTH,
        "n/a" if report["expected_false_supersessions_per_stream"] is None
        else "{:.1f}".format(report["expected_false_supersessions_per_stream"])))
    print("\nreciprocal rank of the current answer (50 'current' questions) | previous+both (15)")
    for name, s in report["summary"].items():
        print("  {:<24} {:.3f} | {:.3f}   stale first {}".format(
            name, s["current"]["rr"], s["previous_both"]["rr"],
            "n/a" if s["current"]["stale_first"] is None else "{:.2f}".format(s["current"]["stale_first"])))
    def line(name, c):
        print("  {:<34} mean {:+.3f}  95% CI [{:+.3f}, {:+.3f}]  p={:.4f}  better {} / worse {}".format(
            name, c["mean"], *c["ci95"], c["p"], c["better"], c["worse"]))
    print("\nchanges in reciprocal rank (current questions):")
    for name in ("oracle_vs_none", "detector_vs_none", "detector_evidence_only_vs_none", "detector_vs_random"):
        line(name, report["current"][name])
    rec = report["current"]["recovery"]
    print("  share of the oracle gain recovered: {} (95% CI {})".format(
        "n/a" if rec["ratio"] is None else "{:.0%}".format(rec["ratio"]),
        "n/a" if rec["ci95"] is None else "{:.0%} to {:.0%}".format(*rec["ci95"])))
    print("\nchanges in reciprocal rank (previous + both questions):")
    for name in ("oracle_vs_none", "detector_vs_none"):
        line(name, report["costed"][name])
    print("\nVerdict: " + report["verdict"])


if __name__ == "__main__":
    main()
