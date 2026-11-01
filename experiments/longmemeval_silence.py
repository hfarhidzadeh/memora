"""Silence on LongMemEval: can a score gate tell when Memora has no answer?

Question. Search always returns the best matches, so when the answer was never
stored it hands back junk. Can a gate on the raw search signals stay silent for
questions the user never answered while still speaking for those it can answer?

Data. LongMemEval_S (cleaned), one store per question: its memories the user's
turns (240 to 250 per question on average), no links, no updates, nothing else.

- near-miss: the 30 abstention questions (ids ending ``_abs``). Each asks about
  something close to what the user did say ("name of my hamster" when they only
  mentioned a cat). Silence is correct.
- twin: for 29 of them the benchmark also holds the original, answerable
  question without the ``_abs`` suffix. Its chat history holds the same
  evidence turns as the near-miss's, with different filler sessions (no session
  is shared within a pair). Speaking is correct. A near-miss and its twin are a
  matched pair: same topic and question type.
- calibration: a random sample of 100 answerable questions of the same four
  types (40 multi-session, 20 each of single-session-user, temporal-reasoning,
  knowledge-update; seed 0), none of them a twin. It only sets thresholds.
  Without an embedder the rest of that pool is searched too, for a sensitivity
  check.

Gate statistics, each the best value over all memories for the question:
``lexical`` (raw BM25-style score), ``coverage`` (share of the question's
inverse-document-frequency mass found in one memory) and, with an embedder,
``vector`` (cosine similarity, ``min_similarity`` 0 so nothing is hidden). Search
runs with ``limit`` equal to the number of memories. The gate speaks when the
statistic reaches a threshold. The threshold keeps a chosen share of the
calibration questions (retention, 0.90 primary; 0.80 and 0.95 also reported)
and never sees a near-miss question.

Controls.
- positive (foreign): each twin's store is asked a different twin's question
  whose evidence sessions are not in that store. Silence is correct and easy. A
  statistic that cannot silence these has no power to detect absence.
- negative: the same random-word query (as many words as the twin's question,
  drawn from the calibration stores' vocabulary) goes to both stores of each
  pair. Any separation then comes from the stores, not the question.

Decision rule, fixed before running:
  1. A statistic passes its controls if AUC(twin vs foreign) >= 0.80 and its
     negative control shows no separation (paired sign test p >= 0.05 / 3).
     A statistic that fails is reported as inconclusive.
  2. A statistic separates near-miss questions if, over the matched pairs, the
     twin scores higher than its near-miss more often than the reverse, with a
     two-sided exact sign test p < 0.05 / 3 (three statistics; ties dropped).
  3. A statistic is a useful gate if it separates them and, at retention 0.90,
     silences at least half of the 30 near-miss questions.
  A statistic is called a useful gate only if it passes rules 1 to 3.

Version 1 (results in ``results/silence_v1_superseded``) used permuted vectors
(each memory embedded as a different memory, seeds 0-2) as the negative control
for the vector statistic. Each statistic is a maximum over all memories of a
store, and a permutation only reassigns the store's own vectors among its own
memories, so the maximum cannot change: the control reproduced the real
statistic exactly (22 of 29 pairs, p = 0.008, in all three seeds) and the rule's
verdict, "inconclusive (a control failed)", was an artifact of the design, not
a sign of a problem with the statistic. Version 2 applies the random-word-query
control to all three statistics. Nothing else changed: no threshold or
criterion of the rule was touched, and the lexical and coverage numbers are
identical to version 1.

Run from the project root:

    python3 -I experiments/longmemeval_silence.py
    python3 -I experiments/longmemeval_silence.py --embedder sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore, SentenceTransformerEmbedder  # noqa: E402
from dynmemory.evaluation import CachedEmbedder  # noqa: E402
from dynmemory.longmemeval import question_memories  # noqa: E402

DATA = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
TYPES = ("multi-session", "single-session-user", "temporal-reasoning", "knowledge-update")
CALIBRATION_PER_TYPE = {"multi-session": 40, "single-session-user": 20,
                        "temporal-reasoning": 20, "knowledge-update": 20}
RETENTIONS = (0.80, 0.90, 0.95)
PRIMARY_RETENTION = 0.90
STATISTICS = ("lexical", "coverage", "vector")
ALPHA = 0.05 / 3
CONTROL_AUC = 0.80
USEFUL_SILENCED = 0.5
HIT_DEPTH = 5
BOOTSTRAP_SAMPLES = 10000
TOKEN = re.compile(r"\w+", re.UNICODE)


# --- statistics (tested in tests/test_experiment_statistics.py) -------------------


def threshold_for(values, retention):
    """The largest threshold at which at least ``retention`` of the values still pass."""
    ordered = sorted(values)
    # Rounded first: (1 - 0.9) * 100 is 9.999999999999998, which would floor to 9.
    return ordered[int(math.floor(round((1.0 - retention) * len(ordered), 9)))]


def auc(positives, negatives):
    """Chance that a random positive scores above a random negative (ties count half)."""
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def sign_test(first, second):
    """Pairs where ``first`` is higher, where ``second`` is, ties, and the two-sided exact p."""
    higher = sum(a > b for a, b in zip(first, second))
    lower = sum(a < b for a, b in zip(first, second))
    ties = len(first) - higher - lower
    n = higher + lower
    if n == 0:
        return higher, lower, ties, 1.0
    tail = sum(math.comb(n, k) for k in range(max(higher, lower), n + 1)) / 2 ** n
    return higher, lower, ties, min(1.0, 2 * tail)


def paired_auc(first, second):
    """Share of pairs where ``first`` scores higher (ties count half)."""
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in zip(first, second)) / len(first)


def paired_auc_interval(first, second, seed=0):
    rng = random.Random(seed)
    pairs = list(zip(first, second))
    draws = sorted(
        paired_auc(*zip(*[rng.choice(pairs) for _ in pairs])) for _ in range(BOOTSTRAP_SAMPLES)
    )
    return draws[int(0.025 * BOOTSTRAP_SAMPLES)], draws[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def gate_values(results):
    """The three gate statistics: the best value over all of a search's results."""
    return {
        "lexical": max((r.lexical_score for r in results), default=0.0),
        "coverage": max((r.coverage for r in results), default=0.0),
        "vector": max((r.vector_score for r in results), default=0.0),
    }


def evidence_hit(results, key_of, evidence, signal):
    """Is an evidence turn among the top ``HIT_DEPTH`` by one raw signal? None without evidence."""
    if not evidence:
        return None
    ranked = sorted(
        (r for r in results if getattr(r, signal) > 0), key=lambda r: -getattr(r, signal)
    )[:HIT_DEPTH]
    return any(key_of[r.memory.id] in evidence for r in ranked)


# --- design ----------------------------------------------------------------------


def calibration_sample(questions, excluded_ids, seed=0):
    """Type-matched random sample of answerable questions, plus the pool it came from."""
    pool = {name: [] for name in TYPES}
    for question in questions:
        if (
            question["question_type"] in pool
            and not question["question_id"].endswith("_abs")
            and question["question_id"] not in excluded_ids
            and question_memories(question)[2]
        ):
            pool[question["question_type"]].append(question)
    rng = random.Random(seed)
    sample = []
    for name in TYPES:
        ordered = sorted(pool[name], key=lambda q: q["question_id"])
        sample += rng.sample(ordered, CALIBRATION_PER_TYPE[name])
    return sample, [q for name in TYPES for q in pool[name]]


def foreign_queries(twins, seed=0):
    """For each twin, another twin's question whose evidence is not in the twin's store."""
    rng = random.Random(seed)
    queries = {}
    for question in twins:
        haystack = set(question["haystack_session_ids"])
        candidates = [
            other for other in twins
            if other["question_id"] != question["question_id"]
            and not haystack & set(other["answer_session_ids"])
        ]
        queries[question["question_id"]] = rng.choice(candidates)["question"]
    return queries


def random_word_queries(twins, vocabulary, seed=0):
    rng = random.Random(seed)
    return {
        q["question_id"]: " ".join(
            rng.sample(vocabulary, len(TOKEN.findall(q["question"].lower())))
        )
        for q in twins
    }


def build_store(question, embedder):
    memories, _, evidence = question_memories(question)
    store = MemoryStore(":memory:")
    ids = {key: store.add_memory(m["content"], m["kind"]).id for key, m in memories.items()}
    store.embedder = embedder  # attached last, so the first search embeds every memory at once
    return store, {memory_id: key for key, memory_id in ids.items()}, {key for _, key in evidence}


def measure(question, embedder, queries):
    """Gate statistics (and evidence hits) of each named query against the question's store."""
    store, key_of, evidence = build_store(question, embedder)
    measured = {}
    for name, text in queries.items():
        results = store.search(text, limit=len(key_of), record_access=False)
        measured[name] = {
            "values": gate_values(results),
            "hit": {
                signal: evidence_hit(results, key_of, evidence, signal + "_score")
                for signal in ("lexical", "vector")
            },
        }
    store.close()
    return measured


# --- analysis --------------------------------------------------------------------


def analyse(statistic, groups, calibration_ids):
    """All numbers for one gate statistic. ``groups`` maps group -> {question id: value}."""
    value = lambda group, ids=None: [groups[group][i] for i in (ids or groups[group])]
    pair_ids = sorted(i for i in groups["twin"] if i + "_abs" in groups["near_miss"])
    twin = [groups["twin"][i] for i in pair_ids]
    near_miss = [groups["near_miss"][i + "_abs"] for i in pair_ids]
    foreign = [groups["foreign"][i] for i in pair_ids]
    higher, lower, ties, p = sign_test(twin, near_miss)
    negative = groups["negative"]
    out = {
        "pairs": len(pair_ids),
        "twin_higher": higher, "near_miss_higher": lower, "ties": ties, "sign_p": p,
        "paired_auc": paired_auc(twin, near_miss),
        "paired_auc_ci95": paired_auc_interval(twin, near_miss),
        "foreign_auc": auc(twin, foreign),
        "median": {
            "calibration": sorted(value("calibration", calibration_ids))[len(calibration_ids) // 2],
            "twin": sorted(twin)[len(twin) // 2],
            "near_miss": sorted(near_miss)[len(near_miss) // 2],
            "foreign": sorted(foreign)[len(foreign) // 2],
        },
        "negative_control": [
            dict(zip(("first_higher", "second_higher", "ties", "p"), sign_test(first, second)))
            for first, second in negative
        ],
        "gates": {},
    }
    for retention in RETENTIONS:
        threshold = threshold_for(value("calibration", calibration_ids), retention)
        speaks = lambda x: x >= threshold
        kept = sum(speaks(x) for x in twin)
        silenced_near = sum(not speaks(x) for x in groups["near_miss"].values())
        silenced_foreign = sum(not speaks(x) for x in foreign)
        out["gates"][str(retention)] = {
            "threshold": threshold,
            "calibration_kept": sum(speaks(x) for x in value("calibration", calibration_ids))
            / len(calibration_ids),
            "twin_kept": kept / len(twin),
            "near_miss_silenced": silenced_near / len(groups["near_miss"]),
            "near_miss_silenced_ci95": wilson(silenced_near, len(groups["near_miss"])),
            "foreign_silenced": silenced_foreign / len(foreign),
        }
    return out


def verdict(analysis):
    """Apply the decision rule to one statistic's analysis."""
    controls = analysis["foreign_auc"] >= CONTROL_AUC and all(
        run["p"] >= ALPHA for run in analysis["negative_control"]
    )
    if not controls:
        return "inconclusive (a control failed)"
    separates = analysis["sign_p"] < ALPHA and analysis["twin_higher"] > analysis["near_miss_higher"]
    if not separates:
        return "no signal for near-miss questions"
    if analysis["gates"][str(PRIMARY_RETENTION)]["near_miss_silenced"] >= USEFUL_SILENCED:
        return "useful gate"
    return "separates near-miss questions but silences too few"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=("none", "sentence-transformers"), default="none")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    started = time.time()

    questions = json.loads(DATA.read_text(encoding="utf-8"))
    by_id = {q["question_id"]: q for q in questions}
    near_miss = [q for q in questions if q["question_id"].endswith("_abs")]
    twins = [by_id[q["question_id"][: -len("_abs")]] for q in near_miss
             if q["question_id"][: -len("_abs")] in by_id]
    sample, pool = calibration_sample(questions, {q["question_id"] for q in twins})
    vector = args.embedder == "sentence-transformers"
    searched = sample if vector else pool  # MiniLM on the whole pool would take hours
    statistics = [s for s in STATISTICS if s != "vector" or vector]
    print("near-miss {} | twins {} | calibration {} | searched pool {}".format(
        len(near_miss), len(twins), len(sample), len(searched)))

    embedder = CachedEmbedder(SentenceTransformerEmbedder(min_similarity=0.0)) if vector else None
    vocabulary = sorted({t for q in sample for m in question_memories(q)[0].values()
                         for t in TOKEN.findall(m["content"].lower())})
    foreign = foreign_queries(twins)
    random_queries = random_word_queries(twins, vocabulary)

    rows = []
    group_of = {q["question_id"]: "calibration" for q in searched}
    group_of.update({q["question_id"]: "twin" for q in twins})
    group_of.update({q["question_id"]: "near_miss" for q in near_miss})

    def record(group, question, measured, name):
        rows.append({"group": group, "id": question["question_id"], "type": question["question_type"],
                     "query": name, **measured[name]})

    for index, question in enumerate(near_miss + twins + searched, start=1):
        question_id = question["question_id"]
        group = group_of[question_id]
        queries = {"natural": question["question"]}
        if group == "twin":
            queries["foreign"] = foreign[question_id]
            queries["random"] = random_queries[question_id]
        elif group == "near_miss" and question_id[: -len("_abs")] in random_queries:
            queries["random"] = random_queries[question_id[: -len("_abs")]]
        measured = measure(question, embedder, queries)
        record(group, question, measured, "natural")
        if "foreign" in measured:
            record("foreign", question, measured, "foreign")
        if "random" in measured:
            record("random_" + group, question, measured, "random")
        if index % 25 == 0:
            print("  {}/{} stores, {:.0f}s".format(index, len(near_miss) + len(twins) + len(searched),
                                                   time.time() - started), flush=True)

    sample_ids = {q["question_id"] for q in sample}
    report = {"embedder": args.embedder, "statistics": {}, "verdict": {}, "sensitivity": {}}
    for statistic in statistics:
        def values(group):
            return {r["id"]: r["values"][statistic] for r in rows if r["group"] == group}

        groups = {"near_miss": values("near_miss"), "twin": values("twin"),
                  "foreign": values("foreign"), "calibration": values("calibration")}
        pair_ids = sorted(i for i in groups["twin"] if i + "_abs" in groups["near_miss"])
        groups["negative"] = [([values("random_twin")[i] for i in pair_ids],
                               [values("random_near_miss")[i + "_abs"] for i in pair_ids])]
        analysis = analyse(statistic, groups, sorted(sample_ids))
        report["statistics"][statistic] = analysis
        report["verdict"][statistic] = verdict(analysis)
        if not vector:
            everything = sorted(groups["calibration"])
            report["sensitivity"][statistic] = analyse(statistic, groups, everything)["gates"]

    # Does silencing remove retrievals that were working? (answerable questions only)
    answerable = [r for r in rows if r["group"] in ("twin", "calibration") and r["query"] == "natural"]
    report["hits"] = {}
    for signal in ("lexical", "vector") if vector else ("lexical",):
        stat = "lexical" if signal == "lexical" else "vector"
        threshold = report["statistics"][stat]["gates"][str(PRIMARY_RETENTION)]["threshold"]
        scored = [r for r in answerable if r["hit"][signal] is not None]
        kept = [r["hit"][signal] for r in scored if r["values"][stat] >= threshold]
        dropped = [r["hit"][signal] for r in scored if r["values"][stat] < threshold]
        report["hits"][signal] = {
            "answerable": len(scored),
            "hit_at_5_overall": sum(r["hit"][signal] for r in scored) / len(scored),
            "hit_at_5_when_kept": sum(kept) / len(kept) if kept else None,
            "hit_at_5_when_silenced": sum(dropped) / len(dropped) if dropped else None,
            "silenced": len(dropped),
        }

    print_report(report, statistics)
    out = Path(args.out) if args.out else ROOT / "experiments" / "results" / (
        "longmemeval_silence{}.json".format("_sentence-transformers" if vector else ""))
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=1), encoding="utf-8")
    print("Saved {} ({:.0f}s)".format(out, time.time() - started))


def print_report(report, statistics):
    for statistic in statistics:
        a = report["statistics"][statistic]
        print("\n== {} ==".format(statistic))
        print("  median value: " + ", ".join("{} {:.3f}".format(k, v) for k, v in a["median"].items()))
        print("  matched pairs {}: twin higher {}, near-miss higher {}, ties {}; sign p={:.4f}; "
              "paired AUC {:.2f} (95% CI {:.2f} to {:.2f})".format(
                  a["pairs"], a["twin_higher"], a["near_miss_higher"], a["ties"], a["sign_p"],
                  a["paired_auc"], *a["paired_auc_ci95"]))
        print("  positive control: AUC twin vs foreign {:.2f} (need >= {})".format(a["foreign_auc"], CONTROL_AUC))
        for run in a["negative_control"]:
            print("  negative control: first higher {}, second higher {}, ties {}, p={:.3f}".format(
                run["first_higher"], run["second_higher"], run["ties"], run["p"]))
        print("  retention  threshold  calib.kept  twins.kept  near-miss silenced (95% CI)  foreign silenced")
        for retention, g in a["gates"].items():
            print("  {:>9}  {:>9.3f}  {:>10.2f}  {:>10.2f}  {:>9.2f} ({:.2f} to {:.2f})  {:>16.2f}".format(
                retention, g["threshold"], g["calibration_kept"], g["twin_kept"], g["near_miss_silenced"],
                *g["near_miss_silenced_ci95"], g["foreign_silenced"]))
        print("  verdict: {}".format(report["verdict"][statistic]))
    print("\nHit@5 of answerable questions at retention {}:".format(PRIMARY_RETENTION))
    for signal, h in report["hits"].items():
        print("  {}: overall {:.2f}; kept {}; silenced {} ({} questions)".format(
            signal, h["hit_at_5_overall"],
            "n/a" if h["hit_at_5_when_kept"] is None else "{:.2f}".format(h["hit_at_5_when_kept"]),
            "n/a" if h["hit_at_5_when_silenced"] is None else "{:.2f}".format(h["hit_at_5_when_silenced"]),
            h["silenced"]))


if __name__ == "__main__":
    main()
