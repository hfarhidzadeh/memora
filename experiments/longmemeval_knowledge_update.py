"""Knowledge updates on LongMemEval: should links carry over to a fact's new version?

Question. When a newer version of a fact replaces the old one, should the old
version's links move to the new version? Three options:

- keep: links stay on the old, archived version (Memora's default);
- copy: links are copied to the new version when the update happens;
- topic: links stay put, and search treats a link to any version of a fact as
  a link to its current version.

Data. LongMemEval_S (cleaned) knowledge-update questions, converted with
``dynmemory.longmemeval`` and the labels in
``eval/longmemeval_knowledge_update_labels.json``: one store per question,
its memories the user's turns (about 240 per question), its links each user
turn ``related_to`` the next one in the same session (structure only, no
labels). The primary analysis uses the 50 questions labelled "current": the
question asks for the latest value of something stated twice. Updates (newer
replaces older) come from the benchmark's evidence labels, so they are an
oracle: perfect update detection.

Measure. Reciprocal rank of the current evidence turn (MRR over the top 10),
recall@1/3/5/10, and "stale first": the share of questions where an outdated
version ranks above the current one (only meaningful when updates are not
applied, because applied updates archive the outdated versions).

Controls.
- negative: the same links moved to random memories the source is not linked
  to, averaged over seeds. Carrying random links over should not help.
- positive: no session links at all, only one planted link from a non-answer
  memory to the outdated version. The anchor is the best non-answer match
  ranked below the current answer (with updates applied), so the link can lift
  the answer past other memories. Under keep the link points at an archived
  memory; under topic it reaches the current version. Questions without such an
  anchor in the top 50 are left out of this control and counted.

Statistics. For each comparison, the per-question change in reciprocal rank,
its mean, a 95% paired bootstrap interval, and a two-sided sign-flip
permutation test (exact up to 16 non-zero changes, otherwise 100,000 random
sign flips).

Decision rule, version 2, fixed before running this version:
  1. The positive control (topic vs keep, planted links only) must improve
     reciprocal rank with p < 0.05; otherwise the result is inconclusive.
  2. Topic "shows a benefit" only if topic vs keep with session links improves
     reciprocal rank with p < 0.05, and the improvement beyond random links
     (per question: that change minus the seed-averaged change with wrong
     links) is also above 0 with p < 0.05.
  3. Otherwise: no benefit. Automatic link extraction (the next step) is only
     tried if topic shows a benefit.

Version 1 of this experiment (results in ``results/v1_superseded``) was
replaced after an independent review found that (a) the "newest evidence is
current" rule mislabelled 15 of the 70 questions with an update and one had
its session dates reversed, (b) both positive controls added the planted link
on top of the session links, so they contained the effect being tested, and
(c) the decision rule compared point estimates on a result carried by six
questions.

Run from the project root:

    python3 -I experiments/longmemeval_knowledge_update.py
    python3 -I experiments/longmemeval_knowledge_update.py --embedder sentence-transformers
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import NGramHashEmbedder, SentenceTransformerEmbedder  # noqa: E402
from dynmemory.evaluation import (  # noqa: E402
    CachedEmbedder,
    PermutedEmbedder,
    load_eval_suite,
    query_rows,
    wrong_links,
)

DEPTH = 10
ANCHOR_DEPTH = 50
RECALL_CUTOFFS = (1, 3, 5, 10)
BOOTSTRAP_SAMPLES = 10000
EXACT_FLIP_LIMIT = 16
RANDOM_FLIPS = 100000


def per_query(rows):
    metrics = []
    for row in rows:
        relevant = set(row["relevant"])
        entry = {"set": row["set"], "rr": row["reciprocal_rank"], "stale_first": row["stale_first"]}
        for cutoff in RECALL_CUTOFFS:
            entry["recall@{}".format(cutoff)] = len(
                relevant.intersection(row["retrieved"][:cutoff])
            ) / len(relevant)
        metrics.append(entry)
    return metrics


def average_runs(runs):
    averaged = []
    for entries in zip(*runs):
        merged = dict(entries[0])
        for field, value in merged.items():
            if field != "set" and value is not None:
                merged[field] = sum(entry[field] for entry in entries) / len(entries)
        averaged.append(merged)
    return averaged


def summary(metrics):
    result = {"questions": len(metrics)}
    for field in ["rr"] + ["recall@{}".format(cutoff) for cutoff in RECALL_CUTOFFS]:
        result[field] = sum(entry[field] for entry in metrics) / len(metrics)
    stale = [entry["stale_first"] for entry in metrics if entry["stale_first"] is not None]
    result["stale_first"] = sum(stale) / len(stale) if stale else None
    return result


def sign_flip_p(changes, seed=0):
    """Two-sided p-value for a mean change of 0, by flipping the signs of changes."""
    nonzero = [change for change in changes if abs(change) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero)) - 1e-12
    if len(nonzero) <= EXACT_FLIP_LIMIT:
        hits = sum(
            abs(sum(sign * change for sign, change in zip(signs, nonzero))) >= observed
            for signs in itertools.product((1, -1), repeat=len(nonzero))
        )
        return hits / 2 ** len(nonzero)
    rng = random.Random(seed)
    hits = sum(
        abs(sum(change if rng.random() < 0.5 else -change for change in nonzero)) >= observed
        for _ in range(RANDOM_FLIPS)
    )
    return (hits + 1) / (RANDOM_FLIPS + 1)


def paired_changes(changes, seed=0):
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(changes) for _ in changes) / len(changes)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "questions": len(changes),
        "mean": sum(changes) / len(changes),
        "ci95": [means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]],
        "p": sign_flip_p(changes, seed),
        "better": sum(change > 1e-12 for change in changes),
        "worse": sum(change < -1e-12 for change in changes),
        "same": sum(abs(change) <= 1e-12 for change in changes),
    }


def changes_between(before, after, field="rr"):
    """Per-question change after minus before; both lists cover the same questions."""
    if [entry["set"] for entry in before] != [entry["set"] for entry in after]:
        raise ValueError("compared runs cover different questions")
    return [b[field] - a[field] for a, b in zip(before, after)]


def planted_links(eval_sets):
    """Positive control: one link from a non-answer memory ranked below the answer
    to each outdated version. Returns the links and the questions left out."""
    rows = query_rows(
        eval_sets,
        k=ANCHOR_DEPTH,
        links_for=lambda eval_set: [],
        embedder_for=lambda eval_set: None,
    )
    planted, left_out = {}, []
    for eval_set, row in zip(eval_sets, rows):
        query = eval_set["queries"][0]
        relevant, stale = set(query["relevant"]), query.get("stale", [])
        position = next(
            (index for index, key in enumerate(row["retrieved"]) if key in relevant), None
        )
        anchor = None
        if position is not None:
            anchor = next(
                (
                    key
                    for key in row["retrieved"][position + 1:]
                    if key not in relevant and key not in stale
                ),
                None,
            )
        if anchor is None or not stale:
            left_out.append(eval_set["id"])
            planted[eval_set["id"]] = []
        else:
            planted[eval_set["id"]] = [
                {"source": anchor, "target": key, "relation": "related_to"} for key in stale
            ]
    return planted, left_out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", default=str(ROOT / "data/longmemeval/knowledge_update.jsonl"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--embedder", choices=["none", "ngram", "sentence-transformers"], default="none")
    parser.add_argument("--out", default=str(ROOT / "experiments/results"))
    args = parser.parse_args()

    suite = load_eval_suite(args.suite)

    def category(eval_set):
        tags = eval_set["queries"][0]["tags"]
        return tags[1] if len(tags) > 1 else "unlabelled"

    current = [eval_set for eval_set in suite if category(eval_set) == "current"]
    costed = [eval_set for eval_set in suite if category(eval_set) in ("previous", "both")]
    planted, left_out = planted_links(current)
    embedder = None
    if args.embedder == "ngram":
        embedder = CachedEmbedder(NGramHashEmbedder())
    elif args.embedder == "sentence-transformers":
        embedder = CachedEmbedder(SentenceTransformerEmbedder())

    def run(name, eval_sets, *, links, policy="keep", updates=True, vectors=None):
        started = time.time()
        rows = query_rows(
            eval_sets,
            k=DEPTH,
            links_for=links,
            embedder_for=vectors or (lambda eval_set: None),
            link_carryover=policy,
            apply_updates=updates,
        )
        print("  {:<54} {:5.1f}s".format(name, time.time() - started), flush=True)
        return per_query(rows)

    def run_seeds(name, eval_sets, *, policy, vectors=None):
        return average_runs(
            [
                run(
                    "{} (seed {})".format(name, seed),
                    eval_sets,
                    links=lambda eval_set, seed=seed: wrong_links(eval_set, seed),
                    policy=policy,
                    vectors=vectors,
                )
                for seed in range(args.seeds)
            ]
        )

    def no_links(eval_set):
        return []

    def session_links(eval_set):
        return eval_set["links"]

    def planted_only(eval_set):
        return planted[eval_set["id"]]

    results = {}
    print("lexical retrieval, {} 'current' questions:".format(len(current)), flush=True)
    results["lexical, updates not applied"] = run("updates not applied", current, links=no_links, updates=False)
    results["lexical"] = run("oracle updates", current, links=no_links)
    for policy in ("keep", "copy", "topic"):
        results["lexical + links ({})".format(policy)] = run(
            "oracle updates + session links, " + policy, current, links=session_links, policy=policy
        )
    for policy in ("keep", "topic"):
        results["lexical + wrong links ({}) [negative control]".format(policy)] = run_seeds(
            "wrong links, " + policy, current, policy=policy
        )
        results["lexical + planted link only ({}) [positive control]".format(policy)] = run(
            "planted link only, " + policy, current, links=planted_only, policy=policy
        )
    print("lexical retrieval, {} 'previous'/'both' questions:".format(len(costed)), flush=True)
    results["previous/both: lexical, updates not applied"] = run(
        "updates not applied", costed, links=no_links, updates=False
    )
    results["previous/both: lexical"] = run("oracle updates", costed, links=no_links)

    if embedder is not None:
        def vectors(eval_set):
            return embedder

        print("hybrid retrieval ({}), 'current' questions:".format(args.embedder), flush=True)
        results["hybrid, updates not applied"] = run(
            "updates not applied", current, links=no_links, updates=False, vectors=vectors
        )
        results["hybrid"] = run("oracle updates", current, links=no_links, vectors=vectors)
        results["hybrid, permuted vectors [negative control]"] = average_runs(
            [
                run(
                    "permuted vectors (seed {})".format(seed),
                    current,
                    links=no_links,
                    vectors=lambda eval_set, seed=seed: PermutedEmbedder(
                        embedder,
                        [memory["content"].strip() for memory in eval_set["memories"].values()],
                        seed,
                    ),
                )
                for seed in range(args.seeds)
            ]
        )
        for policy in ("keep", "topic"):
            results["hybrid + links ({})".format(policy)] = run(
                "oracle updates + session links, " + policy,
                current,
                links=session_links,
                policy=policy,
                vectors=vectors,
            )
            results["hybrid + wrong links ({}) [negative control]".format(policy)] = run_seeds(
                "wrong links, " + policy, current, policy=policy, vectors=vectors
            )

    def subset(metrics, ids):
        return [entry for entry in metrics if entry["set"] in ids]

    anchored = {eval_set["id"] for eval_set in current} - set(left_out)
    contrasts = {
        "update handling: oracle updates vs none": changes_between(
            results["lexical, updates not applied"], results["lexical"]
        ),
        "session links (keep) vs no links": changes_between(
            results["lexical"], results["lexical + links (keep)"]
        ),
        "PRIMARY carry-over: topic vs keep": changes_between(
            results["lexical + links (keep)"], results["lexical + links (topic)"]
        ),
        "copy vs topic (expected identical)": changes_between(
            results["lexical + links (topic)"], results["lexical + links (copy)"]
        ),
        "negative control: topic vs keep, wrong links": changes_between(
            results["lexical + wrong links (keep) [negative control]"],
            results["lexical + wrong links (topic) [negative control]"],
        ),
        "positive control: topic vs keep, planted link only": changes_between(
            subset(results["lexical + planted link only (keep) [positive control]"], anchored),
            subset(results["lexical + planted link only (topic) [positive control]"], anchored),
        ),
        "cost on previous/both questions: updates vs none": changes_between(
            results["previous/both: lexical, updates not applied"], results["previous/both: lexical"]
        ),
    }
    contrasts["PRIMARY beyond random links"] = [
        real - random_links
        for real, random_links in zip(
            contrasts["PRIMARY carry-over: topic vs keep"],
            contrasts["negative control: topic vs keep, wrong links"],
        )
    ]
    if embedder is not None:
        contrasts["hybrid vs lexical (oracle updates, no links)"] = changes_between(
            results["lexical"], results["hybrid"]
        )
        contrasts["hybrid: vectors vs permuted vectors"] = changes_between(
            results["hybrid, permuted vectors [negative control]"], results["hybrid"]
        )
        contrasts["hybrid: session links (keep) vs no links"] = changes_between(
            results["hybrid"], results["hybrid + links (keep)"]
        )
        contrasts["hybrid carry-over: topic vs keep"] = changes_between(
            results["hybrid + links (keep)"], results["hybrid + links (topic)"]
        )
        contrasts["hybrid negative control: topic vs keep, wrong links"] = changes_between(
            results["hybrid + wrong links (keep) [negative control]"],
            results["hybrid + wrong links (topic) [negative control]"],
        )
        contrasts["hybrid carry-over beyond random links"] = [
            real - random_links
            for real, random_links in zip(
                contrasts["hybrid carry-over: topic vs keep"],
                contrasts["hybrid negative control: topic vs keep, wrong links"],
            )
        ]
    statistics = {name: paired_changes(changes) for name, changes in contrasts.items()}

    positive = statistics["positive control: topic vs keep, planted link only"]
    primary = statistics["PRIMARY carry-over: topic vs keep"]
    beyond = statistics["PRIMARY beyond random links"]
    if not (positive["mean"] > 0 and positive["p"] < 0.05):
        verdict = "inconclusive: the positive control did not show a significant improvement"
    elif primary["mean"] > 0 and primary["p"] < 0.05 and beyond["mean"] > 0 and beyond["p"] < 0.05:
        verdict = "benefit: topic beats keep beyond random links; try automatic link extraction next"
    else:
        verdict = "no benefit: topic does not beat keep beyond random links under the decision rule"

    suite_path = Path(args.suite).resolve()
    report = {
        "suite": str(
            suite_path.relative_to(ROOT) if ROOT in suite_path.parents else suite_path
        ),
        "questions_current": len(current),
        "questions_previous_or_both": len(costed),
        "positive_control_left_out": left_out,
        "seeds": args.seeds,
        "embedder": args.embedder,
        "summaries": {name: summary(metrics) for name, metrics in results.items()},
        "contrasts": statistics,
        "verdict": verdict,
        "per_question": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.embedder == "none" else "_" + args.embedder
    path = out / "longmemeval_knowledge_update{}.json".format(suffix)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummaries (each row: its own questions; see the name):")
    print("{:<58} {:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}".format(
        "configuration", "n", "MRR", "R@1", "R@3", "R@5", "R@10", "stale1st"))
    for name, metrics in report["summaries"].items():
        print(
            "{:<58} {:>4} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:>8}".format(
                name,
                metrics["questions"],
                metrics["rr"],
                metrics["recall@1"],
                metrics["recall@3"],
                metrics["recall@5"],
                metrics["recall@10"],
                "-" if metrics["stale_first"] is None else "{:.3f}".format(metrics["stale_first"]),
            )
        )
    print("\nPaired change in reciprocal rank (after minus before):")
    for name, result in statistics.items():
        print(
            "  {:<52} n={:<3} {:+.3f}  95% CI [{:+.3f}, {:+.3f}]  p={:.4f}  better {:>2} worse {:>2} same {:>2}".format(
                name, result["questions"], result["mean"], result["ci95"][0], result["ci95"][1],
                result["p"], result["better"], result["worse"], result["same"],
            )
        )
    print("\nPositive control left out {} question(s): {}".format(len(left_out), ", ".join(left_out) or "none"))
    print("Verdict:", verdict)
    print("Saved", path)


if __name__ == "__main__":
    main()
