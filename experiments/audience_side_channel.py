"""Side channel in search scores: can a viewer detect words in memories they cannot see?

Question (H2 in docs/multi_user_memory_review.md). Suppose per-user visibility
were added to Memora by dropping hidden memories from the results after
ranking. Search computes document frequencies over every active memory, so a
viewer's own scores depend on hidden content. Can a viewer use that to learn
whether a chosen word appears in a channel they are not in? Does filtering
before ranking close the channel?

Data. Scripted and synthetic, no LLM, no real text. Each trial builds a
workspace of pseudo-word memories (12 words each from a 500-word vocabulary):
200 in channels the viewer is in ("visible") and 50 in a channel the viewer is
not in ("hidden"). Each memory carries its channel, and the viewer is a member
of the visible channel only.

The probe. The viewer picks an anchor word they can see in one of their own
memories, a target word ``w`` they are curious about, and a nonsense word ``z``
that cannot occur anywhere. They search "anchor w" and "anchor z" and read the
best ``coverage`` among the results they are shown. Coverage is the share of
the query's inverse-document-frequency mass a memory contains. If ``w`` occurs
in no memory at all, ``w`` and ``z`` have the same document frequency (zero)
and the two coverages are equal. If ``w`` occurs in any memory, its mass drops
and the coverage of an anchor-only match rises. The statistic is

    D = coverage("anchor w") - coverage("anchor z")

and needs no calibration and no knowledge of the store's size.

Worlds, per trial (the visible memories are the same in all of them except
``shared``):
- none: no hidden memories.
- absent: 50 hidden memories, none containing ``w``.
- present-k: 50 hidden memories, k of them containing ``w`` (k = 1, 5, 20).
- shared: as ``absent``, but ``w`` is also in 5 of the viewer's own visible
  memories.

Arms.
- after: one store holds everything; hidden memories are dropped from the
  results after ranking.
- before: the store the viewer searches holds only their visible memories.
  This is what filtering before ranking computes. It cannot see hidden
  memories by construction, so its result in the ``present`` worlds is a
  sanity check, not a finding.
- built: the same store as the after arm, searched with
  ``search(..., viewers=viewer)``. This is the fix as built in the store, and
  the hidden memories are in the database it reads.

Controls.
- positive (shared): ``w`` is somewhere the viewer may legitimately see it.
  The probe must fire in both arms. If it does not, D = 0 elsewhere means
  nothing.
- negative (absent, none): ``w`` is nowhere. D must be zero in both arms.

The review's first wording of the negative control was "hidden memories
without the query words change nothing in either arm". Reading the scoring
code before any run showed that this is false for raw scores: any hidden
memory changes the memory count and average length, so every score shifts. The
control was therefore restated on D, which cancels those shifts. The shift in
raw scores is reported separately as a second, coarser leak (the viewer can
tell that hidden memories exist).

Decision rule, fixed before running (|D| <= 1e-12 counts as zero):
  1. The probe works if D != 0 in at least 95% of ``shared`` trials in each arm.
  2. The negative control holds if D = 0 in every ``absent`` and ``none`` trial
     in each arm.
  3. H2 is supported if, for each k, D > 0 in at least 95% of ``present-k``
     trials in the after arm and D = 0 in every such trial in the before arm.
  If rule 1 or 2 fails the result is reported as inconclusive.

Version 2 added the built arm once the store gained ``viewers``. Version 1 had
the after and before arms only, and its numbers for them are reproduced
exactly. Rules 1 to 3 apply to the built arm as they do to the before arm, and
one rule was added, also before the run:
  4. The fix works as built if, in every trial and every world, the built arm
     returns exactly the same probe values as the before arm.

Run from the project root:

    python3 -I experiments/audience_side_channel.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore, MemoryType  # noqa: E402

VOCABULARY_SIZE = 500
WORDS_PER_MEMORY = 12
VISIBLE = 200
HIDDEN = 50
PRESENT_COUNTS = (1, 5, 20)
SHARED_COUNT = 5
TARGET = "qtargetword"
NONSENSE = "qnonsenseword"
ZERO = 1e-12
HIDDEN_CHANNEL = "hidden"
VISIBLE_CHANNEL = "visible"
VIEWER = "viewer"
ARMS = ("after", "before", "built")
ARM_LABELS = {
    "after": "filter after ranking",
    "before": "filter before ranking (emulated)",
    "built": "search with viewers (built)",
}


def _texts(rng: random.Random, vocabulary: list, count: int) -> list:
    return [
        " ".join(rng.choice(vocabulary) for _ in range(WORDS_PER_MEMORY))
        for _ in range(count)
    ]


def _with_target(rng: random.Random, texts: list, count: int) -> list:
    """Replace one word with the target in ``count`` of the texts (length kept)."""
    chosen = set(rng.sample(range(len(texts)), count))
    changed = []
    for index, text in enumerate(texts):
        if index in chosen:
            words = text.split()
            words[rng.randrange(len(words))] = TARGET
            text = " ".join(words)
        changed.append(text)
    return changed


def _store(visible: list, hidden: list) -> MemoryStore:
    store = MemoryStore(":memory:")
    store.join_channel(VISIBLE_CHANNEL, VIEWER)
    store.join_channel(HIDDEN_CHANNEL, "insider")
    for channel, texts in ((VISIBLE_CHANNEL, visible), (HIDDEN_CHANNEL, hidden)):
        for text in texts:
            store.add_memory(text, MemoryType.FACT, channel=channel)
    return store


def _probe(search, anchor: str) -> dict:
    """Run the two probe queries through ``search`` (query -> results shown)."""
    with_target = search("{} {}".format(anchor, TARGET))
    with_nonsense = search("{} {}".format(anchor, NONSENSE))
    return {
        "d": max(r.coverage for r in with_target)
        - max(r.coverage for r in with_nonsense),
        "lexical": max(r.lexical_score for r in with_nonsense),
    }


def run_trial(seed: int) -> dict:
    rng = random.Random(seed)
    vocabulary = ["w{:04d}".format(index) for index in range(VOCABULARY_SIZE)]
    visible = _texts(rng, vocabulary, VISIBLE)
    hidden = _texts(rng, vocabulary, HIDDEN)
    anchor = rng.choice(rng.choice(visible).split())

    worlds = {"none": (visible, []), "absent": (visible, hidden)}
    for count in PRESENT_COUNTS:
        worlds["present-{}".format(count)] = (
            visible,
            _with_target(rng, hidden, count),
        )
    shared_visible = _with_target(rng, visible, SHARED_COUNT)
    if anchor not in " ".join(shared_visible).split():
        raise AssertionError("the anchor was overwritten")
    worlds["shared"] = (shared_visible, hidden)

    outcome = {}
    for name, (visible_texts, hidden_texts) in worlds.items():
        total = len(visible_texts) + len(hidden_texts)
        with _store(visible_texts, hidden_texts) as full:
            after_probe = _probe(
                lambda query: [
                    result
                    for result in full.search(query, limit=total, record_access=False)
                    if result.memory.channel != HIDDEN_CHANNEL
                ],
                anchor,
            )
            built_probe = _probe(
                lambda query: full.search(
                    query, limit=total, record_access=False, viewers=VIEWER
                ),
                anchor,
            )
        with _store(visible_texts, []) as alone:
            before_probe = _probe(
                lambda query: alone.search(query, limit=total, record_access=False),
                anchor,
            )
        outcome[name] = {
            "after": after_probe,
            "before": before_probe,
            "built": built_probe,
        }
    return outcome


def summarise(trials: list) -> dict:
    worlds = list(trials[0])
    summary = {}
    for world in worlds:
        summary[world] = {}
        for arm in ARMS:
            values = [trial[world][arm]["d"] for trial in trials]
            summary[world][arm] = {
                "positive": sum(value > ZERO for value in values),
                "zero": sum(abs(value) <= ZERO for value in values),
                "negative": sum(value < -ZERO for value in values),
                "mean_d": sum(values) / len(values),
                "min_d": min(values),
                "max_d": max(values),
            }
    # Second leak: the raw score of a query with no hidden match still moves
    # when hidden memories exist at all.
    shifts = [
        trial["absent"]["after"]["lexical"] / trial["none"]["after"]["lexical"] - 1
        for trial in trials
    ]
    summary["raw_score_shift"] = {
        "after_changed": sum(abs(shift) > ZERO for shift in shifts),
        "after_mean_relative_shift": sum(shifts) / len(shifts),
    }
    for arm in ("before", "built"):
        summary["raw_score_shift"]["{}_changed".format(arm)] = sum(
            trial["absent"][arm]["lexical"] != trial["none"][arm]["lexical"]
            for trial in trials
        )
    summary["built_differs_from_before"] = sum(
        trial[world]["built"] != trial[world]["before"]
        for trial in trials
        for world in worlds
    )
    return summary


def decide(summary: dict, trials: int) -> dict:
    needed = 0.95 * trials
    probe_works = all(
        summary["shared"][arm]["positive"] + summary["shared"][arm]["negative"]
        >= needed
        for arm in ARMS
    )
    negative_holds = all(
        summary[world][arm]["zero"] == trials
        for world in ("absent", "none")
        for arm in ARMS
    )
    present = ["present-{}".format(count) for count in PRESENT_COUNTS]
    after_leaks = all(summary[world]["after"]["positive"] >= needed for world in present)
    clean = {
        arm: all(summary[world][arm]["zero"] == trials for world in present)
        for arm in ("before", "built")
    }
    built_matches = summary["built_differs_from_before"] == 0
    if not (probe_works and negative_holds):
        verdict = "inconclusive (a control failed)"
    elif after_leaks and clean["before"]:
        verdict = "H2 supported"
    else:
        verdict = "H2 not supported"
    if not (probe_works and negative_holds):
        fix = "inconclusive (a control failed)"
    elif clean["built"] and built_matches:
        fix = "the built fix closes the channel"
    else:
        fix = "the built fix does not close the channel"
    return {
        "rule_1_probe_works": probe_works,
        "rule_2_negative_control_holds": negative_holds,
        "rule_3_after_arm_leaks": after_leaks,
        "rule_3_before_arm_clean": clean["before"],
        "rule_3_built_arm_clean": clean["built"],
        "rule_4_built_matches_before": built_matches,
        "verdict": verdict,
        "verdict_on_the_fix": fix,
    }


def render(report: dict) -> str:
    trials = report["trials"]
    lines = [
        "# Side channel in search scores (H2)",
        "",
        "Synthetic, scripted data; lexical search only. {} trials (seeds 0 to {}).".format(
            trials, trials - 1
        ),
        "D = coverage(\"anchor w\") - coverage(\"anchor z\"), read from the results",
        "the viewer is shown. See the docstring of",
        "`experiments/audience_side_channel.py` for the design and decision rule.",
        "",
        "| World | Arm | D > 0 | D = 0 | D < 0 | Mean D |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for world in report["worlds"]:
        for arm in ARMS:
            row = report["summary"][world][arm]
            lines.append(
                "| {} | {} | {} | {} | {} | {:.4f} |".format(
                    world,
                    ARM_LABELS[arm],
                    row["positive"],
                    row["zero"],
                    row["negative"],
                    row["mean_d"],
                )
            )
    shift = report["summary"]["raw_score_shift"]
    decision = report["decision"]
    lines += [
        "",
        "Raw score shift (best lexical score for \"anchor z\", `absent` against `none`):",
        "",
        "- {}: changed in {} of {} trials, mean relative shift {:+.2%}".format(
            ARM_LABELS["after"],
            shift["after_changed"],
            trials,
            shift["after_mean_relative_shift"],
        ),
        "- {}: changed in {} of {} trials".format(
            ARM_LABELS["before"], shift["before_changed"], trials
        ),
        "- {}: changed in {} of {} trials".format(
            ARM_LABELS["built"], shift["built_changed"], trials
        ),
        "",
        "Decision:",
        "",
        "- Rule 1, probe works (shared): {}".format(decision["rule_1_probe_works"]),
        "- Rule 2, negative control holds (absent, none): {}".format(
            decision["rule_2_negative_control_holds"]
        ),
        "- Rule 3, filter-after leaks for every k: {}".format(
            decision["rule_3_after_arm_leaks"]
        ),
        "- Rule 3, filter-before is clean for every k: {}".format(
            decision["rule_3_before_arm_clean"]
        ),
        "- Rule 3, the built arm is clean for every k: {}".format(
            decision["rule_3_built_arm_clean"]
        ),
        "- Rule 4, the built arm equals filter-before in every trial and world: {}"
        " ({} of {} comparisons differ)".format(
            decision["rule_4_built_matches_before"],
            report["summary"]["built_differs_from_before"],
            trials * len(report["worlds"]),
        ),
        "- **Verdict: {}; {}**".format(
            decision["verdict"], decision["verdict_on_the_fix"]
        ),
        "",
        "Limits: the data were written alongside the code, so this shows that the",
        "mechanism exists and that the fix removes it, not how often it matters in",
        "real use. Only the lexical path was tested here (no embedder, no links);",
        "the link paths are covered by unit tests in `tests/test_audience.py`.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "audience_side_channel"
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")

    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "worlds": list(trials[0]),
        "settings": {
            "vocabulary": VOCABULARY_SIZE,
            "words_per_memory": WORDS_PER_MEMORY,
            "visible": VISIBLE,
            "hidden": HIDDEN,
            "present_counts": PRESENT_COUNTS,
            "shared_count": SHARED_COUNT,
        },
        "summary": summary,
        "decision": decide(summary, arguments.trials),
    }
    text = render(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    arguments.output.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
