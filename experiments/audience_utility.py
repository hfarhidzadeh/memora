"""Zero leakage at no utility cost: does filtering before retrieval cost a viewer answers?

Question (H1 in section 9 of docs/multi_user_memory_review.md). Does filtering
before retrieval, as built (``search(..., viewers=...)``), give zero leakage at
no utility cost, compared with filtering after retrieval? The review lists a
fourth arm, filtering before retrieval plus per-audience summaries. It does not
apply here: Memvane has no consolidation, so there are no summaries to build
per audience.

Data. Scripted and synthetic, no LLM, no real text. Each trial builds one
workspace in one store:
- Members ana, ben, cy, dee, eli and fay. Channel ``general`` holds all six;
  ``p1``, ``p2`` and ``p3`` each hold a random subset of 2, 3 or 4 members
  (size drawn uniformly). Memberships do not change during a trial.
- 25 pseudo-word subjects (sub00 to sub24) and eight attributes (deadline,
  budget, owner, venue, status, vendor, risk, metric). Values are pseudo-words
  (val000 to val199); filler words come from a separate vocabulary of 200
  (fil000 to fil199). No word belongs to two vocabularies.
- Facts: 40 in ``general`` and 30 in each private channel. Each takes a
  (subject, attribute) pair drawn at random with replacement, so a pair can
  recur within a channel and across channels, with different values. Its text
  is "The {attribute} of {subject} is {value}." followed by 0 to 6 filler
  words; its kind is ``fact`` and its speaker a random member of the channel.
- Canaries: 5 per channel, ``general`` included, written exactly like facts
  (same template, same 0 to 6 filler words) but with the unique value
  ``canary<seed>x<k>``, k = 0 to 19 within the trial. The speaker is a random
  member of the channel and is the canary's owner.
- 150 memories in all, written in one random order. Equal scores rank the
  newer memory first, so writing channel by channel would let one channel win
  every tie. No links, no keys, lexical search only (no embedder), k = 10.
Workspace draws come from ``Random("<seed>-workspace")`` and question draws
from ``Random("<seed>-questions")``. Memory IDs are mapped to labels assigned
at writing ("p2/fact07", "general/canary3"); an ID with no label would show as
"?" + ID. Nothing reported depends on IDs or on wall-clock time.

Questions. The query for a (subject, attribute) pair is "{attribute}
{subject}". Each member is a viewer on their own (``viewers`` names one
member). A memory is visible to a viewer if they belong to its channel; the
after arms and every statistic decide this from the experiment's own record of
channels and memberships, never from the store's ``viewers`` code.
- Utility: per viewer, 20 pairs drawn without replacement from the pairs that
  have at least one memory visible to that viewer. Gold is every memory
  visible to the viewer with that pair, canaries included. Scores: the
  reciprocal rank (RR) of the first gold memory among the first k results
  shown (0 if none), and hit@k. Gold does not depend on the arm, so the pooled
  arm is scored against what the viewer may see.
- Leakage: for each viewer and each canary in a private channel the viewer is
  not in, the viewer asks the canary's query. A leak: the canary is among the
  first k results the viewer is shown.
- Owner: for each canary in a private channel, its owner asks its query. A
  hit: the canary is among the first k results shown.
- Over-blocking: for each ``general`` canary and each of the six members, the
  member asks its query in the built arm with ``limit=1000``. The canary must
  be returned; hit@k (within the first k of that list) is also reported.

Arms (BIG = 1000, more than the 150 memories in the store; every read passes
``record_access=False``).
- pooled: ``search(q, limit=BIG)``, unfiltered, first k: one shared store with
  no access control.
- after-k: ``search(q, limit=k)``, then the results the viewer cannot see are
  dropped. This is the naive version and can show fewer than k results.
- after-all: ``search(q, limit=BIG)``, drop what the viewer cannot see, then
  take the first k.
- built: ``search(q, limit=k, viewers=viewer)``.
All four read the same store, and reads do not change it, so each distinct
call is made once per trial and its result is shared by the arms and viewers
that make the same call (pooled and after-all both call
``search(q, limit=BIG)``).

Controls.
- Positive: the pooled arm must leak canaries, or a clean filtered arm means
  nothing; and owners must find their own canaries in the built arm, or a
  clean built arm could simply be returning too little.
- Negative (no over-blocking): a canary in ``general`` must be returned to
  every member by the built arm.
- Sanity, reported: every store held at most BIG memories, so after-all saw
  the whole ranking; and write timestamps rose strictly in write order, so
  equal scores were ordered by write order and never by random IDs.

Decision rule, fixed before running:
  1. Positive controls: pooled leaks in at least 50% of leakage items, pooled
     over all trials; and the owner hit@k in the built arm is at least 95%.
  2. Negative control: every over-blocking item is returned by the built arm,
     in every trial (100%).
  3. Zero leakage: built, after-k and after-all leak in 0 leakage items in
     every trial.
  4. Utility: D = the per-trial mean RR of built minus that of after-all, over
     the utility items. Take the mean of D over trials and a 95% percentile
     bootstrap CI over trials (2,000 resamples drawn with ``Random(0)``; the
     bounds are the 51st and 1,950th of the sorted resampled means, as in
     ``longmemeval_knowledge_update.py``). The reading is "no utility cost" if
     the CI's lower bound is at least -0.01, "utility cost" if its upper bound
     is below -0.01, and "unclear" otherwise. Built minus after-k and built
     minus pooled are reported the same way (each with a fresh ``Random(0)``),
     descriptively, with hit@k and the mean number of results shown per arm.
Verdict: "inconclusive (a control failed)" if rule 1 or 2 fails; otherwise
"H1 supported" if rule 3 holds and rule 4 reads "no utility cost"; otherwise
"H1 not supported: <reasons>".

Run from the project root (200 trials by default):

    python3 -I experiments/audience_utility.py
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

MEMBERS = ("ana", "ben", "cy", "dee", "eli", "fay")
GENERAL = "general"
PRIVATE_CHANNELS = ("p1", "p2", "p3")
CHANNELS = (GENERAL,) + PRIVATE_CHANNELS
PRIVATE_SIZES = (2, 3, 4)
SUBJECTS = tuple("sub{:02d}".format(index) for index in range(25))
ATTRIBUTES = (
    "deadline",
    "budget",
    "owner",
    "venue",
    "status",
    "vendor",
    "risk",
    "metric",
)
VALUES = tuple("val{:03d}".format(index) for index in range(200))
FILLER = tuple("fil{:03d}".format(index) for index in range(200))
MAX_FILLER_WORDS = 6
FACTS_PER_CHANNEL = {GENERAL: 40, "p1": 30, "p2": 30, "p3": 30}
CANARIES_PER_CHANNEL = 5
K = 10
BIG = 1000
UTILITY_PAIRS_PER_VIEWER = 20
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
UTILITY_MARGIN = 0.01
POOLED_LEAK_SHARE = 0.5
OWNER_HIT_SHARE = 0.95
ZERO = 1e-12
DEFAULT_TRIALS = 200
ITEM_TYPES = ("utility", "leakage", "owner", "over_blocking")
ARMS = ("pooled", "after-k", "after-all", "built")
FILTERED_ARMS = ("after-k", "after-all", "built")
ARM_LABELS = {
    "pooled": "pooled (no access control)",
    "after-k": "filter after, first k only",
    "after-all": "filter after, whole ranking",
    "built": "search with viewers (built)",
}
# (summary key, arm subtracted from built); the first one is rule 4.
COMPARISONS = (
    ("built_minus_after_all", "after-all"),
    ("built_minus_after_k", "after-k"),
    ("built_minus_pooled", "pooled"),
)


def _content(rng: random.Random, subject: str, attribute: str, value: str) -> str:
    words = [rng.choice(FILLER) for _ in range(rng.randint(0, MAX_FILLER_WORDS))]
    return " ".join(["The {} of {} is {}.".format(attribute, subject, value)] + words)


def build_workspace(seed: int) -> dict:
    """Memberships and memories of one trial, memories in write order."""
    rng = random.Random("{}-workspace".format(seed))
    members = {GENERAL: list(MEMBERS)}
    for channel in PRIVATE_CHANNELS:
        members[channel] = sorted(rng.sample(MEMBERS, rng.choice(PRIVATE_SIZES)))
    memories = []
    canary_number = 0
    for channel in CHANNELS:
        facts = FACTS_PER_CHANNEL[channel]
        for index in range(facts + CANARIES_PER_CHANNEL):
            canary = index >= facts
            subject = rng.choice(SUBJECTS)
            attribute = rng.choice(ATTRIBUTES)
            if canary:
                value = "canary{}x{}".format(seed, canary_number)
                label = "{}/canary{}".format(channel, index - facts)
                canary_number += 1
            else:
                value = rng.choice(VALUES)
                label = "{}/fact{:02d}".format(channel, index)
            memories.append(
                {
                    "label": label,
                    "channel": channel,
                    "subject": subject,
                    "attribute": attribute,
                    "value": value,
                    "canary": canary,
                    "speaker": rng.choice(members[channel]),
                    "content": _content(rng, subject, attribute, value),
                }
            )
    rng.shuffle(memories)
    return {"seed": seed, "members": members, "memories": memories}


def visible_channels(workspace: dict) -> dict:
    """Each member's channels, from the experiment's own record."""
    return {
        member: frozenset(
            channel for channel in CHANNELS if member in workspace["members"][channel]
        )
        for member in MEMBERS
    }


def query_for(memory: dict) -> str:
    return "{} {}".format(memory["attribute"], memory["subject"])


def questions(workspace: dict) -> list:
    """Every question of the trial, before any store exists."""
    rng = random.Random("{}-questions".format(workspace["seed"]))
    channels = visible_channels(workspace)
    memories = workspace["memories"]
    asked = []
    for viewer in MEMBERS:
        visible = [memory for memory in memories if memory["channel"] in channels[viewer]]
        # Sorted, so the draw does not depend on set order.
        pairs = sorted({(memory["subject"], memory["attribute"]) for memory in visible})
        for subject, attribute in rng.sample(
            pairs, min(UTILITY_PAIRS_PER_VIEWER, len(pairs))
        ):
            gold = sorted(
                memory["label"]
                for memory in visible
                if (memory["subject"], memory["attribute"]) == (subject, attribute)
            )
            asked.append(
                {
                    "type": "utility",
                    "viewer": viewer,
                    "query": "{} {}".format(attribute, subject),
                    "gold": gold,
                }
            )
    # Labels sort in the order canaries were drawn: general, p1, p2, p3.
    canaries = sorted(
        (memory for memory in memories if memory["canary"]),
        key=lambda memory: memory["label"],
    )
    private = [canary for canary in canaries if canary["channel"] != GENERAL]
    for viewer in MEMBERS:
        for canary in private:
            if canary["channel"] not in channels[viewer]:
                asked.append(
                    {
                        "type": "leakage",
                        "viewer": viewer,
                        "query": query_for(canary),
                        "canary": canary["label"],
                    }
                )
    for canary in private:
        asked.append(
            {
                "type": "owner",
                "viewer": canary["speaker"],
                "query": query_for(canary),
                "canary": canary["label"],
            }
        )
    for canary in canaries:
        if canary["channel"] == GENERAL:
            for member in MEMBERS:
                asked.append(
                    {
                        "type": "over_blocking",
                        "viewer": member,
                        "query": query_for(canary),
                        "canary": canary["label"],
                    }
                )
    return asked


def populate(store: MemoryStore, workspace: dict) -> tuple:
    """Write the workspace; return the ID-to-label map and the write timestamps."""
    for channel in CHANNELS:
        for member in workspace["members"][channel]:
            store.join_channel(channel, member)
    labels = {}
    stamps = []
    for memory in workspace["memories"]:
        written = store.add_memory(
            memory["content"],
            MemoryType.FACT,
            channel=memory["channel"],
            speaker=memory["speaker"],
        )
        labels[written.id] = memory["label"]
        stamps.append(written.created_at)
    return labels, stamps


def shown_by_arm(read, visible, viewer: str, query: str) -> dict:
    """What the viewer is shown in each arm.

    ``read(query, limit, viewer=None)`` returns the labels of a search, in
    order; ``visible(viewer, label)`` applies the experiment's own record.
    """
    return {
        "pooled": read(query, BIG)[:K],
        "after-k": [label for label in read(query, K) if visible(viewer, label)],
        "after-all": [label for label in read(query, BIG) if visible(viewer, label)][:K],
        "built": list(read(query, K, viewer)),
    }


def answer(store: MemoryStore, labels: dict, workspace: dict, asked: list) -> list:
    """Ask every question in every arm; return the questions with what was shown."""
    channels = visible_channels(workspace)
    channel_of = {memory["label"]: memory["channel"] for memory in workspace["memories"]}
    calls = {}

    def read(query: str, limit: int, viewer=None) -> list:
        key = (query, limit, viewer)
        if key not in calls:
            calls[key] = [
                labels.get(result.memory.id, "?" + result.memory.id)
                for result in store.search(
                    query, limit=limit, record_access=False, viewers=viewer
                )
            ]
        return calls[key]

    def visible(viewer: str, label: str) -> bool:
        # An unlabelled result has no recorded channel and is never visible.
        return channel_of.get(label) in channels[viewer]

    answered = []
    for item in asked:
        if item["type"] == "over_blocking":
            shown = {"built": list(read(item["query"], BIG, item["viewer"]))}
        else:
            shown = shown_by_arm(read, visible, item["viewer"], item["query"])
        answered.append(dict(item, shown=shown))
    return answered


def reciprocal_rank(shown: list, gold: list) -> float:
    for rank, label in enumerate(shown[:K], start=1):
        if label in gold:
            return 1.0 / rank
    return 0.0


def score_items(items: list) -> dict:
    """Per-trial counts and means from the answered questions."""
    by_type = {name: [item for item in items if item["type"] == name] for name in ITEM_TYPES}
    utility = by_type["utility"]
    scores = {"utility": {"items": len(utility), "rr": {}, "hit": {}, "shown": {}}}
    for arm in ARMS:
        ranks = [reciprocal_rank(item["shown"][arm], item["gold"]) for item in utility]
        scores["utility"]["rr"][arm] = sum(ranks) / len(ranks)
        scores["utility"]["hit"][arm] = sum(rank > 0 for rank in ranks) / len(ranks)
        scores["utility"]["shown"][arm] = sum(
            len(item["shown"][arm]) for item in utility
        ) / len(utility)
    for name, field in (("leakage", "leaks"), ("owner", "hits")):
        scores[name] = {
            "items": len(by_type[name]),
            field: {
                arm: sum(item["canary"] in item["shown"][arm][:K] for item in by_type[name])
                for arm in ARMS
            },
        }
    over = by_type["over_blocking"]
    scores["over_blocking"] = {
        "items": len(over),
        "returned": sum(item["canary"] in item["shown"]["built"] for item in over),
        "hit_at_k": sum(item["canary"] in item["shown"]["built"][:K] for item in over),
    }
    return scores


def run_trial(seed: int) -> dict:
    workspace = build_workspace(seed)
    asked = questions(workspace)
    with MemoryStore(":memory:") as store:
        labels, stamps = populate(store, workspace)
        items = answer(store, labels, workspace, asked)
    return {
        "seed": seed,
        "members": {channel: workspace["members"][channel] for channel in PRIVATE_CHANNELS},
        "memories": len(labels),
        "timestamps_increase": all(
            earlier < later for earlier, later in zip(stamps, stamps[1:])
        ),
        "items": items,
        "scores": score_items(items),
    }


def _share(part: float, whole: float):
    return part / whole if whole else None


def _first_seed(trials: list, failed) -> tuple:
    """The lowest seed whose trial failed, and how many trials failed."""
    seeds = [trial["seed"] for trial in trials if failed(trial)]
    return (min(seeds) if seeds else None), len(seeds)


def bootstrap(values: list) -> dict:
    """Mean and 95% percentile bootstrap CI over trials, drawn with ``Random(0)``."""
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "mean": sum(values) / len(values),
        "ci95": [
            means[int(0.025 * BOOTSTRAP_SAMPLES)],
            means[int(0.975 * BOOTSTRAP_SAMPLES) - 1],
        ],
        "higher": sum(value > ZERO for value in values),
        "equal": sum(abs(value) <= ZERO for value in values),
        "lower": sum(value < -ZERO for value in values),
        "values": values,
    }


def summarise(trials: list) -> dict:
    def total(name: str, field: str, arm: str) -> int:
        return sum(trial["scores"][name][field][arm] for trial in trials)

    leakage_items = sum(trial["scores"]["leakage"]["items"] for trial in trials)
    leakage = {"items": leakage_items, "by_arm": {}}
    for arm in ARMS:
        first, failing = _first_seed(
            trials, lambda trial, arm=arm: trial["scores"]["leakage"]["leaks"][arm] > 0
        )
        leaks = total("leakage", "leaks", arm)
        leakage["by_arm"][arm] = {
            "leaks": leaks,
            "share": _share(leaks, leakage_items),
            "trials_with_leak": failing,
            "first_seed": first,
        }

    owner_items = sum(trial["scores"]["owner"]["items"] for trial in trials)
    owner = {
        "items": owner_items,
        "by_arm": {
            arm: {
                "hits": total("owner", "hits", arm),
                "share": _share(total("owner", "hits", arm), owner_items),
            }
            for arm in ARMS
        },
    }
    owner["built_first_miss_seed"], owner["built_trials_with_miss"] = _first_seed(
        trials,
        lambda trial: trial["scores"]["owner"]["hits"]["built"]
        < trial["scores"]["owner"]["items"],
    )

    over_items = sum(trial["scores"]["over_blocking"]["items"] for trial in trials)
    returned = sum(trial["scores"]["over_blocking"]["returned"] for trial in trials)
    hit_at_k = sum(trial["scores"]["over_blocking"]["hit_at_k"] for trial in trials)
    first, failing = _first_seed(
        trials,
        lambda trial: trial["scores"]["over_blocking"]["returned"]
        < trial["scores"]["over_blocking"]["items"],
    )
    over_blocking = {
        "items": over_items,
        "returned": returned,
        "missing": over_items - returned,
        "hit_at_k": hit_at_k,
        "hit_at_k_share": _share(hit_at_k, over_items),
        "trials_with_missing": failing,
        "first_seed": first,
    }

    per_trial = {
        arm: [trial["scores"]["utility"]["rr"][arm] for trial in trials] for arm in ARMS
    }
    utility = {
        "items": sum(trial["scores"]["utility"]["items"] for trial in trials),
        "by_arm": {
            arm: {
                "mean_rr": sum(per_trial[arm]) / len(trials),
                "hit_at_k": sum(trial["scores"]["utility"]["hit"][arm] for trial in trials)
                / len(trials),
                "mean_shown": sum(
                    trial["scores"]["utility"]["shown"][arm] for trial in trials
                )
                / len(trials),
            }
            for arm in ARMS
        },
        "differences": {
            name: bootstrap(
                [built - other for built, other in zip(per_trial["built"], per_trial[arm])]
            )
            for name, arm in COMPARISONS
        },
    }
    return {
        "trials": len(trials),
        "leakage": leakage,
        "owner": owner,
        "over_blocking": over_blocking,
        "utility": utility,
        "sanity": {
            "most_memories": max(trial["memories"] for trial in trials),
            "big_covers_every_store": all(trial["memories"] <= BIG for trial in trials),
            "timestamps_increase": sum(trial["timestamps_increase"] for trial in trials),
        },
    }


def utility_reading(ci95: list) -> str:
    lower, upper = ci95
    if lower >= -UTILITY_MARGIN:
        return "no utility cost"
    if upper < -UTILITY_MARGIN:
        return "utility cost"
    return "unclear"


def decide(summary: dict, trials: int) -> dict:
    """Apply the decision rule to a summary.

    ``trials`` is the number of trials, kept for the signature the audience
    experiments share; every rule here is a share of items or a count of
    failures, read from the summary.
    """
    leakage = summary["leakage"]
    pooled_share = _share(leakage["by_arm"]["pooled"]["leaks"], leakage["items"])
    pooled_leaks = pooled_share is not None and pooled_share >= POOLED_LEAK_SHARE
    owner_share = _share(
        summary["owner"]["by_arm"]["built"]["hits"], summary["owner"]["items"]
    )
    owner_finds = owner_share is not None and owner_share >= OWNER_HIT_SHARE
    over = summary["over_blocking"]
    no_over_blocking = over["items"] > 0 and over["missing"] == 0
    clean = {arm: leakage["by_arm"][arm]["leaks"] == 0 for arm in FILTERED_ARMS}
    zero_leakage = all(clean.values())
    difference = summary["utility"]["differences"]["built_minus_after_all"]
    reading = utility_reading(difference["ci95"])

    reasons = []
    if not zero_leakage:
        reasons.append(
            "rule 3, leaks in "
            + ", ".join(
                "{} ({} of {} items, first failing seed {})".format(
                    arm,
                    leakage["by_arm"][arm]["leaks"],
                    leakage["items"],
                    leakage["by_arm"][arm]["first_seed"],
                )
                for arm in FILTERED_ARMS
                if not clean[arm]
            )
        )
    if reading != "no utility cost":
        reasons.append(
            "rule 4, {} (built minus after-all {:+.4f}, 95% CI {:+.4f} to {:+.4f})".format(
                "utility cost" if reading == "utility cost" else "utility unclear",
                difference["mean"],
                difference["ci95"][0],
                difference["ci95"][1],
            )
        )
    if not (pooled_leaks and owner_finds and no_over_blocking):
        verdict = "inconclusive (a control failed)"
    elif not reasons:
        verdict = "H1 supported"
    else:
        verdict = "H1 not supported: " + "; ".join(reasons)
    return {
        "rule_1_pooled_leaks": pooled_leaks,
        "rule_1_owner_finds_canary": owner_finds,
        "rule_1_positive_controls_hold": pooled_leaks and owner_finds,
        "rule_2_no_over_blocking": no_over_blocking,
        "rule_3_zero_leakage": zero_leakage,
        "rule_3_zero_leakage_by_arm": clean,
        "rule_4_utility": reading,
        "reasons": reasons,
        "verdict": verdict,
    }


def _percent(part: float, whole: float) -> str:
    return "{:.1%}".format(part / whole) if whole else "n/a"


def render(report: dict) -> str:
    trials = report["trials"]
    summary = report["summary"]
    decision = report["decision"]
    leakage = summary["leakage"]
    owner = summary["owner"]
    over = summary["over_blocking"]
    utility = summary["utility"]
    sanity = summary["sanity"]
    lines = [
        "# Zero leakage at no utility cost (H1)",
        "",
        "Synthetic, scripted workspace; lexical search only; k = {}. {} trials"
        " (seeds 0 to {}),".format(K, trials, trials - 1),
        "{} leakage, {} owner, {} utility and {} over-blocking items. See the".format(
            leakage["items"], owner["items"], utility["items"], over["items"]
        ),
        "docstring of `experiments/audience_utility.py` for the design and decision rule.",
        "",
        "| Arm | Leaks | Owner hit@k | Utility mean RR | Utility hit@k | Mean results shown |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        row = utility["by_arm"][arm]
        lines.append(
            "| {} | {} ({}) | {} ({}) | {:.4f} | {:.1%} | {:.2f} |".format(
                ARM_LABELS[arm],
                leakage["by_arm"][arm]["leaks"],
                _percent(leakage["by_arm"][arm]["leaks"], leakage["items"]),
                owner["by_arm"][arm]["hits"],
                _percent(owner["by_arm"][arm]["hits"], owner["items"]),
                row["mean_rr"],
                row["hit_at_k"],
                row["mean_shown"],
            )
        )
    lines += [
        "",
        "Utility differences: per-trial mean RR of built minus the other arm, with a",
        "95% percentile bootstrap CI over trials ({:,} resamples).".format(
            BOOTSTRAP_SAMPLES
        ),
        "",
        "| Comparison | Mean D | 95% CI | Trials higher / equal / lower | Reading |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, (name, arm) in enumerate(COMPARISONS):
        difference = utility["differences"][name]
        lines.append(
            "| built minus {} ({}) | {:+.4f} | {:+.4f} to {:+.4f} | {} / {} / {} | {} |".format(
                arm,
                "rule 4" if index == 0 else "descriptive",
                difference["mean"],
                difference["ci95"][0],
                difference["ci95"][1],
                difference["higher"],
                difference["equal"],
                difference["lower"],
                utility_reading(difference["ci95"]),
            )
        )
    lines += [
        "",
        "Over-blocking (each `general` canary asked by each member, built arm,"
        " `limit={}`):".format(BIG),
        "returned in {} of {} items, within the first k in {} ({}).".format(
            over["returned"],
            over["items"],
            over["hit_at_k"],
            _percent(over["hit_at_k"], over["items"]),
        ),
        "",
        "Sanity: the largest store held {} memories (BIG = {} covers every store: {});".format(
            sanity["most_memories"], BIG, sanity["big_covers_every_store"]
        ),
        "write timestamps rose strictly in write order in {} of {} trials.".format(
            sanity["timestamps_increase"], trials
        ),
        "",
        "Decision:",
        "",
        "- Rule 1, pooled leaks in at least {:.0%} of leakage items: {} ({})".format(
            POOLED_LEAK_SHARE,
            decision["rule_1_pooled_leaks"],
            _percent(leakage["by_arm"]["pooled"]["leaks"], leakage["items"]),
        ),
        "- Rule 1, owner hit@k in the built arm at least {:.0%}: {} ({}{})".format(
            OWNER_HIT_SHARE,
            decision["rule_1_owner_finds_canary"],
            _percent(owner["by_arm"]["built"]["hits"], owner["items"]),
            ""
            if owner["built_first_miss_seed"] is None
            else ", first seed with a miss {}".format(owner["built_first_miss_seed"]),
        ),
        "- Rule 2, every over-blocking item returned by the built arm: {} ({} missing{})".format(
            decision["rule_2_no_over_blocking"],
            over["missing"],
            ""
            if over["first_seed"] is None
            else ", first failing seed {}".format(over["first_seed"]),
        ),
        "- Rule 3, zero leakage in after-k, after-all and built: {}{}".format(
            decision["rule_3_zero_leakage"],
            "".join(
                "; {} leaked in {} trials (first failing seed {})".format(
                    arm,
                    leakage["by_arm"][arm]["trials_with_leak"],
                    leakage["by_arm"][arm]["first_seed"],
                )
                for arm in FILTERED_ARMS
                if leakage["by_arm"][arm]["leaks"]
            ),
        ),
        "- Rule 4, utility of built against after-all: {}".format(
            decision["rule_4_utility"]
        ),
        "- **Verdict: {}**".format(decision["verdict"]),
        "",
        "Limits: the data were written alongside the code, so this shows that the",
        "mechanism works, not how often it matters in real use. Gold memories are the",
        "only memories that contain both query words, so a ranker that keeps every",
        "visible memory puts gold first in almost every item; rule 4 checks that the",
        "built path loses or misorders no visible memory, and has little power to show",
        "a subtler ranking cost. Only lexical search was tested (no embedder, no links,",
        "no keys); the link paths are tested in `experiments/audience_link_paths.py`.",
        "",
    ]
    return "\n".join(lines)


def report_for(trials: list) -> dict:
    summary = summarise(trials)
    return {
        "trials": len(trials),
        "settings": {
            "members": list(MEMBERS),
            "channels": list(CHANNELS),
            "private_channel_sizes": list(PRIVATE_SIZES),
            "subjects": len(SUBJECTS),
            "attributes": list(ATTRIBUTES),
            "values": len(VALUES),
            "filler_words": len(FILLER),
            "max_filler_words": MAX_FILLER_WORDS,
            "facts_per_channel": FACTS_PER_CHANNEL,
            "canaries_per_channel": CANARIES_PER_CHANNEL,
            "k": K,
            "big": BIG,
            "utility_pairs_per_viewer": UTILITY_PAIRS_PER_VIEWER,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "utility_margin": UTILITY_MARGIN,
            "pooled_leak_share": POOLED_LEAK_SHARE,
            "owner_hit_share": OWNER_HIT_SHARE,
        },
        "summary": summary,
        "decision": decide(summary, len(trials)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "audience_utility"
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")

    report = report_for([run_trial(seed) for seed in range(arguments.trials)])
    text = render(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    arguments.output.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
