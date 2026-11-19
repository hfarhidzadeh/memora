"""Write outcomes: does a bound writer's write depend on claims it cannot see?

Question (E3; the write side of section 8 of docs/multi_user_memory_review.md).
Reads with ``viewers`` were shown not to depend on hidden memories
(``audience_noninterference.py``). A write can still depend on them: a write
is refused when it would change a key's scope or personal subject, or when a
lapsed authority revises its own delegated claim, and both checks once read
every claim of the key. A refusal that depends on a hidden claim tells the
writer that the claim exists. Does the outcome of a bound writer's write (and
what the viewers read afterwards) depend on claims the writer cannot see?

Data. Scripted and synthetic, no LLM, no real text. Every pair is built from
seed s. Members alice, bob, carol and dana; dana is a workspace admin. Channel
``design`` holds all four; channel ``dm`` holds bob and dana only, so alice and
carol cannot see anything drawn from a source message in ``dm``. Keys k0 to k2.
All writes go through ``store.for_writer(member)``: a source message in the
claim's source channel, then the keyed claim in ``design`` listing it.

- Visible script ("s-e3-visible"), 4 to 8 claims, each by alice or carol (50/50)
  on a random key, with probability 0.25 about a person (a personal fact whose
  subject is alice, bob or carol at random), otherwise a team claim; its source
  is ``design``. With probability 0.6 the script starts by granting alice the
  delegation of one random key (dana, the admin, grants it before any hidden
  write), and with probability 0.7 of those, revokes it after a random later
  claim.
- Hidden script ("s-e3-hidden"), 2 to 5 claims, each by bob or dana on a random
  key. dana's are team claims. bob's are, with probability 0.5, personal facts
  about carol or alice (at random), otherwise team claims. The source is ``dm``.
  Each is inserted at a random position after the first visible step.
- Contents are unique strings ("k1 visible 3", "k0 hidden 1") and serve as
  labels, so results never depend on IDs or timestamps.

Worlds.
- W: the visible script alone.
- H: the visible script with the hidden claims inserted (source ``dm``).
- H2: H rebuilt from the same seeds.
- V (positive control): as H, but the hidden claims' source is ``design``, so
  alice and carol can see them.

Observations. Per world: the outcome of every visible step (``ok`` or the
store's refusal message), and the reads of three audiences ([alice], [carol],
[alice, carol]): ``search`` for each key name (limit 10, ``record_access``
False) with claim status, scores and links, ``list_memories`` with archived
versions, and ``get_memory`` for each visible claim that was written. Visible
memories show as their content, and a ``superseded_by`` that names a memory
outside the visible script shows as "?".

Checks, per pair.
- Determinism (negative control): H2 equals H in every outcome and read.
- Power (positive control): V differs from W in some outcome or read.
- Noninterference: H equals W in every visible outcome and every read.

Reported, not part of the verdict: pairs whose visible script was refused at
least once; hidden claims accepted; which part of H differs from W when
noninterference fails (an outcome, or only reads), and the kinds of refusal
involved.

Decision rule, fixed before running:
  1. Determinism holds in every pair.
  2. Power: V differs from W in at least 50% of pairs.
  3. Noninterference holds in every pair.
Verdict: "inconclusive (control <n> failed)" if rule 1 or 2 fails; otherwise
"the write path passes" if rule 3 holds, else "the write path fails rule 3
(first failing seed <s>)".

Validation (this script's positive control against the store), fixed before
running: the script is run unchanged against the store of commit 5809ca9,
before the write checks were limited to the writer's view. It is judged able
to detect the leak if rule 3 fails in at least 25% of pairs there.

Default: 500 pairs (seeds 0 to 499).

Run from the project root:

    python3 -I experiments/audience_write_outcomes.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore  # noqa: E402
from dynmemory.store import SCHEMA_VERSION  # noqa: E402

KEYS = ("k0", "k1", "k2")
MEMBERS = ("alice", "bob", "carol", "dana")
DESIGN = "design"
DM = "dm"
VISIBLE_WRITERS = ("alice", "carol")
VIEWER_SETS = (("alice",), ("carol",), ("alice", "carol"))
VISIBLE_CLAIMS = (4, 8)
HIDDEN_CLAIMS = (2, 5)
VISIBLE_PERSONAL = 0.25
HIDDEN_PERSONAL = 0.5
DELEGATE = 0.6
REVOKE = 0.7
SEARCH_LIMIT = 10
LIST_LIMIT = 1000
POWER_SHARE = 0.5
DEFAULT_TRIALS = 500
WORLDS = ("W", "H", "H2", "V")
CHECKS = ("determinism", "power", "noninterference")
RULES = {number: name for number, name in enumerate(CHECKS, start=1)}
DETAILS_KEPT = 10


def _visible_script(seed: int) -> list:
    rng = random.Random("{}-e3-visible".format(seed))
    steps = []
    for index in range(rng.randint(*VISIBLE_CLAIMS)):
        subject = (
            rng.choice(("alice", "bob", "carol"))
            if rng.random() < VISIBLE_PERSONAL
            else None
        )
        steps.append(
            {
                "op": "claim",
                "writer": rng.choice(VISIBLE_WRITERS),
                "key": rng.choice(KEYS),
                "subject": subject,
                "content": "",
            }
        )
    if rng.random() < DELEGATE:
        key = rng.choice(KEYS)
        steps.insert(0, {"op": "grant", "key": key})
        if rng.random() < REVOKE:
            steps.insert(rng.randint(2, len(steps)), {"op": "revoke", "key": key})
    for index, step in enumerate(steps):
        if step["op"] == "claim":
            step["content"] = "{} visible {}".format(step["key"], index)
    return steps


def _hidden_script(seed: int, length: int) -> list:
    rng = random.Random("{}-e3-hidden".format(seed))
    claims = []
    for index in range(rng.randint(*HIDDEN_CLAIMS)):
        writer = rng.choice(("bob", "dana"))
        subject = None
        if writer == "bob" and rng.random() < HIDDEN_PERSONAL:
            subject = rng.choice(("carol", "alice"))
        key = rng.choice(KEYS)
        claims.append(
            {
                "op": "claim",
                "writer": writer,
                "key": key,
                "subject": subject,
                "content": "{} hidden {}".format(key, index),
                "position": rng.randint(1, length),
            }
        )
    return claims


def make_pair(seed: int) -> dict:
    """Everything about a pair that involves no store."""
    visible = _visible_script(seed)
    return {
        "seed": seed,
        "visible": visible,
        "hidden": _hidden_script(seed, len(visible)),
    }


def sequence(pair: dict, world: str) -> list:
    """(script, step) pairs in the order a world runs them."""
    steps = [("visible", step) for step in pair["visible"]]
    if world == "W":
        return steps
    # Insert from the highest position down so earlier insertions stay valid.
    for claim in sorted(pair["hidden"], key=lambda item: -item["position"]):
        steps.insert(min(claim["position"], len(steps)), ("hidden", claim))
    return steps


def _outcome(error: ValueError) -> str:
    return "refused: {}".format(error)


def _run_step(store: MemoryStore, step: dict, source: str, names: dict) -> str:
    try:
        if step["op"] == "grant":
            store.grant_delegation(DESIGN, step["key"], "alice", granted_by="dana")
        elif step["op"] == "revoke":
            store.revoke_delegation(DESIGN, step["key"], "alice", revoked_by="dana")
        else:
            writer = store.for_writer(step["writer"])
            message = writer.add_message(step["content"], channel=source)
            memory = writer.add_memory(
                step["content"],
                "fact",
                channel=DESIGN,
                memory_key=step["key"],
                fact_subject=step["subject"],
                source_message_ids=[message.id],
            )
            names[memory.id] = step["content"]
    except ValueError as error:
        return _outcome(error)
    return "ok"


def _name(names: dict, memory_id) -> object:
    if memory_id is None:
        return None
    return names.get(memory_id, "?")


def _memory_row(names: dict, memory) -> object:
    if memory is None:
        return None
    return (
        memory.content,
        memory.speaker,
        memory.archived_at is not None,
        _name(names, memory.superseded_by),
    )


def _reads(store: MemoryStore, names: dict, pair: dict) -> dict:
    reads = {}
    for viewers in VIEWER_SETS:
        label = "+".join(viewers)
        reads[label] = {
            "search": {
                key: [
                    (
                        result.memory.content,
                        result.claim_status,
                        result.score,
                        result.lexical_score,
                        result.coverage,
                        tuple(
                            sorted(
                                (
                                    link.relation.value,
                                    _name(names, link.source_id),
                                    _name(names, link.target_id),
                                )
                                for link in result.links
                            )
                        ),
                    )
                    for result in store.search(
                        key,
                        limit=SEARCH_LIMIT,
                        record_access=False,
                        viewers=list(viewers),
                    )
                ]
                for key in KEYS
            },
            "list": sorted(
                (
                    _memory_row(names, memory)
                    for memory in store.list_memories(
                        include_archived=True, limit=LIST_LIMIT, viewers=list(viewers)
                    )
                ),
                key=repr,
            ),
            "get": {
                content: _memory_row(
                    names, store.get_memory(memory_id, viewers=list(viewers))
                )
                for memory_id, content in sorted(names.items(), key=lambda item: item[1])
                if content.split()[1] == "visible"
            },
        }
    return reads


def observe(pair: dict, world: str) -> dict:
    """Build one world in a fresh store; return its outcomes and reads."""
    hidden_source = DESIGN if world == "V" else DM
    names: dict = {}
    with MemoryStore(":memory:") as store:
        for member in MEMBERS:
            store.join_channel(DESIGN, member)
        for member in ("bob", "dana"):
            store.join_channel(DM, member)
        store.add_admin("dana")
        visible, hidden = [], []
        for script, step in sequence(pair, world):
            if script == "visible":
                visible.append(_run_step(store, step, DESIGN, names))
            else:
                hidden.append(_run_step(store, step, hidden_source, names))
        return {"visible": visible, "hidden": hidden, "reads": _reads(store, names, pair)}


def _refusal_kind(outcome: str) -> str:
    if "scope or personal subject" in outcome:
        return "scope"
    if "lapsed authority" in outcome:
        return "lapsed authority"
    return "other refusal"


def _difference(left: dict, right: dict) -> list:
    """What differs between two observations of the visible script."""
    found = []
    pairs = [
        (index, one, other)
        for index, (one, other) in enumerate(zip(left["visible"], right["visible"]))
        if one != other
    ]
    if pairs:
        found.append(
            "outcome of step {}: {} against {}".format(
                pairs[0][0], pairs[0][1][:80], pairs[0][2][:80]
            )
        )
    if left["reads"] != right["reads"]:
        for audience in left["reads"]:
            for part in left["reads"][audience]:
                if left["reads"][audience][part] != right["reads"][audience][part]:
                    found.append("reads of {} ({})".format(audience, part))
    return found


def evaluate(pair: dict, seen: dict) -> dict:
    w, h, h2, v = (seen[world] for world in WORLDS)
    details = {}
    determinism = h == h2
    if not determinism:
        details["determinism"] = "H and H2 differ"
    power = bool(_difference(v, w))
    differences = _difference(h, w)
    if differences:
        details["noninterference"] = "; ".join(differences[:3])
    outcome_differs = any(one != other for one, other in zip(h["visible"], w["visible"]))
    kinds = sorted(
        {
            _refusal_kind(outcome)
            for outcome in h["visible"] + w["visible"]
            if outcome != "ok"
        }
    )
    return {
        "seed": pair["seed"],
        "checks": {
            "determinism": determinism,
            "power": power,
            "noninterference": not differences,
        },
        "details": details,
        "visible_refused": any(outcome != "ok" for outcome in w["visible"]),
        "hidden_accepted": sum(outcome == "ok" for outcome in h["hidden"]),
        "hidden_total": len(h["hidden"]),
        "outcome_differs": outcome_differs,
        "reads_only_differ": bool(differences) and not outcome_differs,
        "refusal_kinds": kinds,
    }


def run_trial(seed: int) -> dict:
    pair = make_pair(seed)
    return evaluate(pair, {world: observe(pair, world) for world in WORLDS})


def summarise(trials: list) -> dict:
    checks = {
        name: {
            "held": sum(trial["checks"][name] for trial in trials),
            "not_held_seeds": [trial["seed"] for trial in trials if not trial["checks"][name]],
        }
        for name in CHECKS
    }
    failing = [trial for trial in trials if not trial["checks"]["noninterference"]]
    failures = [
        {"seed": trial["seed"], "detail": trial["details"]["noninterference"]}
        for trial in failing[:DETAILS_KEPT]
    ]
    return {
        "checks": checks,
        "pairs_with_visible_refusal": sum(trial["visible_refused"] for trial in trials),
        "hidden_claims": sum(trial["hidden_total"] for trial in trials),
        "hidden_claims_accepted": sum(trial["hidden_accepted"] for trial in trials),
        "failing_with_outcome_difference": sum(trial["outcome_differs"] for trial in failing),
        "failing_with_reads_only": sum(trial["reads_only_differ"] for trial in failing),
        "failing_by_refusal_kind": {
            kind: sum(kind in trial["refusal_kinds"] for trial in failing)
            for kind in ("scope", "lapsed authority", "other refusal")
        },
        "failures": failures,
    }


def decide(summary: dict, trials: int) -> dict:
    checks = summary["checks"]
    holds = {
        1: checks["determinism"]["held"] == trials,
        2: checks["power"]["held"] >= POWER_SHARE * trials,
        3: checks["noninterference"]["held"] == trials,
    }
    failed_controls = [number for number in (1, 2) if not holds[number]]
    if failed_controls:
        verdict = "inconclusive (control {} failed)".format(
            ", ".join(str(number) for number in failed_controls)
        )
    elif holds[3]:
        verdict = "the write path passes"
    else:
        verdict = "the write path fails rule 3 (first failing seed {})".format(
            min(checks["noninterference"]["not_held_seeds"])
        )
    decision = {"rule_{}_{}".format(number, name): holds[number] for number, name in RULES.items()}
    decision["verdict"] = verdict
    return decision


def _share(count: int, total: int) -> str:
    return "-" if not total else "{} of {} ({:.1f}%)".format(count, total, 100.0 * count / total)


def render(report: dict) -> str:
    trials, summary, decision = report["trials"], report["summary"], report["decision"]
    checks = summary["checks"]
    failing = trials - checks["noninterference"]["held"]
    lines = [
        "# Write outcomes of bound writers (E3)",
        "",
        "Synthetic, scripted data; no LLM. {} pairs (seeds 0 to {}).".format(trials, trials - 1),
        "Store under test: schema version {}.".format(report["store_schema_version"]),
        "See the docstring of `experiments/audience_write_outcomes.py` for the design",
        "and decision rule.",
        "",
        "| Rule | Check | Held in | Needed | Holds |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = (
        (1, "Determinism: H and H2 identical", "every pair"),
        (2, "Power: V (visible hidden claims) differs from W", "at least 50%"),
        (3, "Noninterference: H equals W in every visible outcome and read", "every pair"),
    )
    for number, text, needed in rows:
        name = RULES[number]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                number,
                text,
                _share(checks[name]["held"], trials),
                needed,
                decision["rule_{}_{}".format(number, name)],
            )
        )
    lines += [
        "",
        "**Verdict: {}**".format(decision["verdict"]),
        "",
        "Reported, not part of the verdict:",
        "",
        "- Pairs whose visible script (world W) was refused at least once: {}.".format(
            _share(summary["pairs_with_visible_refusal"], trials)
        ),
        "- Hidden claims accepted in H: {} of {}.".format(
            summary["hidden_claims_accepted"], summary["hidden_claims"]
        ),
        "- Pairs failing noninterference: {}. In {} a visible outcome differed; in {}"
        " only reads differed. Refusal kinds present in failing pairs: scope {},"
        " lapsed authority {}, other {}.".format(
            failing,
            summary["failing_with_outcome_difference"],
            summary["failing_with_reads_only"],
            summary["failing_by_refusal_kind"]["scope"],
            summary["failing_by_refusal_kind"]["lapsed authority"],
            summary["failing_by_refusal_kind"]["other refusal"],
        ),
    ]
    if summary["failures"]:
        lines += ["", "## Pairs where noninterference did not hold (first {})".format(DETAILS_KEPT), ""]
        lines += [
            "- seed {}: {}".format(item["seed"], item["detail"]) for item in summary["failures"]
        ]
    lines += [
        "",
        "Limits: the data were written alongside the code, so a pass shows that the",
        "write path holds on these scripts, not on every database. Only bound writers",
        "who cannot see the hidden claims are compared; bob and dana, who can see them,",
        "legitimately get different outcomes. Not exercised: retractions, pruning,",
        "`restore`, other admins, and delegation by anyone but dana to anyone but alice.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "audience_write_outcomes"
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")
    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "store_schema_version": SCHEMA_VERSION,
        "settings": {
            "keys": len(KEYS),
            "visible_claims": VISIBLE_CLAIMS,
            "hidden_claims": HIDDEN_CLAIMS,
            "visible_personal": VISIBLE_PERSONAL,
            "hidden_personal": HIDDEN_PERSONAL,
            "delegate": DELEGATE,
            "revoke": REVOKE,
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
