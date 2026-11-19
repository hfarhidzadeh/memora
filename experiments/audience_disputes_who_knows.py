"""Disputes and who-knows-what: do hidden claims or acknowledgements change the answers? (H8)

Question (H8 of section 9 of docs/multi_user_memory_review.md, and the dispute
table of decision 7). ``disputes(viewers)`` shows disagreements as a table, and
``who_knows(memory, viewers)`` reports who stated, acknowledged or only had
access to a memory. Both must count only statements and acknowledgements, never
presence, and both must draw only on what the viewers can see: a hidden claim
must not add a row, change a status, open or close a dispute, or make someone a
stater. Do they?

Data. Scripted and synthetic, no LLM. Each pair is built from seed s. Members
alice, bob, carol and dana; dana is a workspace admin. ``design`` holds all
four; ``dm`` holds bob and dana, so alice and carol cannot see anything drawn
from a ``dm`` message. Keys k0 to k2, and a planted key ``kp``.

- Planted step (every pair): alice says a claim on ``kp`` from a ``design``
  message, and carol replies to that message. Nobody else writes ``kp``, and
  dana never writes or acknowledges in the visible script.
- Visible script ("s-e5-visible"), 5 to 10 steps after the planted one: with
  probability 0.7 a claim by alice or carol on a random key k0 to k2, from a
  ``design`` message, its text one of three values ("<key> is v0", "v1", "v2");
  otherwise alice or carol acknowledges (replied or reacted) a random earlier
  visible message.
- Hidden script ("s-e5-hidden"), 2 to 6 claims by bob or dana on a random key k0
  to k2 with one of the same three values, from a ``dm`` message, each inserted
  at a random position after the planted step; dana's claims carry admin
  authority. With probability 0.5 each is followed by bob or dana reacting to
  its own ``dm`` message.

Worlds. W: the visible script alone. H: with the hidden script. H2: H rebuilt.
V (positive control): as H, but the hidden claims come from ``design`` messages,
so alice and carol can see them.

Observations, for audiences [alice], [carol] and [alice, carol]: the dispute
tables (key, open, and per row speaker, text, status, standing) and, for every
visible claim, who_knows (stated, acknowledged, had access). Visible claims are
named by their text and their position in the visible script (never by their
position in a world's interleaved sequence); hidden ones show as "?".

Checks, per pair.
- Determinism (negative control): H2 equals H.
- Power (positive control): V differs from W.
- Acknowledgement counts (positive control): in W and H, for audience [alice],
  the planted claim's who_knows lists alice as stating it and carol, who
  replied, as acknowledging it.
- Presence does not count (negative control): in W and H, dana, who only ever
  belonged to ``design``, appears in the planted claim's "had access" and in
  neither "stated" nor "acknowledged".
- Noninterference: H equals W in every observation.

Decision rule, fixed before running:
  1. Determinism holds in every pair.
  2. Power: V differs from W in at least 50% of pairs.
  3. Acknowledgement counts in every pair.
  4. Presence does not count in every pair.
  5. Noninterference holds in every pair.
Verdict: "inconclusive (control <n> failed)" if rule 1, 2, 3 or 4 fails;
otherwise "disputes and who-knows pass" if rule 5 holds, else "they fail rule 5
(first failing seed <s>)".

Validation (positive control against the store), fixed before running: with
``--fault administrative`` the script replaces ``disputes`` and ``who_knows``
with versions that compute over every memory (still returning None for a
memory hidden from the viewers). Rule 5 must then fail in at least 25% of
pairs, or this experiment cannot detect the leak it is for.

Default: 500 pairs (seeds 0 to 499).

Run from the project root:

    python3 -I experiments/audience_disputes_who_knows.py
    python3 -I experiments/audience_disputes_who_knows.py --fault administrative \\
        --output experiments/results/audience_disputes_who_knows_fault
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
from dynmemory.store import SCHEMA_VERSION  # noqa: E402

MEMBERS = ("alice", "bob", "carol", "dana")
VISIBLE_WRITERS = ("alice", "carol")
AUDIENCES = (("alice",), ("carol",), ("alice", "carol"))
KEYS = ("k0", "k1", "k2")
VALUES = ("v0", "v1", "v2")
VISIBLE_STEPS = (5, 10)
CLAIM_SHARE = 0.7
HIDDEN_CLAIMS = (2, 6)
HIDDEN_ACK = 0.5
POWER_SHARE = 0.5
DEFAULT_TRIALS = 500
WORLDS = ("W", "H", "H2", "V")
CHECKS = ("determinism", "power", "acknowledgement", "presence", "noninterference")
RULES = {number: name for number, name in enumerate(CHECKS, start=1)}
CONTROL_RULES = (1, 2, 3, 4)


def make_pair(seed: int) -> dict:
    rng = random.Random("{}-e5-visible".format(seed))
    visible = [{"op": "planted"}]
    for _ in range(rng.randint(*VISIBLE_STEPS)):
        if rng.random() < CLAIM_SHARE or len(visible) == 1:
            key = rng.choice(KEYS)
            visible.append(
                {
                    "op": "claim",
                    "writer": rng.choice(VISIBLE_WRITERS),
                    "key": key,
                    "value": "{} is {}".format(key, rng.choice(VALUES)),
                }
            )
        else:
            visible.append(
                {
                    "op": "ack",
                    "member": rng.choice(VISIBLE_WRITERS),
                    "kind": rng.choice(("replied", "reacted")),
                    "pick": rng.random(),
                }
            )
    for index, step in enumerate(visible):
        step["index"] = index  # labels must not depend on how a world interleaves
    hidden_rng = random.Random("{}-e5-hidden".format(seed))
    hidden = []
    for _ in range(hidden_rng.randint(*HIDDEN_CLAIMS)):
        key = hidden_rng.choice(KEYS)
        hidden.append(
            {
                "op": "claim",
                "writer": hidden_rng.choice(("bob", "dana")),
                "key": key,
                "value": "{} is {}".format(key, hidden_rng.choice(VALUES)),
                "ack": hidden_rng.choice(("bob", "dana")) if hidden_rng.random() < HIDDEN_ACK else None,
                "position": hidden_rng.randint(1, len(visible)),
            }
        )
    return {"seed": seed, "visible": visible, "hidden": hidden}


def sequence(pair: dict, world: str) -> list:
    steps = [("visible", step) for step in pair["visible"]]
    if world == "W":
        return steps
    for claim in sorted(pair["hidden"], key=lambda item: -item["position"]):
        steps.insert(min(claim["position"], len(steps)), ("hidden", claim))
    return steps


def install_fault(name: str) -> None:
    """Make the store compute disputes and who-knows over every memory."""
    if name != "administrative":
        raise ValueError("unknown fault: {}".format(name))
    disputes, who_knows = MemoryStore.disputes, MemoryStore.who_knows

    def every_claim_disputes(store, *, viewers=None, channel=None, memory_key=None):
        return disputes(store, viewers=None, channel=channel, memory_key=memory_key)

    def every_claim_who_knows(store, memory_id, *, viewers=None):
        if store.get_memory(memory_id, viewers=viewers) is None:
            return None
        return who_knows(store, memory_id, viewers=None)

    MemoryStore.disputes = every_claim_disputes
    MemoryStore.who_knows = every_claim_who_knows


def observe(pair: dict, world: str) -> dict:
    source_for_hidden = "design" if world == "V" else "dm"
    with MemoryStore(":memory:") as store:
        for member in MEMBERS:
            store.join_channel("design", member)
        for member in ("bob", "dana"):
            store.join_channel("dm", member)
        store.add_admin("dana")
        names, messages = {}, []
        for script, step in sequence(pair, world):
            if step["op"] == "planted":
                writer = store.for_writer("alice")
                message = writer.add_message("kp is planted", channel="design")
                memory = writer.add_memory(
                    "kp is planted", MemoryType.FACT, channel="design",
                    memory_key="kp", source_message_ids=[message.id],
                )
                store.for_writer("carol").acknowledge(message.id, "replied")
                names[memory.id] = "planted"
                messages.append(message.id)
            elif step["op"] == "ack":
                target = messages[int(step["pick"] * len(messages))]
                store.for_writer(step["member"]).acknowledge(target, step["kind"])
            else:
                writer = store.for_writer(step["writer"])
                source = "design" if script == "visible" else source_for_hidden
                message = writer.add_message(step["value"], channel=source)
                memory = writer.add_memory(
                    step["value"], MemoryType.FACT, channel="design",
                    memory_key=step["key"], source_message_ids=[message.id],
                )
                if script == "visible":
                    names[memory.id] = "{} #{}".format(step["value"], step["index"])
                    messages.append(message.id)
                elif step.get("ack"):
                    store.for_writer(step["ack"]).acknowledge(message.id, "reacted")
        reads = {}
        for audience in AUDIENCES:
            label = "+".join(audience)
            reads[label] = {
                "disputes": [
                    (
                        dispute.memory_key,
                        dispute.open,
                        [
                            (row.speaker, row.says, row.status, row.standing,
                             names.get(row.memory_id, "?"))
                            for row in dispute.rows
                        ],
                    )
                    for dispute in store.disputes(viewers=list(audience))
                ],
                "who_knows": {
                    name: _knowledge(store.who_knows(memory_id, viewers=list(audience)))
                    for memory_id, name in sorted(names.items(), key=lambda item: item[1])
                },
            }
        return reads


def _knowledge(knowledge):
    if knowledge is None:
        return None
    return (knowledge.stated, knowledge.acknowledged, knowledge.had_access)


def evaluate(pair: dict, seen: dict) -> dict:
    w, h, h2, v = (seen[world] for world in WORLDS)
    planted = [world["alice"]["who_knows"]["planted"] for world in (w, h)]
    acknowledgement = all(
        knowledge is not None and knowledge[0] == ("alice",) and knowledge[1] == ("carol",)
        for knowledge in planted
    )
    presence = all(
        knowledge is not None and "dana" in knowledge[2]
        and "dana" not in knowledge[0] and "dana" not in knowledge[1]
        for knowledge in planted
    )
    differs = [
        "{} ({})".format(audience, part)
        for audience in h
        for part in h[audience]
        if h[audience][part] != w[audience][part]
    ]
    return {
        "seed": pair["seed"],
        "checks": {
            "determinism": h == h2,
            "power": v != w,
            "acknowledgement": acknowledgement,
            "presence": presence,
            "noninterference": not differs,
        },
        "differs": differs[:3],
        "disputes_in_w": sum(len(w[audience]["disputes"]) for audience in w),
    }


def run_trial(seed: int) -> dict:
    pair = make_pair(seed)
    return evaluate(pair, {world: observe(pair, world) for world in WORLDS})


def summarise(trials: list) -> dict:
    return {
        "checks": {
            name: {
                "held": sum(trial["checks"][name] for trial in trials),
                "not_held_seeds": [t["seed"] for t in trials if not t["checks"][name]],
            }
            for name in CHECKS
        },
        "pairs_with_a_dispute_in_w": sum(trial["disputes_in_w"] > 0 for trial in trials),
        "failures": [
            {"seed": trial["seed"], "differs": trial["differs"]}
            for trial in trials
            if not trial["checks"]["noninterference"]
        ][:10],
    }


def decide(summary: dict, trials: int) -> dict:
    checks = summary["checks"]
    holds = {
        1: checks["determinism"]["held"] == trials,
        2: checks["power"]["held"] >= POWER_SHARE * trials,
        3: checks["acknowledgement"]["held"] == trials,
        4: checks["presence"]["held"] == trials,
        5: checks["noninterference"]["held"] == trials,
    }
    failed = [number for number in CONTROL_RULES if not holds[number]]
    if failed:
        verdict = "inconclusive (control {} failed)".format(", ".join(map(str, failed)))
    elif holds[5]:
        verdict = "disputes and who-knows pass"
    else:
        verdict = "they fail rule 5 (first failing seed {})".format(
            min(checks["noninterference"]["not_held_seeds"])
        )
    decision = {"rule_{}_{}".format(number, name): holds[number] for number, name in RULES.items()}
    decision["verdict"] = verdict
    return decision


def _share(count: int, total: int) -> str:
    return "{} of {} ({:.1f}%)".format(count, total, 100.0 * count / total) if total else "-"


def render(report: dict) -> str:
    trials, summary, decision = report["trials"], report["summary"], report["decision"]
    checks = summary["checks"]
    lines = [
        "# Disputes and who-knows-what (H8)",
        "",
        "Synthetic, scripted data; no LLM. {} pairs (seeds 0 to {}).".format(trials, trials - 1),
        "Store under test: schema version {}{}.".format(
            report["store_schema_version"],
            "; injected fault: {}".format(report["fault"]) if report["fault"] else "",
        ),
        "See the docstring of `experiments/audience_disputes_who_knows.py` for the design",
        "and decision rule.",
        "",
        "| Rule | Check | Held in | Needed | Holds |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = (
        (1, "Determinism: H and H2 identical", "every pair"),
        (2, "Power: V (visible hidden claims) differs from W", "at least 50%"),
        (3, "A reply counts as knowing (planted claim)", "every pair"),
        (4, "Presence alone is only access (dana)", "every pair"),
        (5, "Noninterference: H equals W for every audience", "every pair"),
    )
    for number, text, needed in rows:
        name = RULES[number]
        lines.append("| {} | {} | {} | {} | {} |".format(
            number, text, _share(checks[name]["held"], trials), needed,
            decision["rule_{}_{}".format(number, name)],
        ))
    lines += [
        "",
        "**Verdict: {}**".format(decision["verdict"]),
        "",
        "Reported, not part of the verdict: pairs whose visible script alone produced at least"
        " one dispute: {}.".format(_share(summary["pairs_with_a_dispute_in_w"], trials)),
    ]
    if summary["failures"]:
        lines += ["", "First pairs where noninterference did not hold:", ""]
        lines += ["- seed {}: {}".format(item["seed"], "; ".join(item["differs"])) for item in summary["failures"]]
    lines += [
        "",
        "Limits: synthetic claims and acknowledgements written alongside the code, so a pass",
        "shows the mechanism, not how well acknowledgements capture knowing in practice. Text",
        "equality is exact; the embedding equivalence option is not exercised here.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--fault", choices=["administrative"])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "audience_disputes_who_knows"
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")
    if arguments.fault:
        install_fault(arguments.fault)
    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "store_schema_version": SCHEMA_VERSION,
        "fault": arguments.fault,
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
