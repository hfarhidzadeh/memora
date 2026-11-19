"""Retraction canaries: does retracting a message erase its text from the database? (H5)

Question (H5 of section 9 of docs/multi_user_memory_review.md). A memory can
list source messages. When a message is retracted, is every stored copy of its
text gone from the database, including the text of the memories derived from
it, and is nothing else touched? Review on 2026-11-18 found that the message
became a tombstone and the derived memories were hidden from every read, but
their rows kept the retracted text. This experiment measures that directly, by
scanning every value in every table.

Data. Scripted and synthetic, no LLM, no real text. Each trial is built from
seed s in its own file-backed database. Members alice and bob share the channel
``design``.

- Messages ("s-e4"): 4 to 8, each said by alice or bob. The text is "note <i>
  <canary> <3 filler words>", where the canary is a unique token, "cnr<s>x<i>".
- Derived memories: 3 to 8 keyed or unkeyed facts in ``design`` (key k0 to k2
  with probability 0.4), written by alice or bob. Each lists 1 or 2 distinct
  messages as its sources, and its text repeats the canary of every source and
  adds 2 filler words, as a derived summary would.
- Retractions: 1 or 2 messages that have at least one dependent memory, each
  retracted by its speaker.

Scan. Every value of every column of every table, read through SQLite, with
bytes decoded as Latin-1. A canary is "stored" if it occurs in any value. A
second, descriptive scan reads the raw bytes of the database file.

Checks, per trial.
- Determinism (negative control): the trial run twice gives the same result.
- Probe (positive control): before any retraction, the scan finds the canary of
  every message.
- Specificity: after the retractions, every message that was not retracted
  still has its text, and every memory with no retracted source still has the
  text it was written with.
- Erasure: after the retractions, the canary of no retracted message occurs in
  any stored value.

Reported, not part of the verdict: trials in which the raw bytes of the
database file still contain a retracted canary (SQLite may keep old bytes in
free space); how many memories were derived from retracted messages.

Decision rule, fixed before running:
  1. Determinism holds in every trial.
  2. The probe finds every canary in every trial.
  3. Specificity holds in every trial.
  4. Erasure holds in every trial.
Verdict: "inconclusive (control <n> failed)" if rule 1 or 2 fails; otherwise
"retraction erases" if rules 3 and 4 hold; otherwise "retraction does not erase
(rules <list> failed, first failing seed <s>)".

Validation (positive control against the store), fixed before running: the
script is run unchanged against the store of commit 2114e4e, before derived
text was cleared on retraction. It is judged able to detect the fault if rule 4
fails in at least 95% of trials there.

Runs. Run 1 (2026-11-19) was on the store that cleared derived text but did not
overwrite deleted bytes: rules 1 to 4 held in 500 of 500 trials, and the raw
file still held a retracted canary in 495 of 500 (reports in
``experiments/results/retraction_canary_run1_superseded/``). The store then
gained ``secure_delete`` (on by default; SQLite overwrites deleted and replaced
bytes). Nothing else changed: the script's data, scan, checks and decision
rule are those of run 1, and the report now also names the store's
``secure_delete`` setting. Expected before run 2 (a prediction, not a rule): the
raw file holds a retracted canary in few or none of the trials; some residue is
possible, for example in pages that were freed before a later overwrite.

Default: 500 trials (seeds 0 to 499).

Run from the project root:

    python3 -I experiments/audience_retraction_canary.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore, MemoryType  # noqa: E402
from dynmemory.store import SCHEMA_VERSION  # noqa: E402

MEMBERS = ("alice", "bob")
CHANNEL = "design"
FILLER = tuple("f{:02d}".format(index) for index in range(40))
MESSAGES = (4, 8)
MEMORIES = (3, 8)
KEY_SHARE = 0.4
KEYS = ("k0", "k1", "k2")
RETRACTIONS = (1, 2)
DEFAULT_TRIALS = 500
CHECKS = ("determinism", "probe", "specificity", "erasure")
RULES = {number: name for number, name in enumerate(CHECKS, start=1)}
CONTROL_RULES = (1, 2)
BUILT_RULES = (3, 4)


def make_trial(seed: int) -> dict:
    rng = random.Random("{}-e4".format(seed))
    messages = []
    for index in range(rng.randint(*MESSAGES)):
        canary = "cnr{}x{}".format(seed, index)
        words = " ".join(rng.choice(FILLER) for _ in range(3))
        messages.append(
            {
                "speaker": rng.choice(MEMBERS),
                "canary": canary,
                "text": "note {} {} {}".format(index, canary, words),
            }
        )
    memories = []
    for _ in range(rng.randint(*MEMORIES)):
        sources = rng.sample(range(len(messages)), rng.randint(1, min(2, len(messages))))
        text = " ".join(
            ["summary"]
            + [messages[index]["canary"] for index in sources]
            + [rng.choice(FILLER) for _ in range(2)]
        )
        memories.append(
            {
                "speaker": rng.choice(MEMBERS),
                "text": text,
                "key": rng.choice(KEYS) if rng.random() < KEY_SHARE else None,
                "sources": sources,
            }
        )
    depended_on = sorted({index for memory in memories for index in memory["sources"]})
    retract = sorted(rng.sample(depended_on, min(rng.randint(*RETRACTIONS), len(depended_on))))
    return {"seed": seed, "messages": messages, "memories": memories, "retract": retract}


def stored_values(store: MemoryStore) -> list:
    """Every text value of every column of every table."""
    values = []
    with store._transaction(write=False) as connection:
        tables = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        for table in tables:
            for row in connection.execute("SELECT * FROM {}".format(table)):
                for value in tuple(row):
                    if isinstance(value, str):
                        values.append(value)
                    elif isinstance(value, bytes):
                        values.append(value.decode("latin-1"))
    return values


def holds(values: list, text: str) -> bool:
    return any(text in value for value in values)


def run(trial: dict) -> dict:
    seed = trial["seed"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "e4.sqlite3"
        store = MemoryStore(path)
        for member in MEMBERS:
            store.join_channel(CHANNEL, member)
        message_ids, memory_ids = [], []
        for message in trial["messages"]:
            message_ids.append(
                store.for_writer(message["speaker"]).add_message(message["text"], channel=CHANNEL).id
            )
        for memory in trial["memories"]:
            written = store.for_writer(memory["speaker"]).add_memory(
                memory["text"],
                MemoryType.FACT,
                channel=CHANNEL,
                memory_key=memory["key"],
                source_message_ids=[message_ids[index] for index in memory["sources"]],
            )
            memory_ids.append(written.id)

        before = stored_values(store)
        canaries = [message["canary"] for message in trial["messages"]]
        probe = all(holds(before, canary) for canary in canaries)

        for index in trial["retract"]:
            store.retract_message(message_ids[index], actor_id=trial["messages"][index]["speaker"])
        after = stored_values(store)
        raw = path.read_bytes()

        retracted = set(trial["retract"])
        erased = not any(holds(after, canaries[index]) for index in retracted)
        kept_messages = all(
            store.get_message(message_ids[index]).content == trial["messages"][index]["text"]
            for index in range(len(message_ids))
            if index not in retracted
        )
        kept_memories = all(
            store.get_memory(memory_ids[number]).content == memory["text"]
            for number, memory in enumerate(trial["memories"])
            if not retracted & set(memory["sources"])
        )
        dependents = sum(bool(retracted & set(memory["sources"])) for memory in trial["memories"])
        store_text = {
            "queue": sorted(memory.id for memory in store.stale_memories()),
        }
        return {
            "probe": probe,
            "specificity": kept_messages and kept_memories,
            "erasure": erased,
            "raw_file_still_holds": any(canaries[i].encode() in raw for i in retracted),
            "dependents": dependents,
            "queue_size": len(store_text["queue"]),
            "seed": seed,
        }


def run_trial(seed: int) -> dict:
    trial = make_trial(seed)
    first, second = run(trial), run(trial)
    result = {key: value for key, value in first.items()}
    result["determinism"] = {k: v for k, v in first.items()} == {k: v for k, v in second.items()}
    return result


def summarise(trials: list) -> dict:
    checks = {
        name: {
            "held": sum(bool(trial[name]) for trial in trials),
            "not_held_seeds": [trial["seed"] for trial in trials if not trial[name]],
        }
        for name in CHECKS
    }
    return {
        "checks": checks,
        "raw_file_still_holds": sum(trial["raw_file_still_holds"] for trial in trials),
        "dependents": sum(trial["dependents"] for trial in trials),
        "queue_size": sum(trial["queue_size"] for trial in trials),
    }


def decide(summary: dict, trials: int) -> dict:
    checks = summary["checks"]
    holds_ = {number: checks[name]["held"] == trials for number, name in RULES.items()}
    failed_controls = [number for number in CONTROL_RULES if not holds_[number]]
    failed_rules = [number for number in BUILT_RULES if not holds_[number]]
    if failed_controls:
        verdict = "inconclusive (control {} failed)".format(
            ", ".join(str(number) for number in failed_controls)
        )
    elif not failed_rules:
        verdict = "retraction erases"
    else:
        seeds = [
            seed for number in failed_rules for seed in checks[RULES[number]]["not_held_seeds"]
        ]
        verdict = "retraction does not erase (rules {} failed, first failing seed {})".format(
            ", ".join(str(number) for number in failed_rules), min(seeds)
        )
    decision = {"rule_{}_{}".format(number, name): holds_[number] for number, name in RULES.items()}
    decision["verdict"] = verdict
    return decision


def _share(count: int, total: int) -> str:
    return "-" if not total else "{} of {} ({:.1f}%)".format(count, total, 100.0 * count / total)


def render(report: dict) -> str:
    trials, summary, decision = report["trials"], report["summary"], report["decision"]
    checks = summary["checks"]
    lines = [
        "# Retraction canaries (H5)",
        "",
        "Synthetic, scripted data; no LLM. {} trials (seeds 0 to {}).".format(trials, trials - 1),
        "Store under test: schema version {}; secure_delete: {}.".format(
            report["store_schema_version"], report["store_secure_delete"]
        ),
        "See the docstring of `experiments/audience_retraction_canary.py` for the design",
        "and decision rule.",
        "",
        "| Rule | Check | Held in | Needed | Holds |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = (
        (1, "Determinism: the same seed gives the same result"),
        (2, "Probe: before retraction the scan finds every canary"),
        (3, "Specificity: only memories derived from retracted messages change"),
        (4, "Erasure: no retracted canary in any stored value"),
    )
    for number, text in rows:
        name = RULES[number]
        lines.append(
            "| {} | {} | {} | every trial | {} |".format(
                number,
                text,
                _share(checks[name]["held"], trials),
                decision["rule_{}_{}".format(number, name)],
            )
        )
    lines += [
        "",
        "**Verdict: {}**".format(decision["verdict"]),
        "",
        "Reported, not part of the verdict:",
        "",
        "- Memories derived from a retracted message: {} (all trials); stale memories in the"
        " rebuild queue: {}.".format(summary["dependents"], summary["queue_size"]),
        "- Trials in which the raw bytes of the database file still contain a retracted"
        " canary: {}. The scan of stored values reads live rows; SQLite can keep old bytes"
        " in free space and backups, which this experiment does not claim to erase.".format(
            _share(summary["raw_file_still_holds"], trials)
        ),
        "",
        "Limits: synthetic text written alongside the code, so a pass shows the mechanism, not",
        "how often derived text repeats a source word in practice. A derived memory is linked",
        "to its sources only through the source list the caller supplies; text copied into a",
        "memory that does not list the message is not found by the store and not tested here.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "audience_retraction_canary"
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")
    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "store_schema_version": SCHEMA_VERSION,
        "store_secure_delete": getattr(
            MemoryStore(":memory:"), "secure_delete", "not available"
        ),
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
