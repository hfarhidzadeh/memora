"""Noninterference of reads with viewers: does anything hidden change what viewers read?

Question (E2; every path of section 8 of docs/multi_user_memory_review.md at
once, including keys). Two databases, A and B, agree on everything the viewers
may see and differ only in what they may not see: hidden memories, hidden keys
and versions, links touching a hidden memory, and other people's memberships.
Do all reads with ``viewers`` (``search``, ``get_memory``, ``history`` and
``list_memories``) return identical results on A and B (noninterference)? Do
they also equal the reads, without ``viewers``, of a database R that never held
the hidden part (exactness)? Filtering after ranking is run as a positive
control, and its differences between A and B are attributed to fields.

Data. Scripted and synthetic, no LLM, no real text. One trial is one pair,
built from seed s:

- People and channels. ana and ben may be viewers; cy, dee and eli are others.
  Channels: both (ana, ben), ana-only (ana), ben-only (ben), others (cy, dee,
  eli). In A and B, each of cy, dee and eli also joins each of both, ana-only
  and ben-only with probability 0.5, drawn from that world's hidden stream, so
  these memberships differ between A and B.
- Viewers ("s-pair"): ["ana"] (visible channels both and ana-only) or
  ["ana", "ben"] (visible channel both), 50/50. With probability 0.5 the
  viewers join ``others`` before the first operation and leave it after the
  last, before any read, so they were members while every hidden memory there
  was written. This churn is the same in A, A2, A+, B and R.
- Store options, the same in every world of a pair ("s-pair"): link carry-over
  keep, copy or topic; no embedder or ``NGramHashEmbedder()``, 50/50.
- Vocabulary: 60 pseudo-words w00-w59 (small, so hidden content moves the
  statistics). Kinds: all seven MemoryType values. Keys: k0-k5.
- Visible script ("s-visible"), 20 to 60 operations:
  - add (70%): a random visible channel and kind, 3 to 10 words, a key with
    probability 0.3, and as speaker a random default member of the channel
    (ana or ben). Others' memberships belong to the hidden part, and the
    visible script must not depend on them.
  - relate (30%): a random ordered pair of memories that are active according
    to the script's own model, with a random relation from all five. A relate
    drawn while fewer than two memories are active becomes an add.
  - The model tracks which visible memories exist and are active, including
    key rewrites within a channel, supersessions and the key hand-over (a
    source without a key takes over the target's key), and predicts each
    operation's outcome: a ``supersedes`` is refused across channels or when
    the source is the older memory, and everything else is accepted. A refused
    ``supersedes`` raises ValueError, which is caught and recorded.
- Hidden script ("s-hidden-A" for A, "s-hidden-B" for B), 10 to 60 operations:
  - add (55%): a random hidden channel, or no channel with probability 0.1,
    otherwise as above with the same key names; the speaker is a random member
    of the channel as the hidden script tracks memberships (anyone if it has
    none, or for no channel).
  - relate (35%): one end is a random hidden memory written so far, archived
    or not; the other is another hidden memory with probability 0.5 (when one
    exists) or a random visible label; the direction is random. The relation
    is random from all five for two hidden memories in the same channel and
    from the four others otherwise. 5% of relates are instead a deliberate
    cross-channel ``supersedes`` (the other end is a visible label or a hidden
    memory in another channel), which must raise ValueError; they are counted.
    A relate to a memory archived at that moment raises ValueError, which is
    caught and counted. A relate drawn before any hidden memory exists
    becomes an add.
  - membership (10%): a random other person joins a random channel, or leaves
    it if already a member.
  - The hidden script is generated before interleaving, so a relate may name a
    visible label that is not written yet, or is archived, when it runs. It is
    then skipped and the skip counted (decided from the visible model's state
    at that moment, never from the store).
- Interleaving ("s-interleave-A" or "-B"): the visible and hidden scripts are
  merged uniformly at random, each keeping its own order.
- Reads ("s-reads"), the same in every world: 6 searches (1 to 3 distinct
  vocabulary words; kind None with probability 0.7, else a random kind; limit
  from {1, 3, 10, 1000}); ``get_memory`` and ``history`` for every label of
  the visible script, including archived versions; and ``list_memories`` with
  ``include_archived`` False and True, ``limit=1000``. Every read uses
  ``record_access=False`` where the method has it.

Worlds. A and B: the visible script merged with that world's hidden script. A2:
A rebuilt from the same seeds. A+: A followed by one extra visible add (its
content is the first search's words, its kind that search's kind filter or
``fact``, its channel a random visible channel from "s-extra", no key). R: the
visible script alone, with ana's and ben's default memberships and the
viewers' churn; nobody else's memberships.

Labels. Visible memories are v<i> in the order the visible script adds them;
hidden ones are hA<i> in A (and A2, A+) and hB<i> in B, so any mention of a
hidden memory differs between A and B. Results are compared by label and never
depend on IDs, timestamps or wall-clock time; an ID with no label would be
shown as "?" + id.

Arms.
- after-naive: ``search(q, kind=kind, limit=limit, record_access=False)``,
  then results the viewers cannot see are dropped. Links are left as they are.
- after-careful: the same search with ``limit=BIG`` (at least the number of
  memories in the store), drop results the viewers cannot see, keep only
  links with both ends visible, then truncate to ``limit``.
  Both after arms decide visibility from the experiment's own record of each
  memory's channel and of the memberships, never from the store's ``viewers``
  code. A memory is visible if every viewer is currently in its channel; one
  without a channel never is.
- built: every read with ``viewers=...``.
- plain(R): the same reads without ``viewers`` on R. This is filtering before
  ranking, emulated, and the reference for exactness.
The built arm is compared on all reads; the after arms on the 6 searches only.

Canonical forms. A memory is (label, kind, content, memory_key, channel,
speaker, confidence, importance, archived, superseded_by label or None). A
search is, per result in order, (label, kind, content, memory_key, channel,
speaker, score, matched_by, lexical_score, vector_score, coverage, links), with
links sorted as (label(source), label(target), relation). Floats are compared
exactly. Searches and histories are compared in the order returned; the two
lists are compared sorted by label (see ``_reads`` for why).

Controls and checks, per pair.
- Determinism (negative control): A2 is A rebuilt from the same seeds. Every
  arm, and every operation's outcome, must be identical in A and A2.
- Sensitivity (positive control for the comparison): built(A) and built(A+)
  must differ. How often the first search alone differs is also reported.
- Power (positive control): after-naive(A) and after-naive(B) differ.
- Noninterference: built(A) equals built(B), and every visible operation went
  as the visible model predicted in A and in B. An outcome that differs from
  the prediction, or between worlds, means hidden activity changed what
  happened to visible memories.
- Exactness: built(A) equals plain(R).
- Soundness: no built read in A or B returns or links to (including through
  ``superseded_by``) a hidden label; ``get_memory(hidden_id, viewers=...)`` is
  None, and ``history(hidden_id, viewers=...)`` is empty, for every hidden
  memory in A and B.

Attribution (descriptive). For after-naive and after-careful, the share of
pairs in which, in at least one of the 6 searches, A and B differ in: the
result set (labels shown); the order (relative order of the labels both show);
the count; and, for a label both show, score, lexical_score, coverage,
vector_score, matched_by or links. Also broken down by link carry-over and by
embedder.

Decision rule, fixed before running:
  1. Determinism: zero differences between A and A2 in every arm (and in every
     operation's outcome) and every pair.
  2. Sensitivity: built(A) differs from built(A+) in every pair.
  3. Power: after-naive differs between A and B in at least 50% of pairs.
  4. Noninterference: built(A) equals built(B) in every pair, and every visible
     operation went as predicted in A and in B.
  5. Exactness: built(A) equals plain(R) in every pair.
  6. Soundness holds in every pair.
  Verdicts: "inconclusive (control <n> failed)" if rule 1, 2 or 3 fails (all
  failing controls are named); otherwise "the built read path passes" if rules
  4 to 6 hold; otherwise "the built read path fails rules <list> (first failing
  seed <s>)".

Readings chosen where the specification left room, all before any run: the
speakers of visible adds and the timing of the viewers' churn (above); others'
extra memberships drawn from the hidden stream; R keeps ana's and ben's default
memberships whatever the viewer set; A+ is read with A's read plan, so its
extra memory shows only through searches and lists; operation outcomes count
under rule 1 as they do under rule 4; soundness also asks ``history`` of each
hidden ID; outcomes are recorded with the store's reason for a refusal.

Default: 500 pairs (seeds 0 to 499).

Run from the project root:

    python3 -I experiments/audience_noninterference.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import (  # noqa: E402
    MemoryStore,
    MemoryType,
    NGramHashEmbedder,
    Relationship,
)

VOCABULARY = tuple("w{:02d}".format(index) for index in range(60))
WORDS_PER_MEMORY = (3, 10)
KINDS = tuple(kind.value for kind in MemoryType)
KEYS = tuple("k{}".format(index) for index in range(6))
KEY_SHARE = 0.3
RELATIONS = tuple(relation.value for relation in Relationship)
SUPERSEDES = Relationship.SUPERSEDES.value
PLAIN_RELATIONS = tuple(relation for relation in RELATIONS if relation != SUPERSEDES)
POTENTIAL_VIEWERS = ("ana", "ben")
OTHERS = ("cy", "dee", "eli")
PEOPLE = POTENTIAL_VIEWERS + OTHERS
MEMBERS = {
    "both": ("ana", "ben"),
    "ana-only": ("ana",),
    "ben-only": ("ben",),
    "others": OTHERS,
}
CHANNELS = tuple(MEMBERS)
OPEN_TO_OTHERS = ("both", "ana-only", "ben-only")
OTHERS_JOIN = 0.5
CHURN_CHANNEL = "others"
CHURN = 0.5
VIEWER_SETS = (("ana",), ("ana", "ben"))
CARRYOVERS = ("keep", "copy", "topic")
EMBEDDERS = ("none", "n-gram")
VISIBLE_OPERATIONS = (20, 60)
VISIBLE_ADD = 0.7
HIDDEN_OPERATIONS = (10, 60)
HIDDEN_ADD = 0.55
HIDDEN_RELATE = 0.35
NO_CHANNEL = 0.1
CROSS_SUPERSEDES = 0.05
HIDDEN_PARTNER = 0.5
SEARCHES = 6
QUERY_WORDS = (1, 3)
KIND_FILTER = 0.3
LIMITS = (1, 3, 10, 1000)
LIST_LIMIT = 1000
BIG = 10000
POWER_SHARE = 0.5
DEFAULT_TRIALS = 500
WORLDS = ("A", "A2", "B", "A+", "R")
AFTER_ARMS = ("after-naive", "after-careful")
WORLD_ARMS = {
    "A": AFTER_ARMS + ("built",),
    "A2": AFTER_ARMS + ("built",),
    "B": AFTER_ARMS + ("built",),
    "A+": ("built",),
    "R": ("plain",),
}
SEARCH_FIELDS = (
    "label",
    "kind",
    "content",
    "memory_key",
    "channel",
    "speaker",
    "score",
    "matched_by",
    "lexical_score",
    "vector_score",
    "coverage",
    "links",
)
MEMORY_FIELDS = (
    "label",
    "kind",
    "content",
    "memory_key",
    "channel",
    "speaker",
    "confidence",
    "importance",
    "archived",
    "superseded_by",
)
SHARED_FIELDS = ("score", "lexical_score", "coverage", "vector_score", "matched_by", "links")
ATTRIBUTION_FIELDS = ("result set", "order", "count") + SHARED_FIELDS
CHECKS = (
    "determinism",
    "sensitivity",
    "power",
    "noninterference",
    "exactness",
    "soundness",
)
RULES = {number: name for number, name in enumerate(CHECKS, start=1)}
CONTROL_RULES = (1, 2, 3)
BUILT_RULES = (4, 5, 6)
GROUPS = (
    ("all", None, None),
    ("keep", "carryover", "keep"),
    ("copy", "carryover", "copy"),
    ("topic", "carryover", "topic"),
    ("no embedder", "embedder", "none"),
    ("n-gram embedder", "embedder", "n-gram"),
)
REFUSALS = (
    ("exist and be active", "refused: not active"),
    ("same channel", "refused: other channel"),
    ("older one", "refused: older"),
)
DETAILS_KEPT = 10


def visible_channels(viewers) -> tuple:
    """The channels every viewer is in at read time (their default memberships)."""
    return tuple(
        channel
        for channel in CHANNELS
        if all(viewer in MEMBERS[channel] for viewer in viewers)
    )


def hidden_channels(viewers) -> tuple:
    shown = visible_channels(viewers)
    return tuple(channel for channel in CHANNELS if channel not in shown)


class Versions:
    """Which memories of one script exist and are active, as a correct store keeps them.

    It follows key rewrites within a channel (memories without a channel share
    one namespace), supersessions and the key hand-over, and predicts the
    outcome of each relate in the store's own order of checks.
    """

    def __init__(self) -> None:
        self.memories = {}
        self.holders = {}

    def active(self) -> list:
        return [label for label, memory in self.memories.items() if memory["active"]]

    def add(self, operation: dict) -> None:
        channel, key = operation["channel"], operation["key"]
        if key is not None:
            previous = self.holders.get((channel, key))
            if previous is not None:
                self.memories[previous]["active"] = False
            self.holders[(channel, key)] = operation["label"]
        self.memories[operation["label"]] = {
            "channel": channel,
            "key": key,
            "active": True,
            "order": len(self.memories),
        }

    def relate(self, source: str, target: str, relation: str) -> str:
        """Predict ``relate(source, target, relation)`` and apply it to the model."""
        newer, older = self.memories.get(source), self.memories.get(target)
        if newer is None or older is None or not (newer["active"] and older["active"]):
            return "refused: not active"
        if relation != SUPERSEDES:
            return "ok"
        if newer["channel"] != older["channel"]:
            return "refused: other channel"
        if newer["order"] < older["order"]:
            return "refused: older"
        older["active"] = False
        if older["key"] is not None:
            if newer["key"] is None:
                newer["key"] = older["key"]
                self.holders[(older["channel"], older["key"])] = source
            else:
                del self.holders[(older["channel"], older["key"])]
        return "ok"


def _add(rng: random.Random, label: str, channel, speakers) -> dict:
    kind = rng.choice(KINDS)
    words = rng.randint(*WORDS_PER_MEMORY)
    content = " ".join(rng.choice(VOCABULARY) for _ in range(words))
    key = rng.choice(KEYS) if rng.random() < KEY_SHARE else None
    return {
        "op": "add",
        "label": label,
        "channel": channel,
        "kind": kind,
        "content": content,
        "key": key,
        "speaker": rng.choice(speakers),
        "expect": "ok",
    }


def visible_script(seed: int, viewers) -> list:
    """The operations the viewers may see, each with its predicted outcome."""
    rng = random.Random("{}-visible".format(seed))
    channels = visible_channels(viewers)
    model = Versions()
    script = []
    for _ in range(rng.randint(*VISIBLE_OPERATIONS)):
        active = model.active()
        if rng.random() < VISIBLE_ADD or len(active) < 2:
            channel = rng.choice(channels)
            operation = _add(rng, "v{}".format(len(model.memories)), channel, MEMBERS[channel])
            model.add(operation)
        else:
            source, target = rng.sample(active, 2)
            relation = rng.choice(RELATIONS)
            operation = {
                "op": "relate",
                "source": source,
                "target": target,
                "relation": relation,
                "expect": model.relate(source, target, relation),
            }
        script.append(operation)
    return script


def _hidden_relate(rng: random.Random, model: Versions, visible_labels: list) -> dict:
    hidden = list(model.memories)
    end = rng.choice(hidden)
    channel = model.memories[end]["channel"]
    deliberate = rng.random() < CROSS_SUPERSEDES
    if deliberate:
        elsewhere = [
            label for label in hidden if model.memories[label]["channel"] != channel
        ]
        other = rng.choice(elsewhere + list(visible_labels))
        relation = SUPERSEDES
    else:
        partners = [label for label in hidden if label != end]
        if partners and rng.random() < HIDDEN_PARTNER:
            other = rng.choice(partners)
        else:
            other = rng.choice(visible_labels)
        same_channel = (
            other in model.memories and model.memories[other]["channel"] == channel
        )
        relation = rng.choice(RELATIONS if same_channel else PLAIN_RELATIONS)
    source, target = (end, other) if rng.random() < 0.5 else (other, end)
    operation = {
        "op": "relate",
        "source": source,
        "target": target,
        "relation": relation,
        "deliberate": deliberate,
    }
    if other in model.memories:
        operation["expect"] = model.relate(source, target, relation)
    else:
        # The visible end's state is known only when the operation runs.
        operation["visible_end"] = other
        if not model.memories[end]["active"]:
            operation["expect"] = "refused: not active"
        elif relation == SUPERSEDES:
            operation["expect"] = "refused: other channel"
        else:
            operation["expect"] = "ok"
    return operation


def hidden_script(seed: int, world: str, viewers, churn: bool, visible_labels) -> dict:
    """Other people's extra memberships and the hidden operations of world A or B."""
    rng = random.Random("{}-hidden-{}".format(seed, world))
    channels = hidden_channels(viewers)
    members = {channel: set(people) for channel, people in MEMBERS.items()}
    if churn:
        members[CHURN_CHANNEL].update(viewers)
    joins = []
    for person in OTHERS:
        for channel in OPEN_TO_OTHERS:
            if rng.random() < OTHERS_JOIN:
                joins.append({"person": person, "channel": channel})
                members[channel].add(person)
    model = Versions()
    script = []
    for _ in range(rng.randint(*HIDDEN_OPERATIONS)):
        draw = rng.random()
        if draw >= HIDDEN_ADD + HIDDEN_RELATE:
            person = rng.choice(OTHERS)
            channel = rng.choice(CHANNELS)
            action = "leave" if person in members[channel] else "join"
            if action == "join":
                members[channel].add(person)
            else:
                members[channel].discard(person)
            script.append(
                {
                    "op": "member",
                    "action": action,
                    "person": person,
                    "channel": channel,
                    "expect": "ok",
                }
            )
        elif draw >= HIDDEN_ADD and model.memories:
            script.append(_hidden_relate(rng, model, visible_labels))
        else:
            channel = None if rng.random() < NO_CHANNEL else rng.choice(channels)
            speakers = sorted(members[channel]) if channel is not None else []
            operation = _add(
                rng,
                "h{}{}".format(world, len(model.memories)),
                channel,
                speakers or PEOPLE,
            )
            model.add(operation)
            script.append(operation)
    return {"joins": joins, "operations": script}


def interleave(seed: int, world: str, visible_count: int, hidden_count: int) -> list:
    """A uniformly random merge that keeps each script's own order."""
    rng = random.Random("{}-interleave-{}".format(seed, world))
    total = visible_count + hidden_count
    hidden_slots = set(rng.sample(range(total), hidden_count))
    return ["hidden" if slot in hidden_slots else "visible" for slot in range(total)]


def read_plan(seed: int, labels) -> dict:
    rng = random.Random("{}-reads".format(seed))
    searches = []
    for _ in range(SEARCHES):
        words = rng.sample(VOCABULARY, rng.randint(*QUERY_WORDS))
        kind = rng.choice(KINDS) if rng.random() < KIND_FILTER else None
        searches.append({"query": " ".join(words), "kind": kind, "limit": rng.choice(LIMITS)})
    return {"searches": searches, "labels": list(labels)}


def extra_add(seed: int, viewers, first_search: dict, index: int) -> dict:
    """The visible add that turns A into A+."""
    rng = random.Random("{}-extra".format(seed))
    channel = rng.choice(visible_channels(viewers))
    return {
        "op": "add",
        "label": "v{}".format(index),
        "channel": channel,
        "kind": first_search["kind"] or MemoryType.FACT.value,
        "content": first_search["query"],
        "key": None,
        "speaker": rng.choice(MEMBERS[channel]),
        "expect": "ok",
    }


def make_pair(seed: int) -> dict:
    """Everything about a pair that does not involve a store."""
    rng = random.Random("{}-pair".format(seed))
    viewers = list(rng.choice(VIEWER_SETS))
    churn = rng.random() < CHURN
    carryover = rng.choice(CARRYOVERS)
    embedder = rng.choice(EMBEDDERS)
    visible = visible_script(seed, viewers)
    labels = [operation["label"] for operation in visible if operation["op"] == "add"]
    hidden = {
        world: hidden_script(seed, world, viewers, churn, labels) for world in ("A", "B")
    }
    order = {
        world: interleave(seed, world, len(visible), len(hidden[world]["operations"]))
        for world in ("A", "B")
    }
    reads = read_plan(seed, labels)
    return {
        "seed": seed,
        "viewers": viewers,
        "churn": churn,
        "carryover": carryover,
        "embedder": embedder,
        "visible": visible,
        "hidden": hidden,
        "order": order,
        "reads": reads,
        "extra": extra_add(seed, viewers, reads["searches"][0], len(labels)),
    }


def _part(world: str) -> str:
    return "B" if world == "B" else "A"


def world_sequence(pair: dict, world: str) -> list:
    """(script, operation) pairs in the order a world runs them."""
    if world == "R":
        return [("visible", operation) for operation in pair["visible"]]
    part = _part(world)
    streams = {
        "visible": iter(pair["visible"]),
        "hidden": iter(pair["hidden"][part]["operations"]),
    }
    sequence = [(source, next(streams[source])) for source in pair["order"][part]]
    if world == "A+":
        sequence.append(("visible", pair["extra"]))
    return sequence


def open_store(pair: dict) -> MemoryStore:
    embedder = NGramHashEmbedder() if pair["embedder"] == "n-gram" else None
    return MemoryStore(":memory:", embedder=embedder, link_carryover=pair["carryover"])


def _refusal(error: ValueError) -> str:
    message = str(error)
    for fragment, outcome in REFUSALS:
        if fragment in message:
            return outcome
    return "refused: {}".format(message)


def _apply(store: MemoryStore, record: dict, operation: dict) -> str:
    try:
        if operation["op"] == "add":
            memory = store.add_memory(
                operation["content"],
                operation["kind"],
                memory_key=operation["key"],
                channel=operation["channel"],
                speaker=operation["speaker"],
            )
            record["ids"][operation["label"]] = memory.id
            record["labels"][memory.id] = operation["label"]
            record["channels"][operation["label"]] = operation["channel"]
        elif operation["op"] == "member":
            channel, person = operation["channel"], operation["person"]
            if operation["action"] == "join":
                store.join_channel(channel, person)
                record["members"][channel].add(person)
            else:
                store.leave_channel(channel, person)
                record["members"][channel].discard(person)
        else:
            store.relate(
                record["ids"][operation["source"]],
                record["ids"][operation["target"]],
                operation["relation"],
            )
    except ValueError as error:
        return _refusal(error)
    return "ok"


def build_world(store: MemoryStore, pair: dict, world: str) -> dict:
    """Run one world's memberships and operations; return the experiment's record.

    The record holds the labels, each memory's channel and the memberships as
    the experiment wrote them, and every operation's outcome.
    """
    viewers = pair["viewers"]
    record = {
        "ids": {},
        "labels": {},
        "channels": {},
        "members": {channel: set() for channel in CHANNELS},
        "visible": [],
        "hidden": [],
    }
    memberships = [
        (channel, member)
        for channel, people in MEMBERS.items()
        for member in people
        if world != "R" or member in POTENTIAL_VIEWERS
    ]
    if world != "R":
        memberships += [
            (join["channel"], join["person"]) for join in pair["hidden"][_part(world)]["joins"]
        ]
    if pair["churn"]:
        memberships += [(CHURN_CHANNEL, viewer) for viewer in viewers]
    for channel, member in memberships:
        store.join_channel(channel, member)
        record["members"][channel].add(member)

    shown = Versions()  # the visible model, replayed as the visible script runs
    for source, operation in world_sequence(pair, world):
        if source == "visible":
            record["visible"].append(_apply(store, record, operation))
            if operation["op"] == "add":
                shown.add(operation)
            else:
                shown.relate(operation["source"], operation["target"], operation["relation"])
            continue
        expect, outcome = operation["expect"], None
        end = operation.get("visible_end")
        if end is not None and end not in shown.memories:
            expect = outcome = "skipped: not written yet"
        elif end is not None and not shown.memories[end]["active"]:
            expect = outcome = "skipped: archived"
        if outcome is None:
            outcome = _apply(store, record, operation)
        record["hidden"].append(
            {
                "op": operation["op"],
                "deliberate": operation.get("deliberate", False),
                "expect": expect,
                "outcome": outcome,
            }
        )

    if pair["churn"]:
        for viewer in viewers:
            store.leave_channel(CHURN_CHANNEL, viewer)
            record["members"][CHURN_CHANNEL].discard(viewer)
    return record


def _label(record: dict, memory_id: str) -> str:
    return record["labels"].get(memory_id, "?" + memory_id)


def _memory_row(record: dict, memory) -> tuple:
    if memory is None:
        return None
    return (
        _label(record, memory.id),
        memory.kind.value,
        memory.content,
        memory.memory_key,
        memory.channel,
        memory.speaker,
        memory.confidence,
        memory.importance,
        memory.archived_at is not None,
        None if memory.superseded_by is None else _label(record, memory.superseded_by),
    )


def _search_row(record: dict, result, links) -> tuple:
    memory = result.memory
    return (
        _label(record, memory.id),
        memory.kind.value,
        memory.content,
        memory.memory_key,
        memory.channel,
        memory.speaker,
        result.score,
        result.matched_by,
        result.lexical_score,
        result.vector_score,
        result.coverage,
        tuple(
            sorted(
                (
                    _label(record, link.source_id),
                    _label(record, link.target_id),
                    link.relation.value,
                )
                for link in links
            )
        ),
    )


def _visibility(record: dict, viewers):
    """Whether the viewers may see a memory, from the experiment's own record.

    The after arms use this; it never asks the store's ``viewers`` code.
    """
    channels = {
        channel
        for channel, members in record["members"].items()
        if all(viewer in members for viewer in viewers)
    }

    def visible(memory_id: str) -> bool:
        channel = record["channels"].get(record["labels"].get(memory_id))
        return channel is not None and channel in channels

    return visible


def _after_naive(store: MemoryStore, record: dict, pair: dict) -> dict:
    visible = _visibility(record, pair["viewers"])
    return {
        "search": [
            [
                _search_row(record, result, result.links)
                for result in store.search(
                    search["query"],
                    kind=search["kind"],
                    limit=search["limit"],
                    record_access=False,
                )
                if visible(result.memory.id)
            ]
            for search in pair["reads"]["searches"]
        ]
    }


def _after_careful(store: MemoryStore, record: dict, pair: dict) -> dict:
    visible = _visibility(record, pair["viewers"])
    big = max(BIG, len(record["ids"]))
    searches = []
    for search in pair["reads"]["searches"]:
        shown = [
            _search_row(
                record,
                result,
                [
                    link
                    for link in result.links
                    if visible(link.source_id) and visible(link.target_id)
                ],
            )
            for result in store.search(
                search["query"], kind=search["kind"], limit=big, record_access=False
            )
            if visible(result.memory.id)
        ]
        searches.append(shown[: search["limit"]])
    return {"search": searches}


def _reads(store: MemoryStore, record: dict, pair: dict, viewers) -> dict:
    """Every read of the plan, with ``viewers`` (built) or without (plain)."""
    plan = pair["reads"]

    def listed(include_archived: bool) -> list:
        # list_memories orders by updated_at and then by ID. A key rewrite gives
        # the old and the new version the same updated_at, so their order falls
        # back to random UUIDs and would differ between two identical worlds.
        # The lists are therefore compared sorted by label, not as returned.
        memories = store.list_memories(
            include_archived=include_archived, limit=LIST_LIMIT, viewers=viewers
        )
        rows = [_memory_row(record, memory) for memory in memories]
        return sorted(rows, key=lambda row: row[0])

    return {
        "search": [
            [
                _search_row(record, result, result.links)
                for result in store.search(
                    search["query"],
                    kind=search["kind"],
                    limit=search["limit"],
                    record_access=False,
                    viewers=viewers,
                )
            ]
            for search in plan["searches"]
        ],
        "get": {
            label: _memory_row(
                record, store.get_memory(record["ids"][label], viewers=viewers)
            )
            for label in plan["labels"]
        },
        "history": {
            label: [
                _memory_row(record, memory)
                for memory in store.history(record["ids"][label], viewers=viewers)
            ]
            for label in plan["labels"]
        },
        "list active": listed(False),
        "list all": listed(True),
    }


def _built(store: MemoryStore, record: dict, pair: dict) -> dict:
    return _reads(store, record, pair, pair["viewers"])


def _plain(store: MemoryStore, record: dict, pair: dict) -> dict:
    return _reads(store, record, pair, None)


READERS = {
    "after-naive": _after_naive,
    "after-careful": _after_careful,
    "built": _built,
    "plain": _plain,
}


def _hidden_returned(store: MemoryStore, record: dict, viewers) -> list:
    """Hidden labels that ``get_memory`` or ``history`` with viewers still returns."""
    return [
        label
        for label, memory_id in sorted(record["ids"].items())
        if not label.startswith("v")
        and (
            store.get_memory(memory_id, viewers=viewers) is not None
            or store.history(memory_id, viewers=viewers)
        )
    ]


def observe(pair: dict, world: str) -> dict:
    """Build one world in a fresh store and run its arms' reads."""
    with open_store(pair) as store:
        record = build_world(store, pair, world)
        observation = {
            "labels": sorted(record["ids"]),
            "visible outcomes": record["visible"],
            "hidden outcomes": record["hidden"],
            "arms": {arm: READERS[arm](store, record, pair) for arm in WORLD_ARMS[world]},
        }
        if world in ("A", "B"):
            observation["hidden returned"] = _hidden_returned(
                store, record, pair["viewers"]
            )
        return observation


def _rows_difference(left: list, right: list, fields: tuple) -> str:
    for index, (one, other) in enumerate(zip(left, right)):
        if one == other:
            continue
        if one is None or other is None:
            return "row {}: {} against {}".format(
                index,
                "nothing" if one is None else "a memory",
                "nothing" if other is None else "a memory",
            )
        field = next(name for name, x, y in zip(fields, one, other) if x != y)
        return "row {} differs in {}".format(index, field)
    return "{} rows against {}".format(len(left), len(right))


def first_difference(left: dict, right: dict):
    """Where two observations of the same reads first differ, or None.

    Names the read, row and field only, never values, so a trial's record
    never contains IDs.
    """
    for name in left:
        if name not in right:
            return "{} is missing".format(name)
        one, other = left[name], right[name]
        if one == other:
            continue
        if name == "search":
            for index, (rows, others) in enumerate(zip(one, other)):
                if rows != others:
                    return "search {}: {}".format(
                        index, _rows_difference(rows, others, SEARCH_FIELDS)
                    )
        elif name in ("get", "history"):
            for label in one:
                if label not in other:
                    return "{} {} is missing".format(name, label)
                if one[label] != other[label]:
                    rows, others = one[label], other[label]
                    if name == "get":
                        rows, others = [rows], [others]
                    return "{} {}: {}".format(
                        name, label, _rows_difference(rows, others, MEMORY_FIELDS)
                    )
            return "{}: different labels".format(name)
        return "{}: {}".format(name, _rows_difference(one, other, MEMORY_FIELDS))
    for name in right:
        if name not in left:
            return "{} is missing".format(name)
    return None


def attribute(left: dict, right: dict) -> dict:
    """Which parts of the searches differ between two observations of one arm."""
    found = dict.fromkeys(("any",) + ATTRIBUTION_FIELDS, False)
    found["any"] = left["search"] != right["search"]
    for rows, others in zip(left["search"], right["search"]):
        labels = [row[0] for row in rows]
        other_labels = [row[0] for row in others]
        common = set(labels) & set(other_labels)
        found["result set"] |= set(labels) != set(other_labels)
        found["count"] |= len(rows) != len(others)
        found["order"] |= [label for label in labels if label in common] != [
            label for label in other_labels if label in common
        ]
        by_label = {row[0]: row for row in rows}
        other_by_label = {row[0]: row for row in others}
        for label in common:
            for field in SHARED_FIELDS:
                index = SEARCH_FIELDS.index(field)
                found[field] |= by_label[label][index] != other_by_label[label][index]
    return found


def hidden_mentions(reads: dict) -> list:
    """Places where a built read shows, or links to, a memory the viewers may not see."""
    places = []

    def check(place: str, label) -> None:
        if label is not None and not label.startswith("v"):
            places.append(
                "{} shows {}".format(
                    place, "an unlabelled memory" if label.startswith("?") else label
                )
            )

    for index, rows in enumerate(reads["search"]):
        for row in rows:
            check("search {}".format(index), row[0])
            for source, target, _ in row[11]:
                check("a link in search {}".format(index), source)
                check("a link in search {}".format(index), target)
    memory_rows = [("get {}".format(label), row) for label, row in reads["get"].items()]
    memory_rows += [
        ("history {}".format(label), row)
        for label, rows in reads["history"].items()
        for row in rows
    ]
    memory_rows += [
        (name, row) for name in ("list active", "list all") for row in reads[name]
    ]
    for place, row in memory_rows:
        if row is not None:
            check(place, row[0])
            check("superseded_by in {}".format(place), row[9])
    return places


def _describe(operation: dict) -> str:
    if operation["op"] == "add":
        return "add {} to {}".format(operation["label"], operation["channel"])
    return "{} {} {}".format(operation["source"], operation["relation"], operation["target"])


def _outcome_counts(outcomes: list) -> dict:
    counts = {}
    for item in outcomes:
        name = "{}: {}".format(item["op"], item["outcome"])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def evaluate(pair: dict, seen: dict) -> dict:
    """Every check of one pair, from the observations of its five worlds."""
    expected = [operation["expect"] for operation in pair["visible"]]
    a, a2, b, plus, r = (seen[world] for world in WORLDS)
    built = a["arms"]["built"]
    details = {}

    differing = [arm for arm in WORLD_ARMS["A"] if a["arms"][arm] != a2["arms"][arm]]
    differing += [
        name for name in ("visible outcomes", "hidden outcomes") if a[name] != a2[name]
    ]
    if differing:
        where = ""
        if differing[0] in a["arms"]:
            where = "; first at {}".format(
                first_difference(a["arms"][differing[0]], a2["arms"][differing[0]])
            )
        details["determinism"] = "A and A2 differ in {}{}".format(", ".join(differing), where)

    problems = []
    for world in ("A", "B"):
        outcomes = seen[world]["visible outcomes"]
        for index, (want, got) in enumerate(zip(expected, outcomes)):
            if want != got:
                problems.append(
                    "visible operation {} ({}) in {}: predicted {}, got {}".format(
                        index, _describe(pair["visible"][index]), world, want, got
                    )
                )
                break
    where = first_difference(built, b["arms"]["built"])
    if where is not None:
        problems.append("built(A) and built(B) differ at {}".format(where))
    if problems:
        details["noninterference"] = "; ".join(problems)

    where = first_difference(built, r["arms"]["plain"])
    if where is not None:
        details["exactness"] = "built(A) and plain(R) differ at {}".format(where)

    leaks = []
    for world in ("A", "B"):
        leaks += [
            "{}: {}".format(world, place)
            for place in hidden_mentions(seen[world]["arms"]["built"])
        ]
        leaks += [
            "{}: get_memory or history of {} returns it".format(world, label)
            for label in seen[world]["hidden returned"]
        ]
    if leaks:
        details["soundness"] = "; ".join(leaks[:5]) + (
            "; and {} more".format(len(leaks) - 5) if len(leaks) > 5 else ""
        )

    if built == plus["arms"]["built"]:
        details["sensitivity"] = "built(A) and built(A+) are identical"
    power = a["arms"]["after-naive"] != b["arms"]["after-naive"]
    checks = {
        "determinism": not differing,
        "sensitivity": built != plus["arms"]["built"],
        "power": power,
        "noninterference": not problems,
        "exactness": "exactness" not in details,
        "soundness": not leaks,
    }

    hidden = a["hidden outcomes"] + b["hidden outcomes"]
    deliberate = [item for item in hidden if item["deliberate"]]
    issued = [item for item in deliberate if not item["outcome"].startswith("skipped")]
    return {
        "seed": pair["seed"],
        "viewers": pair["viewers"],
        "churn": pair["churn"],
        "carryover": pair["carryover"],
        "embedder": pair["embedder"],
        "checks": checks,
        "first_search_differs": built["search"][0] != plus["arms"]["built"]["search"][0],
        # R holds no hidden part, so a miss here is the model's own error.
        "model_predicts_visible_outcomes": r["visible outcomes"] == expected,
        "details": details,
        "attribution": {
            arm: attribute(a["arms"][arm], b["arms"][arm]) for arm in AFTER_ARMS
        },
        "operations": {
            "visible": len(expected),
            "visible_refused": sum(outcome != "ok" for outcome in expected),
            "hidden": len(hidden),
            "hidden_outcomes": _outcome_counts(hidden),
            "hidden_unpredicted": sum(item["expect"] != item["outcome"] for item in hidden),
            "cross_channel_attempted": len(deliberate),
            "cross_channel_issued": len(issued),
            "cross_channel_refused": sum(
                item["outcome"].startswith("refused") for item in issued
            ),
            "memories": {world: len(seen[world]["labels"]) for world in ("A", "B", "R")},
        },
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
    attribution = {}
    for arm in AFTER_ARMS:
        attribution[arm] = {}
        for group, field, value in GROUPS:
            chosen = [trial for trial in trials if field is None or trial[field] == value]
            row = {"pairs": len(chosen)}
            for part in ("any",) + ATTRIBUTION_FIELDS:
                row[part] = sum(trial["attribution"][arm][part] for trial in chosen)
            attribution[arm][group] = row
    hidden_outcomes = {}
    for trial in trials:
        for name, count in trial["operations"]["hidden_outcomes"].items():
            hidden_outcomes[name] = hidden_outcomes.get(name, 0) + count
    operations = {
        name: sum(trial["operations"][name] for trial in trials)
        for name in (
            "visible",
            "visible_refused",
            "hidden",
            "hidden_unpredicted",
            "cross_channel_attempted",
            "cross_channel_issued",
            "cross_channel_refused",
        )
    }
    operations["hidden_outcomes"] = dict(sorted(hidden_outcomes.items()))
    failures = {}
    for name in CHECKS:
        kept = [
            {"seed": trial["seed"], "detail": trial["details"][name]}
            for trial in trials
            if name in trial["details"]
        ]
        if kept:
            failures[name] = kept[:DETAILS_KEPT]
    return {
        "checks": checks,
        "first_search_differs": sum(trial["first_search_differs"] for trial in trials),
        "model_predicts_visible_outcomes": sum(
            trial["model_predicts_visible_outcomes"] for trial in trials
        ),
        "pairs": {
            "viewers ana": sum(trial["viewers"] == ["ana"] for trial in trials),
            "viewers ana and ben": sum(trial["viewers"] == ["ana", "ben"] for trial in trials),
            "churn": sum(trial["churn"] for trial in trials),
            **{
                group: sum(trial[field] == value for trial in trials)
                for group, field, value in GROUPS
                if field is not None
            },
        },
        "attribution": attribution,
        "operations": operations,
        "failures": failures,
    }


def decide(summary: dict, trials: int) -> dict:
    checks = summary["checks"]
    holds = {
        1: checks["determinism"]["held"] == trials,
        2: checks["sensitivity"]["held"] == trials,
        3: checks["power"]["held"] >= POWER_SHARE * trials,
        4: checks["noninterference"]["held"] == trials,
        5: checks["exactness"]["held"] == trials,
        6: checks["soundness"]["held"] == trials,
    }
    failed_controls = [number for number in CONTROL_RULES if not holds[number]]
    failed_rules = [number for number in BUILT_RULES if not holds[number]]
    if failed_controls:
        verdict = "inconclusive (control {} failed)".format(
            ", ".join(str(number) for number in failed_controls)
        )
    elif not failed_rules:
        verdict = "the built read path passes"
    else:
        seeds = [
            seed for number in failed_rules for seed in checks[RULES[number]]["not_held_seeds"]
        ]
        verdict = "the built read path fails rules {} (first failing seed {})".format(
            ", ".join(str(number) for number in failed_rules),
            min(seeds) if seeds else "unknown",
        )
    decision = {
        "rule_{}_{}".format(number, name): holds[number] for number, name in RULES.items()
    }
    decision["verdict"] = verdict
    return decision


def _share(count: int, total: int) -> str:
    if not total:
        return "-"
    return "{} of {} ({:.1f}%)".format(count, total, 100.0 * count / total)


def _percent(count: int, total: int) -> str:
    return "-" if not total else "{:.0f}%".format(100.0 * count / total)


def render(report: dict) -> str:
    trials = report["trials"]
    summary = report["summary"]
    decision = report["decision"]
    checks = summary["checks"]
    pairs = summary["pairs"]
    operations = summary["operations"]
    hidden = operations["hidden_outcomes"]
    lines = [
        "# Noninterference of reads with viewers (E2)",
        "",
        "Synthetic, scripted data; no LLM. {} pairs (seeds 0 to {}).".format(
            trials, trials - 1
        ),
        "In each pair, worlds A and B run the same visible operations and different",
        "hidden ones, and R runs the visible operations alone. See the docstring of",
        "`experiments/audience_noninterference.py` for the design and decision rule.",
        "",
        "Pairs: viewers ana alone {}, ana and ben {}; viewers joined and left"
        " `others` {};".format(
            pairs["viewers ana"], pairs["viewers ana and ben"], pairs["churn"]
        ),
        "link carry-over keep {}, copy {}, topic {}; no embedder {}, n-gram"
        " embedder {}.".format(
            pairs["keep"],
            pairs["copy"],
            pairs["topic"],
            pairs["no embedder"],
            pairs["n-gram embedder"],
        ),
        "",
        "| Rule | Check | Held in | Needed | Holds |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = (
        (
            1,
            "Determinism: A and A2 identical in every arm and operation outcome",
            "every pair",
        ),
        (2, "Sensitivity: built(A) differs from built(A+)", "every pair"),
        (3, "Power: after-naive differs between A and B", "at least 50%"),
        (
            4,
            "Noninterference: built(A) equals built(B), and visible operations went"
            " as predicted in both",
            "every pair",
        ),
        (5, "Exactness: built(A) equals plain(R)", "every pair"),
        (
            6,
            "Soundness: no built read in A or B shows or links to a hidden memory;"
            " hidden IDs read as missing",
            "every pair",
        ),
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
        "- Sensitivity, first search alone: built(A) and built(A+) differ in {}.".format(
            _share(summary["first_search_differs"], trials)
        ),
        "- after-careful differs between A and B in {}.".format(
            _share(summary["attribution"]["after-careful"]["all"]["any"], trials)
        ),
        "- The visible script's model predicted every visible outcome in R in {}"
        " (a check of the model itself: R holds no hidden part).".format(
            _share(summary["model_predicts_visible_outcomes"], trials)
        ),
        "- Visible operations: {}, of which the visible model predicted {} to be"
        " refused (a `supersedes` across channels or of a newer memory).".format(
            operations["visible"], operations["visible_refused"]
        ),
        "- Hidden operations in A and B together: {}. Adds {}; relates accepted {},"
        " refused because an end was archived {}, refused because the source was"
        " older {}, refused across channels {}; relates skipped because the visible"
        " end was not written yet {} or was archived {}; membership changes {}."
        " Outcomes that differed from the hidden script's own prediction: {}.".format(
            operations["hidden"],
            hidden.get("add: ok", 0),
            hidden.get("relate: ok", 0),
            hidden.get("relate: refused: not active", 0),
            hidden.get("relate: refused: older", 0),
            hidden.get("relate: refused: other channel", 0),
            hidden.get("relate: skipped: not written yet", 0),
            hidden.get("relate: skipped: archived", 0),
            hidden.get("member: ok", 0),
            operations["hidden_unpredicted"],
        ),
        "- Deliberate cross-channel `supersedes`: {} drawn, {} issued (the rest named"
        " a visible label not written yet or archived), {} of the issued ones"
        " refused with ValueError.".format(
            operations["cross_channel_attempted"],
            operations["cross_channel_issued"],
            operations["cross_channel_refused"],
        ),
        "",
        "## Attribution: where filtering after ranking differs between A and B",
        "",
        "Share of pairs in which at least one of the 6 searches differs in each part.",
        "Result set, order and count compare the labels shown; the other parts compare",
        "the rows of labels shown in both A and B.",
    ]
    for arm in AFTER_ARMS:
        table = summary["attribution"][arm]
        lines += [
            "",
            "### {}".format(arm),
            "",
            "| Part | "
            + " | ".join(
                "{} (n={})".format(group, table[group]["pairs"]) for group, _, _ in GROUPS
            )
            + " |",
            "| --- |" + " --- |" * len(GROUPS),
        ]
        for part in ("any",) + ATTRIBUTION_FIELDS:
            lines.append(
                "| {} | ".format("any difference" if part == "any" else part)
                + " | ".join(
                    _percent(table[group][part], table[group]["pairs"])
                    for group, _, _ in GROUPS
                )
                + " |"
            )
    if summary["failures"]:
        lines += [
            "",
            "## Pairs where a check did not hold (first {} per check)".format(DETAILS_KEPT),
            "",
        ]
        for name, items in summary["failures"].items():
            for item in items:
                lines.append("- {}, seed {}: {}".format(name, item["seed"], item["detail"]))
    lines += [
        "",
        "Limits: the data were written alongside the code, so a pass shows that the",
        "read path holds on these scripts, not on every database. The experiment",
        "compares what reads return, so a fault in one of the store's overlapping",
        "link filters that another filter masks would not show here",
        "(`tests/test_audience.py` checks each layer directly). Timestamps, access",
        "counts and raw IDs are excluded from every comparison, and the order of",
        "`list_memories` is not compared (key rewrites tie on `updated_at`), so a",
        "channel through those would not be detected; access counts are shared by",
        "design (`record_access=False` on every read). The only embedder is the n-gram",
        "baseline; pruning and `restore` are not exercised.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "audience_noninterference",
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")

    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "settings": {
            "vocabulary": len(VOCABULARY),
            "words_per_memory": WORDS_PER_MEMORY,
            "keys": len(KEYS),
            "key_share": KEY_SHARE,
            "visible_operations": VISIBLE_OPERATIONS,
            "visible_add": VISIBLE_ADD,
            "hidden_operations": HIDDEN_OPERATIONS,
            "hidden_add": HIDDEN_ADD,
            "hidden_relate": HIDDEN_RELATE,
            "no_channel": NO_CHANNEL,
            "cross_supersedes": CROSS_SUPERSEDES,
            "hidden_partner": HIDDEN_PARTNER,
            "others_join": OTHERS_JOIN,
            "churn": CHURN,
            "searches": SEARCHES,
            "query_words": QUERY_WORDS,
            "kind_filter": KIND_FILTER,
            "limits": LIMITS,
            "list_limit": LIST_LIMIT,
            "big": BIG,
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
