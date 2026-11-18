"""Graph hop, contradiction flags and topic carry-over: do links reveal hidden memories?

Question (paths 2 to 4 of section 8 of docs/multi_user_memory_review.md).
Suppose per-user visibility were added to Memora by dropping hidden memories
from the results after ranking. Besides the ranking statistics (path 1, see
``audience_side_channel.py``), three paths still read memories the viewer
cannot see: the graph hop admits any active neighbour (path 2), contradiction
flags list every active disputant, whatever the kind filter (path 3), and
topic carry-over resolves links through every version of a fact (path 4).
Through each of them, can a viewer learn about a memory they cannot see? Does
the built ``viewers`` read path close each one?

Data. Scripted and synthetic, no LLM, no real text. Every trial builds a
separate workspace for each path from 500 pseudo-words (w0000 to w0499).
Channel ``visible`` has the members ``viewer`` and ``insider``; channel
``hidden`` has ``insider`` only. The viewer is ``viewer``. Every memory is
said by ``insider``, the one member of both channels (the spec names no
speaker).
- Background: 150 memories in ``visible`` and 50 in ``hidden``, 12 random
  words each, kind ``fact``, and 40 links between distinct random pairs of
  background memories in any channels, with a random relation (related_to,
  supports, contradicts or depends_on) and a random direction.
- Planted memories: only these contain the target word ``qtargetword``, which
  is not in the vocabulary. They have no links except the ones listed below.
- Search: the query is the target word alone, ``limit=10``, lexical only (no
  embedder), ``record_access=False``.
- Link carry-over: drawn per trial and path from keep, copy and topic, except
  where the path fixes it.
All random draws of one trial and path come from ``Random("<seed>-<path>")``
and are shared by its worlds, so the worlds differ only in the planted part.
Writes run in this order: memberships, background memories, background links,
then the planted writes of the path.

Paths and worlds.
- graph (path 2). H is 11 words plus the target at a random position; V is
  12 words in ``visible``; one link joins them (random relation and
  direction). Worlds: absent (V only, no link); present (H in ``hidden``,
  linked to V); shared (H in ``visible``, linked to V; positive control);
  unlinked (H in ``hidden``, no link; negative control).
  L: the viewer is shown a memory that does not contain the target. P = L.
- dispute (path 3). C is 11 words plus the target, kind fact, in
  ``visible``; D is 12 words, kind fact or correction at random; a
  ``contradicts`` link joins D and C in a random direction. In a random half
  of trials (a coin flip per trial) every arm searches with ``kind="fact"``;
  disputants are listed whatever the kind filter is, by design. Worlds:
  absent (C only); present (D in ``hidden``, contradicting C); shared (D in
  ``visible``; positive control); unlinked (D in ``hidden``, no link;
  negative control).
  P: a shown result carries a ``contradicts`` link. L: a shown result carries
  a link whose other end the viewer cannot see.
- topic (path 4). Written in this order: O_old (12 words, key ``planted``, in
  ``visible``); H (11 words plus the target); a link between H and O_old
  (random relation and direction); O_new (12 words, key ``planted``, in
  ``visible``), which supersedes O_old. The store is in ``topic`` mode unless
  stated. Worlds: absent (no H); present (H in ``hidden``, linked); shared (H
  in ``visible``, linked; positive control); unlinked (H in ``hidden``, no
  link; negative control); present-keep (as present, in ``keep`` mode; a
  negative control showing that the leak comes from topic resolution);
  present-copy (as present, in ``copy`` mode, where the link is copied to
  O_new at the rewrite; informative).
  L: the viewer is shown a memory that does not contain the target. P = L.

Visibility. A memory is visible if every viewer currently belongs to its
channel; a memory without a channel is never visible. The after arms and the
statistics decide this from the experiment's own record of each memory's
channel and of the memberships, never from the store's ``viewers`` code.

Arms.
- after-naive: ``search(q, kind=kind, limit=10)`` on the store that holds
  everything, then drop the results the viewer cannot see. Links are left as
  they are. This is the obvious first implementation.
- after-careful: the same search with a limit of at least the number of
  memories in the store; drop the results the viewer cannot see; keep only
  the links with both ends visible; then take the first 10.
- built: ``search(q, kind=kind, limit=10, viewers=["viewer"])`` on the store
  that holds everything, so the hidden memories are in the database it reads.
- before (reference): filtering before ranking, emulated. A second store
  receives the visible part only: the same visible writes in the same order,
  the same link-carryover mode and (no) embedder, the viewer's memberships,
  and the links with both ends visible. It is searched without ``viewers``.

The canonical observation of a search is, for each result in order, (label,
kind, content, memory_key, channel, speaker, score, matched_by,
lexical_score, vector_score, coverage, links), with the links sorted as
(label of source, label of target, relation). Floats are compared exactly.
Labels are assigned when memories are written; an ID without a label shows as
"?" + ID. Timestamps, access counts and raw IDs are left out.

Controls.
- positive (shared): the planted memory is in the viewer's own channel, so
  the probe must fire in every arm. If it does not, silence elsewhere means
  nothing.
- negative (absent, unlinked, and present-keep for topic): nothing hidden is
  linked to what the viewer can find, so L must never fire, in any arm.
- sanity counts, reported: no before store holds a hidden memory, and every
  store the other arms read does hold the hidden ones.

Decision rule, fixed before running (per path; "every arm" includes before):
  1. Probe works: in ``shared``, P fires in at least 95% of trials in every
     arm.
  2. Negative controls hold: in ``absent`` and ``unlinked`` (and
     ``present-keep`` for topic), L is false in every trial in every arm.
  3. The baseline leaks: in ``present``, L fires in at least 95% of trials in
     after-naive.
     Stated predictions, reported but not a condition of the verdict:
     after-careful leaks in at least 95% of trials for graph and topic, and
     in 0% for dispute; ``present-copy`` leaks in after-naive and
     after-careful in at least 95% of trials.
  4. The built path closes it: in every world and trial, L is false in the
     built arm wherever it is false in the before arm, and the built arm's
     canonical observation equals the before arm's.
Verdicts per path: "inconclusive (a control failed)" if rule 1 or 2 fails,
otherwise "the baseline leaks through <path>" or "the baseline does not leak
through <path>" by rule 3; separately, "the built path closes <path>" or
"the built path does not close <path>" by rule 4 alone.

Run from the project root (200 trials by default):

    python3 -I experiments/audience_link_paths.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore, MemoryType, Relationship  # noqa: E402

VOCABULARY_SIZE = 500
VOCABULARY = tuple("w{:04d}".format(index) for index in range(VOCABULARY_SIZE))
WORDS_PER_MEMORY = 12
VISIBLE_BACKGROUND = 150
HIDDEN_BACKGROUND = 50
BACKGROUND_LINKS = 40
TARGET = "qtargetword"
LIMIT = 10
VISIBLE_CHANNEL = "visible"
HIDDEN_CHANNEL = "hidden"
VIEWER = "viewer"
INSIDER = "insider"
VIEWERS = (VIEWER,)
MEMBERSHIPS = (
    (VISIBLE_CHANNEL, VIEWER),
    (VISIBLE_CHANNEL, INSIDER),
    (HIDDEN_CHANNEL, INSIDER),
)
SPEAKER = INSIDER
FACT = MemoryType.FACT.value
DISPUTANT_KINDS = (FACT, MemoryType.CORRECTION.value)
KIND_FILTER = FACT
CONTRADICTS = Relationship.CONTRADICTS.value
LINK_RELATIONS = (
    Relationship.RELATED_TO.value,
    Relationship.SUPPORTS.value,
    CONTRADICTS,
    Relationship.DEPENDS_ON.value,
)
CARRYOVER_MODES = ("keep", "copy", "topic")
PLANTED_KEY = "planted"
PATHS = ("graph", "dispute", "topic")
PATH_TITLES = {
    "graph": "Graph hop (path 2)",
    "dispute": "Contradiction flags (path 3)",
    "topic": "Topic carry-over (path 4)",
}
WORLDS = {
    "graph": ("absent", "present", "shared", "unlinked"),
    "dispute": ("absent", "present", "shared", "unlinked"),
    "topic": (
        "absent",
        "present",
        "shared",
        "unlinked",
        "present-keep",
        "present-copy",
    ),
}
NEGATIVE_WORLDS = {
    "graph": ("absent", "unlinked"),
    "dispute": ("absent", "unlinked"),
    "topic": ("absent", "unlinked", "present-keep"),
}
# The topic path fixes the carry-over mode per world; the others draw it.
TOPIC_WORLD_MODES = {"present-keep": "keep", "present-copy": "copy"}
ARMS = ("after-naive", "after-careful", "built", "before")
ARM_LABELS = {
    "after-naive": "filter after ranking, links left as they are",
    "after-careful": "filter after ranking over every match, links trimmed to visible ends",
    "built": "search with viewers (built)",
    "before": "filter before ranking (emulated reference)",
}
STATISTICS = {
    "graph": "L = P: the viewer is shown a memory that does not contain the target.",
    "dispute": (
        "P: a shown result carries a `contradicts` link. L: a shown result carries "
        "a link whose other end the viewer cannot see."
    ),
    "topic": "L = P: the viewer is shown a memory that does not contain the target.",
}
# Trial settings broken down in the report, per path.
SETTINGS_BY_PATH = {
    "graph": ("mode", "relation"),
    "dispute": ("mode", "kind_filter", "disputant_kind"),
    "topic": ("relation",),
}
NEEDED_SHARE = 0.95
DEFAULT_TRIALS = 200
LISTED_FAILURES = 20
# Positions in a row of a canonical observation.
LABEL, CONTENT, LINKS = 0, 2, 11


def _plain(rng: random.Random) -> str:
    return " ".join(rng.choice(VOCABULARY) for _ in range(WORDS_PER_MEMORY))


def _with_target(rng: random.Random) -> str:
    """11 vocabulary words with the target inserted at a random position."""
    words = [rng.choice(VOCABULARY) for _ in range(WORDS_PER_MEMORY - 1)]
    words.insert(rng.randrange(WORDS_PER_MEMORY), TARGET)
    return " ".join(words)


def _directed(rng: random.Random, first: str, second: str) -> tuple:
    """The two ends of a link in a random direction, as (source, target)."""
    return (first, second) if rng.random() < 0.5 else (second, first)


def draw(seed: int, path: str) -> dict:
    """Every random choice of one trial and path; all of its worlds share them."""
    if path not in PATHS:
        raise ValueError("unknown path: {}".format(path))
    rng = random.Random("{}-{}".format(seed, path))
    background = [
        ("bv{:03d}".format(index), VISIBLE_CHANNEL, _plain(rng))
        for index in range(VISIBLE_BACKGROUND)
    ] + [
        ("bh{:03d}".format(index), HIDDEN_CHANNEL, _plain(rng))
        for index in range(HIDDEN_BACKGROUND)
    ]
    links = []
    pairs = set()
    while len(links) < BACKGROUND_LINKS:
        # sample() returns the pair in random order, which sets the direction.
        first, second = rng.sample(range(len(background)), 2)
        pair = frozenset((first, second))
        if pair in pairs:
            continue
        pairs.add(pair)
        links.append(
            (background[first][0], background[second][0], rng.choice(LINK_RELATIONS))
        )
    chosen = {
        "background": background,
        "background_links": links,
        "mode": None,
        "kind_filter": None,
        "disputant_kind": None,
    }
    if path == "graph":
        chosen["contents"] = {"H": _with_target(rng), "V": _plain(rng)}
        chosen["link"] = _directed(rng, "H", "V") + (rng.choice(LINK_RELATIONS),)
    elif path == "dispute":
        chosen["contents"] = {"C": _with_target(rng), "D": _plain(rng)}
        chosen["disputant_kind"] = rng.choice(DISPUTANT_KINDS)
        chosen["link"] = _directed(rng, "D", "C") + (CONTRADICTS,)
        chosen["kind_filter"] = KIND_FILTER if rng.random() < 0.5 else None
    else:
        chosen["contents"] = {
            "O_old": _plain(rng),
            "H": _with_target(rng),
            "O_new": _plain(rng),
        }
        chosen["link"] = _directed(rng, "H", "O_old") + (rng.choice(LINK_RELATIONS),)
    if path != "topic":
        chosen["mode"] = rng.choice(CARRYOVER_MODES)
    return chosen


def script(chosen: dict, path: str, world: str) -> dict:
    """The writes of one world, as operations over labels, and its search settings.

    Operations are ``("join", channel, member)``,
    ``("add", label, content, kind, memory_key, channel)`` and
    ``("relate", source label, target label, relation)``.
    """
    if world not in WORLDS[path]:
        raise ValueError("unknown world for {}: {}".format(path, world))
    operations = [("join", channel, member) for channel, member in MEMBERSHIPS]
    operations += [
        ("add", label, content, FACT, None, channel)
        for label, channel, content in chosen["background"]
    ]
    operations += [
        ("relate", source, target, relation)
        for source, target, relation in chosen["background_links"]
    ]
    contents = chosen["contents"]
    planted_channel = VISIBLE_CHANNEL if world == "shared" else HIDDEN_CHANNEL
    link = ("relate",) + tuple(chosen["link"])
    linked = world not in ("absent", "unlinked")
    if path == "graph":
        if world != "absent":
            operations.append(("add", "H", contents["H"], FACT, None, planted_channel))
        operations.append(("add", "V", contents["V"], FACT, None, VISIBLE_CHANNEL))
        if linked:
            operations.append(link)
    elif path == "dispute":
        operations.append(("add", "C", contents["C"], FACT, None, VISIBLE_CHANNEL))
        if world != "absent":
            operations.append(
                ("add", "D", contents["D"], chosen["disputant_kind"], None, planted_channel)
            )
        if linked:
            operations.append(link)
    else:
        operations.append(
            ("add", "O_old", contents["O_old"], FACT, PLANTED_KEY, VISIBLE_CHANNEL)
        )
        if world != "absent":
            operations.append(("add", "H", contents["H"], FACT, None, planted_channel))
        if linked:
            operations.append(link)
        # Writing the key again supersedes O_old.
        operations.append(
            ("add", "O_new", contents["O_new"], FACT, PLANTED_KEY, VISIBLE_CHANNEL)
        )
    if path == "topic":
        mode = TOPIC_WORLD_MODES.get(world, "topic")
    else:
        mode = chosen["mode"]
    return {"mode": mode, "kind": chosen["kind_filter"], "operations": operations}


def visible_labels(operations: list) -> frozenset:
    """The labels every viewer may see, from the script's own record, not the store's."""
    members = {}
    channels = {}
    for operation in operations:
        if operation[0] == "join":
            members.setdefault(operation[1], set()).add(operation[2])
        elif operation[0] == "add":
            channels[operation[1]] = operation[5]
    return frozenset(
        label
        for label, channel in channels.items()
        if channel is not None
        and all(viewer in members.get(channel, ()) for viewer in VIEWERS)
    )


def visible_part(operations: list, visible: frozenset) -> list:
    """What the before arm replays: the viewers' memberships, visible memories
    and links with both ends visible, in the original order."""
    kept = []
    for operation in operations:
        action = operation[0]
        if (
            (action == "join" and operation[2] in VIEWERS)
            or (action == "add" and operation[1] in visible)
            or (
                action == "relate"
                and operation[1] in visible
                and operation[2] in visible
            )
        ):
            kept.append(operation)
    return kept


def _apply(store: MemoryStore, operations: list) -> dict:
    """Run the operations on a store; returns the label of each memory ID."""
    ids = {}
    for operation in operations:
        action = operation[0]
        if action == "join":
            store.join_channel(operation[1], operation[2])
        elif action == "add":
            _, label, content, kind, key, channel = operation
            ids[label] = store.add_memory(
                content, kind, memory_key=key, channel=channel, speaker=SPEAKER
            ).id
        elif action == "relate":
            _, source, target, relation = operation
            store.relate(ids[source], ids[target], relation)
        else:
            raise ValueError("unknown operation: {}".format(action))
    return {memory_id: label for label, memory_id in ids.items()}


def observe(results: list, labels: dict) -> list:
    """The canonical observation of a search: what is shown, in order, by label."""

    def label(memory_id: str) -> str:
        return labels.get(memory_id, "?" + memory_id)

    return [
        (
            label(result.memory.id),
            result.memory.kind.value,
            result.memory.content,
            result.memory.memory_key,
            result.memory.channel,
            result.memory.speaker,
            result.score,
            result.matched_by,
            result.lexical_score,
            result.vector_score,
            result.coverage,
            tuple(
                sorted(
                    (label(link.source_id), label(link.target_id), link.relation.value)
                    for link in result.links
                )
            ),
        )
        for result in results
    ]


def _other_ends(label: str, link: tuple) -> list:
    """The ends of a link other than the result it is listed on."""
    ends = [link[0], link[1]]
    if label in ends:
        ends.remove(label)
    return ends


def statistics(path: str, observation: list, visible: frozenset) -> tuple:
    """(P, L) for one arm's observation; unknown labels count as not visible."""
    if path == "dispute":
        probe = any(
            link[2] == CONTRADICTS for row in observation for link in row[LINKS]
        )
        leak = any(
            other not in visible
            for row in observation
            for link in row[LINKS]
            for other in _other_ends(row[LABEL], link)
        )
        return probe, leak
    without_target = any(TARGET not in row[CONTENT].split() for row in observation)
    return without_target, without_target


def _read(
    full: MemoryStore,
    full_labels: dict,
    reference: MemoryStore,
    reference_labels: dict,
    visible: frozenset,
    kind,
    big: int,
) -> dict:
    """The observation of every arm for one world."""

    def seen(memory_id: str) -> bool:
        return full_labels.get(memory_id) in visible

    naive = [
        result
        for result in full.search(TARGET, kind=kind, limit=LIMIT, record_access=False)
        if seen(result.memory.id)
    ]
    careful = [
        replace(
            result,
            links=tuple(
                link
                for link in result.links
                if seen(link.source_id) and seen(link.target_id)
            ),
        )
        for result in full.search(TARGET, kind=kind, limit=big, record_access=False)
        if seen(result.memory.id)
    ][:LIMIT]
    built = full.search(
        TARGET, kind=kind, limit=LIMIT, record_access=False, viewers=list(VIEWERS)
    )
    before = reference.search(TARGET, kind=kind, limit=LIMIT, record_access=False)
    return {
        "after-naive": observe(naive, full_labels),
        "after-careful": observe(careful, full_labels),
        "built": observe(built, full_labels),
        "before": observe(before, reference_labels),
    }


def run_world(chosen: dict, path: str, world: str) -> dict:
    plan = script(chosen, path, world)
    operations = plan["operations"]
    visible = visible_labels(operations)
    added = [operation[1] for operation in operations if operation[0] == "add"]
    hidden = frozenset(added) - visible
    big = len(added)
    with MemoryStore(":memory:", link_carryover=plan["mode"]) as full, MemoryStore(
        ":memory:", link_carryover=plan["mode"]
    ) as reference:
        full_labels = _apply(full, operations)
        reference_labels = _apply(reference, visible_part(operations, visible))
        observations = _read(
            full, full_labels, reference, reference_labels, visible, plan["kind"], big
        )
        full_hidden = sum(
            full_labels.get(memory.id) in hidden
            for memory in full.list_memories(include_archived=True, limit=big)
        )
        before_hidden = sum(
            reference_labels.get(memory.id) not in visible
            for memory in reference.list_memories(include_archived=True, limit=big)
        )
    arms = {}
    for arm in ARMS:
        probe, leak = statistics(path, observations[arm], visible)
        arms[arm] = {"p": probe, "l": leak, "observation": observations[arm]}
    return {
        "mode": plan["mode"],
        "kind": plan["kind"],
        "full_hidden": full_hidden,
        "before_hidden": before_hidden,
        "arms": arms,
    }


def run_path(seed: int, path: str) -> dict:
    chosen = draw(seed, path)
    return {
        "settings": {
            "mode": chosen["mode"],
            "kind_filter": chosen["kind_filter"],
            "disputant_kind": chosen["disputant_kind"],
            "relation": chosen["link"][2],
        },
        "worlds": {world: run_world(chosen, path, world) for world in WORLDS[path]},
    }


def run_trial(seed: int) -> dict:
    trial = {"seed": seed}
    for path in PATHS:
        trial[path] = run_path(seed, path)
    return trial


def summarise(trials: list) -> dict:
    summary = {}
    for path in PATHS:
        worlds = {
            world: {arm: {"p": 0, "l": 0} for arm in ARMS} for world in WORLDS[path]
        }
        failures = []
        differs = 0
        extra_leaks = 0
        before_hidden = 0
        fewest_hidden = None
        breakdown = {name: {} for name in SETTINGS_BY_PATH[path]}
        for trial in trials:
            outcome = trial[path]
            for world in WORLDS[path]:
                result = outcome["worlds"][world]
                arms = result["arms"]
                for arm in ARMS:
                    worlds[world][arm]["p"] += bool(arms[arm]["p"])
                    worlds[world][arm]["l"] += bool(arms[arm]["l"])
                observation_differs = (
                    arms["built"]["observation"] != arms["before"]["observation"]
                )
                extra_leak = bool(arms["built"]["l"]) and not arms["before"]["l"]
                differs += observation_differs
                extra_leaks += extra_leak
                if observation_differs or extra_leak:
                    failures.append(
                        {
                            "seed": trial["seed"],
                            "world": world,
                            "observation_differs": observation_differs,
                            "built_leaks_where_before_does_not": extra_leak,
                        }
                    )
                before_hidden += result["before_hidden"]
                if fewest_hidden is None or result["full_hidden"] < fewest_hidden:
                    fewest_hidden = result["full_hidden"]
            present = outcome["worlds"]["present"]["arms"]
            for name in SETTINGS_BY_PATH[path]:
                value = outcome["settings"][name]
                row = breakdown[name].setdefault(
                    "none" if value is None else str(value),
                    dict({"trials": 0}, **{arm: 0 for arm in ARMS}),
                )
                row["trials"] += 1
                for arm in ARMS:
                    row[arm] += bool(present[arm]["l"])
        failures.sort(key=lambda failure: (failure["seed"], failure["world"]))
        summary[path] = {
            "worlds": worlds,
            "comparisons": len(trials) * len(WORLDS[path]),
            "built_differs_from_before": differs,
            "built_leaks_where_before_does_not": extra_leaks,
            "first_failing_seed": failures[0]["seed"] if failures else None,
            "failures": failures[:LISTED_FAILURES],
            "hidden_memories_in_before_stores": before_hidden,
            "fewest_hidden_memories_in_full_store": fewest_hidden,
            "present_leaks_by_setting": breakdown,
        }
    return summary


def decide(summary: dict, trials: int) -> dict:
    needed = NEEDED_SHARE * trials
    decision = {}
    for path in PATHS:
        worlds = summary[path]["worlds"]
        probe_works = all(worlds["shared"][arm]["p"] >= needed for arm in ARMS)
        negatives_hold = all(
            worlds[world][arm]["l"] == 0
            for world in NEGATIVE_WORLDS[path]
            for arm in ARMS
        )
        baseline_leaks = worlds["present"]["after-naive"]["l"] >= needed
        no_extra_leak = summary[path]["built_leaks_where_before_does_not"] == 0
        equals_before = summary[path]["built_differs_from_before"] == 0
        closes = no_extra_leak and equals_before

        careful = worlds["present"]["after-careful"]["l"]
        if path == "dispute":
            predictions = [
                {
                    "statement": "after-careful leaks in 0% of `present` trials",
                    "count": careful,
                    "held": careful == 0,
                }
            ]
        else:
            predictions = [
                {
                    "statement": "after-careful leaks in at least 95% of `present` trials",
                    "count": careful,
                    "held": careful >= needed,
                }
            ]
        if path == "topic":
            for arm in ("after-naive", "after-careful"):
                count = worlds["present-copy"][arm]["l"]
                predictions.append(
                    {
                        "statement": "{} leaks in at least 95% of `present-copy` trials".format(
                            arm
                        ),
                        "count": count,
                        "held": count >= needed,
                    }
                )

        if not (probe_works and negatives_hold):
            verdict = "inconclusive (a control failed)"
        elif baseline_leaks:
            verdict = "the baseline leaks through {}".format(path)
        else:
            verdict = "the baseline does not leak through {}".format(path)
        if closes:
            on_built = "the built path closes {}".format(path)
        else:
            on_built = "the built path does not close {}".format(path)
        decision[path] = {
            "rule_1_probe_works": probe_works,
            "rule_2_negative_controls_hold": negatives_hold,
            "rule_3_baseline_leaks": baseline_leaks,
            "rule_4_built_never_leaks_where_before_does_not": no_extra_leak,
            "rule_4_built_equals_before": equals_before,
            "rule_4_built_closes": closes,
            "predictions": predictions,
            "verdict": verdict,
            "verdict_on_the_built_path": on_built,
        }
    return decision


def render(report: dict) -> str:
    trials = report["trials"]
    lines = [
        "# Link paths: graph hop, contradiction flags and topic carry-over",
        "",
        "Paths 2 to 4 of section 8 of `docs/multi_user_memory_review.md`. Synthetic,",
        "scripted data; lexical search only; the query is the target word alone,",
        "`limit=10`. {} trials (seeds 0 to {}). See the docstring of".format(
            trials, trials - 1
        ),
        "`experiments/audience_link_paths.py` for the design and decision rule.",
        "",
        "Arms:",
        "",
    ]
    lines += ["- {}: {}".format(arm, ARM_LABELS[arm]) for arm in ARMS]
    lines += [
        "",
        "Each cell gives the number of trials in which P fired / in which L fired.",
    ]
    for path in PATHS:
        summary = report["summary"][path]
        decision = report["decision"][path]
        lines += [
            "",
            "## {}".format(PATH_TITLES[path]),
            "",
            STATISTICS[path],
            "",
            "| World | {} |".format(" | ".join(ARMS)),
            "| --- |{}".format(" --- |" * len(ARMS)),
        ]
        for world in WORLDS[path]:
            counts = summary["worlds"][world]
            lines.append(
                "| {} | {} |".format(
                    world,
                    " | ".join(
                        "{} / {}".format(counts[arm]["p"], counts[arm]["l"])
                        for arm in ARMS
                    ),
                )
            )
        lines += [
            "",
            "Trials in which L fired in `present`, by trial setting:",
            "",
            "| Setting | Value | Trials | {} |".format(" | ".join(ARMS)),
            "| --- | --- | --- |{}".format(" --- |" * len(ARMS)),
        ]
        for name, values in summary["present_leaks_by_setting"].items():
            for value in sorted(values):
                row = values[value]
                lines.append(
                    "| {} | {} | {} | {} |".format(
                        name,
                        value,
                        row["trials"],
                        " | ".join(str(row[arm]) for arm in ARMS),
                    )
                )
        lines += [
            "",
            "Decision:",
            "",
            "- Rule 1, probe works (`shared`, P in at least 95% of trials in every arm): {}".format(
                decision["rule_1_probe_works"]
            ),
            "- Rule 2, negative controls hold ({}: L in no trial in any arm): {}".format(
                ", ".join("`{}`".format(world) for world in NEGATIVE_WORLDS[path]),
                decision["rule_2_negative_controls_hold"],
            ),
            "- Rule 3, the baseline leaks (`present`, L in at least 95% of after-naive trials): {}".format(
                decision["rule_3_baseline_leaks"]
            ),
        ]
        for prediction in decision["predictions"]:
            lines.append(
                "- Prediction, not a condition of the verdict: {}: {} ({} of {})".format(
                    prediction["statement"],
                    "held" if prediction["held"] else "did not hold",
                    prediction["count"],
                    trials,
                )
            )
        lines += [
            "- Rule 4, the built arm never leaks where the before arm does not: {}"
            " ({} of {} trial-world pairs fail)".format(
                decision["rule_4_built_never_leaks_where_before_does_not"],
                summary["built_leaks_where_before_does_not"],
                summary["comparisons"],
            ),
            "- Rule 4, the built arm's canonical observation equals the before arm's: {}"
            " ({} of {} differ)".format(
                decision["rule_4_built_equals_before"],
                summary["built_differs_from_before"],
                summary["comparisons"],
            ),
            "- Sanity: hidden memories held by the before stores: {}; fewest hidden"
            " memories in a store the other arms read: {}".format(
                summary["hidden_memories_in_before_stores"],
                summary["fewest_hidden_memories_in_full_store"],
            ),
        ]
        if summary["failures"]:
            lines.append(
                "- Rule 4 failures (first {} as seed/world; first failing seed {}): {}".format(
                    len(summary["failures"]),
                    summary["first_failing_seed"],
                    ", ".join(
                        "{}/{}".format(failure["seed"], failure["world"])
                        for failure in summary["failures"]
                    ),
                )
            )
        lines.append(
            "- **Verdict: {}; {}**".format(
                decision["verdict"], decision["verdict_on_the_built_path"]
            )
        )
    lines += [
        "",
        "Limits: the data were written alongside the code, so this shows whether each",
        "mechanism exists and whether the built path removes it, not how often it",
        "matters in real use. One viewer, two channels, one planted link per path,",
        "lexical search only (no embedder), and memberships that do not change during",
        "a trial. Rule 4 compares search results only: a fault inside one layer of",
        "the store that another layer hides does not change any result and cannot be",
        "seen here (the unit tests in `tests/test_audience.py` check the layers).",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "audience_link_paths",
    )
    arguments = parser.parse_args(argv)
    if arguments.trials < 1:
        parser.error("--trials must be positive")

    trials = [run_trial(seed) for seed in range(arguments.trials)]
    summary = summarise(trials)
    report = {
        "trials": arguments.trials,
        "paths": list(PATHS),
        "worlds": {path: list(WORLDS[path]) for path in PATHS},
        "arms": list(ARMS),
        "settings": {
            "vocabulary": VOCABULARY_SIZE,
            "words_per_memory": WORDS_PER_MEMORY,
            "visible_background": VISIBLE_BACKGROUND,
            "hidden_background": HIDDEN_BACKGROUND,
            "background_links": BACKGROUND_LINKS,
            "limit": LIMIT,
            "needed_share": NEEDED_SHARE,
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
