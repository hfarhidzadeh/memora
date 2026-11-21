"""LongMemEval as a shared workspace: do the multi-user protections hold on real text? (Stage B)

Question (Stage B of the evaluation plan in docs/multi_user_memory_review.md).
On real conversation text, where users can legitimately hold identical text,
does ``search(..., viewers=...)`` (filter before ranking, as built)
(a) show nobody a memory from a channel they are not all in, in the ranking, in
a link, in ``list_memories`` or in ``get_memory``;
(b) give every viewer exactly the ranking, scores and links of a store that
holds only what they may see (noninterference);
(c) block nothing a viewer may see; and
(d) cost owners no answers compared with filtering after ranking?
Leakage, over-blocking and utility are reported separately. Scope: the owner
question and the other-user question. Cross-user disputes wait for the human
label review. Not tested here: retraction and rebuild (see
audience_retraction_canary.py), the ``record_access`` side channel
(audience_side_channel.py), and near-duplicate (not identical) text.

Data. LongMemEval_S (cleaned). Each question becomes one history with
``dynmemory.longmemeval.question_memories``: the user's turns as ``episode``
memories in date order, ``related_to`` links between consecutive turns of a
session, evidence = every turn the benchmark marks ``has_answer`` (knowledge-
update questions count all their evidence turns). No AI-written label is used.
Eligible: the 419 non-abstention questions with at least one evidence turn.
Pool P: eligible minus ``Random("stageB-pool").sample(sorted(eligible), n)``,
n = len(eligible) mod 8, so P holds 416 questions. Labels are
``<source>/<key>`` with the ``question_memories`` key, so sessions that recur
across questions never collide.

Families.
- full (the main family): for each seed s in 0..2, P is shuffled with
  ``Random("stageB-main-<s>")`` and cut into 52 workspaces of eight members
  (user00 to user07), each member owning one question. Channels: ``p-<member>``
  (the member alone, holding that member's history), ``general`` (everyone)
  and ``team`` (user00 to user03). ``general`` also holds the full real
  history of a shared question S and ``team`` that of a team question T, both
  drawn with ``Random("stageB-extra-<s>-<w>")`` from eligible questions not
  owned in this workspace (speakers are random current channel members). Churn:
  a late joiner (not in ``general`` while writing; joins afterwards) and a
  leaver (leaves ``general`` after the pre-leave reads); both drawn from the
  members with ``Random("stageB-plan-<s>-<w>")``, as is everything below.
  Writes are interleaved by a random merge that keeps every source's session
  order. Extra units per workspace, all made only of invented words and a
  unique token (checked absent from every question): a private canary per
  member, a general canary, and an outsider memory (spoken by member i in the
  channel of member j, i != j, so author and channel are not confounded).
  Links: the session links, plus, for each member, ``related_to`` links from a
  private turn to a random turn of S and to a random private turn of the next
  member, and a ``contradicts`` link from a random turn of S to a private
  turn, so links cross visibility boundaries.
- bare8: seed 0, workspaces 0-12 with nothing shared, no canaries and no
  churn: eight members, each in a private channel only.
- k1: every eligible question alone in a store (one member).
- hybrid: seed 0, workspaces 0-12 of the full family again with
  ``NGramHashEmbedder`` attached after all writes (leakage and exactness only).

Items per full workspace (every read has ``record_access=False``, ``limit`` =
100000, so the ranking holds every matching memory; utility uses its first 10).
- owner: each member asks their question alone.
- other: three random other members each ask that question alone.
- mixed: the owner and one other member ask it together (audience of two).
- shared: every current ``general`` member asks S; the leaver asked it before
  leaving (should answer) and after (should not); the joiner asked before
  joining (should not); a mixed pair of current members asks S.
- team: user00 to user03 ask T; user04 to user06 ask T (should not); the pair
  (user00, user01) asks T; the pair (user00, user04) asks T (should not).
- canary: each owner asks their private canary; three others ask it; every
  current member asks the general canary; the leaver asks it afterwards; the
  outsider memory is asked by its speaker (should not see it) and by the
  channel's member (should).
Visibility is decided by the experiment's own record of channels and
memberships at read time, never by a store read.

Arms (per item).
- built: ``search(..., viewers=audience)``.
- pooled: ``search`` with no viewers (one store, no access control).
- after-k: the pooled top 10 with hidden memories dropped.
- after-all: the whole pooled ranking with hidden memories dropped, then top 10.
- reference: a fresh unchannelled store written with exactly the memories the
  audience may see (same relative order, the links with both ends among them),
  searched without viewers.

Measures.
- Leakage: result labels outside the audience's visible set; link endpoints
  outside it; ``list_memories(viewers)`` returning a label outside it;
  ``get_memory(viewers)`` returning a hidden memory (a sample of 15 hidden
  memories plus every hidden gold turn); a memory whose stored channel is not
  its intended channel (placement audit, in every workspace).
- Exactness: built against the reference, on the whole ranking: label order,
  score, lexical_score, vector_score, coverage, matched_by, claim_status and
  link endpoints. Classes: exact (same order, every field within 1e-9),
  score-only, links, order. In bare8 and k1, built against a single-user
  ``_build_store`` of that member's own history instead of the record.
- Over-blocking: ``list_memories(viewers)`` missing a visible label;
  ``get_memory(viewers)`` hiding a visible one (sample of 15); a canary a
  viewer may see that is not at rank 1.
- Utility (owner items): RR@10, hit@10, hit@5 of any evidence turn, per arm.
  Delta = built minus after-all in RR@10, averaged over seeds per question,
  bootstrapped over questions (2000 resamples, ``Random("stageB-boot")``, 95%
  percentile interval).

Controls (they gate only a claim of no leak or no change; a failure that two
measures confirm is reported as a failure whatever the controls say).
  C1 harness: in k1, built equals the ``_build_store`` ranking for all 419.
  C2 leak probe: for the other items, the pooled top 10 holds a hidden memory
  in at least 95%, and holds an evidence turn of the owner (hidden from the
  asker) in at least 60%.
  C3 canary probe: the pooled search finds every canary at rank 1.
  C4 determinism: workspace (seed 0, index 0) run twice in separate
  processes with different PYTHONHASHSEED gives the same digest.
  C5 placement: the intended placement function puts owner evidence only in
  the owner's private channel, S only in ``general``, T only in ``team``, and
  the audit of the store finds every memory in its intended channel; no
  canary word occurs in any question.
Time is a counter clock patched into the store (it is not under test).

Decision rule, fixed before running:
  1. Leakage: no leak of any kind in any built read (full, bare8, k1, hybrid).
  2. Exactness: every built read is exact against its reference.
  3. No over-blocking: no missing visible label, no hidden visible sample,
     every visible canary at rank 1.
  4. Utility (reported with rules 1 to 3): "no cost" if the whole interval of
     delta lies above -0.02; "cost" if it lies below -0.02; otherwise
     "unclear". If built and after-all give a different top 10 in under 5% of
     owner items, the reading is "insensitive" and says nothing.
Verdict: "fails (rules <list> failed, first failing <family> seed <s>
workspace <w> item <i>)" if rule 1, 2 or 3 fails, whatever the controls;
otherwise "inconclusive (control <n> failed)" if a control fails; otherwise
"zero leakage, exactness and no over-blocking supported; utility <reading>".

Validation (fixed before running): with a fault injected, a reduced run (seed
0, workspaces 0-3, no k1, bare8 or hybrid) must show the pattern below in
every workspace, and the trace must point at the injected cause.
  ignore-private: rule 1 fails, every leaked label in p-user00, and every
    other member leaks at least once.
  union: rule 1 fails on mixed items only; every single-viewer read is exact.
  overblock-joiner: rule 3 fails (the joiner misses the general canary);
    rule 1 holds.
  links-ignore: rule 1 fails through link endpoints (the cross-channel
    ``contradicts`` links, the only links search shows to a memory outside the
    results). Two independent checks guard that path (the both-ends filter of
    the link lookup and the active-disputant check), so the fault removes both;
    removing either one alone is invisible through ``search``.
  filter-after: rule 2 fails on order or score; rule 1 may also fail.
  speaker-sees: rule 1 fails, every leaked label was written by a member of the
    audience that read it, and the outsider memory leaks to its speaker.

Design-time findings, all made on a synthetic fixture before any run on the
real data: the first predictions for two faults were wrong and were corrected
here. ``speaker-sees`` leaks everything an audience member wrote, not only the
outsider memory (a mixed audience, or a leaver reading their own earlier text);
``links-ignore`` was invisible until both link-side checks were removed (above).
The cross-channel ``contradicts`` links were added for the same reason.

Run from the project root:

    python3 -I experiments/longmemeval_shared_workspace.py --jobs 4
    python3 -I experiments/longmemeval_shared_workspace.py --fault union \\
        --seeds 1 --workspaces 4 --k1 0 --bare 0 --hybrid 0 --no-determinism \\
        --output experiments/results/longmemeval_shared_workspace_fault_union
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynmemory import MemoryStore, NGramHashEmbedder  # noqa: E402
from dynmemory.evaluation import _build_store  # noqa: E402
from dynmemory.longmemeval import question_memories  # noqa: E402
from dynmemory.store import SCHEMA_VERSION  # noqa: E402

DATA = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
BIG = 100000
DEPTH = 10
GENERAL, TEAM = "general", "team"
MEMBERS = tuple("user{:02d}".format(i) for i in range(8))
TEAM_MEMBERS = MEMBERS[:4]
TOLERANCE = 1e-9
MARGIN = 0.02
INSENSITIVE_SHARE = 0.05
LEAK_POWER = 0.95
INFORMATIVE_SHARE = 0.60
BOOTSTRAPS = 2000
SAMPLE = 15
FAULTS = (
    "ignore-private", "union", "overblock-joiner", "links-ignore", "filter-after", "speaker-sees",
)
INVENTED = (
    "blorvik quendal fyrmos zathen krellun moptide vexlor drunqua yibbol thornak glimwex "
    "sporvane jexquil umbrelo tavkorin nolbrix wuzzard cyndrel pravnik ozmelt hexuvan lumbrag "
    "qwistle fendrow"
).split()
CHECKS_RULES = ("leakage", "exactness", "overblocking")
TOKEN = re.compile(r"\w+")

_TICK = itertools.count()
CURRENT = {"joiner": None}


_REAL_NOW = MemoryStore.__dict__["_now"]


def restore_clock() -> None:
    MemoryStore._now = _REAL_NOW


def install_clock() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tick() -> str:
        return (base + timedelta(microseconds=next(_TICK))).isoformat(timespec="microseconds")

    MemoryStore._now = staticmethod(tick)


# --- data ------------------------------------------------------------------------


def parse_questions(questions: list) -> dict:
    parsed = {}
    for question in questions:
        if question["question_id"].endswith("_abs"):
            continue
        memories, links, evidence = question_memories(question)
        if not evidence:
            continue
        keys = list(memories)
        units = [
            list(group)
            for _, group in itertools.groupby(keys, key=lambda key: key.split(":")[0])
        ]
        parsed[question["question_id"]] = {
            "question": question["question"],
            "answer": question["answer"],
            "memories": {key: memories[key]["content"] for key in keys},
            "links": [(link["source"], link["target"]) for link in links],
            "evidence": sorted({key for _, key in evidence}),
            "units": units,
        }
    return parsed


def vocabulary(parsed: dict) -> set:
    words = set()
    for entry in parsed.values():
        words.update(TOKEN.findall(entry["question"].lower()))
        for content in entry["memories"].values():
            words.update(TOKEN.findall(content.lower()))
    return words


def pool_of(eligible: list) -> list:
    drop = len(eligible) % len(MEMBERS)
    dropped = set(random.Random("stageB-pool").sample(sorted(eligible), drop)) if drop else set()
    return sorted(set(eligible) - dropped)


def workspaces_of(pool: list, eligible: list, seed: int, count: int) -> list:
    shuffled = random.Random("stageB-main-{}".format(seed)).sample(sorted(pool), len(pool))
    groups = [shuffled[i:i + len(MEMBERS)] for i in range(0, len(shuffled), len(MEMBERS))]
    specs = []
    for index, owners in enumerate(groups[:count]):
        others = sorted(set(eligible) - set(owners))
        extras = random.Random("stageB-extra-{}-{}".format(seed, index)).sample(others, 2)
        specs.append({"seed": seed, "index": index, "owners": owners, "extras": extras})
    return specs


# --- plan: a pure function of the data and the seed, never of a store -------------


def private(member: str) -> str:
    return "p-" + member


def plan_workspace(parsed: dict, spec: dict, mode: str) -> dict:
    """Return the writes, links, memberships and items of one workspace."""
    owners = spec["owners"]
    members = MEMBERS[: len(owners)]
    rng = random.Random("stageB-plan-{}-{}".format(spec["seed"], spec["index"]))
    full = mode == "full"
    joiner = leaver = None
    if full:
        joiner, leaver = rng.sample(members, 2)
    sources = {}
    for member, question_id in zip(members, owners):
        sources[member] = {"question": question_id, "channel": private(member), "speakers": [member]}
    if full:
        writers = [m for m in members if m != joiner]
        sources["S"] = {"question": spec["extras"][0], "channel": GENERAL, "speakers": writers}
        sources["T"] = {"question": spec["extras"][1], "channel": TEAM, "speakers": list(TEAM_MEMBERS)}

    queues, content, channel_of, speaker_of = {}, {}, {}, {}
    for name, source in sources.items():
        entry = parsed[source["question"]]
        queues[name] = [
            [(name + "/" + key, entry["memories"][key]) for key in unit] for unit in entry["units"]
        ]
        for key in entry["memories"]:
            label = name + "/" + key
            content[label] = entry["memories"][key]
            channel_of[label] = source["channel"]

    items_extra = {}
    if full:
        words = rng.sample(INVENTED, 2 * (len(members) + 2))
        pairs = [words[i:i + 2] for i in range(0, len(words), 2)]

        def canary(label, pair, token):
            text = "{} {} {}".format(pair[0], pair[1], token)
            return (label, text)

        for number, member in enumerate(members):
            token = "zq{}x{}x{}".format(spec["seed"], spec["index"], number)
            label, text = canary(member + "/canary", pairs[number], token)
            queues[member].insert(rng.randint(0, len(queues[member])), [(label, text)])
            content[label], channel_of[label] = text, private(member)
            items_extra[member] = (label, " ".join(pairs[number] + [token]))
        token = "zq{}x{}xg".format(spec["seed"], spec["index"])
        label, text = canary("S/canary-general", pairs[len(members)], token)
        queues["S"].insert(rng.randint(0, len(queues["S"])), [(label, text)])
        content[label], channel_of[label] = text, GENERAL
        items_extra["general"] = (label, " ".join(pairs[len(members)] + [token]))
        speaker, holder = rng.sample(members, 2)
        token = "zq{}x{}xo".format(spec["seed"], spec["index"])
        label, text = canary("outsider", pairs[len(members) + 1], token)
        queues["O"] = [[(label, text)]]
        content[label], channel_of[label] = text, private(holder)
        items_extra["outsider"] = (label, " ".join(pairs[len(members) + 1] + [token]), speaker, holder)
        sources["O"] = {"channel": private(holder), "speakers": [speaker]}

    writes = []
    pending = {name: list(units) for name, units in queues.items()}
    while any(pending.values()):
        names = [name for name in sorted(pending) if pending[name]]
        pick = rng.choices(names, weights=[len(pending[name]) for name in names])[0]
        for label, text in pending[pick].pop(0):
            speaker = rng.choice(sources[pick]["speakers"])
            speaker_of[label] = speaker
            writes.append(
                {"label": label, "content": text, "channel": channel_of[label], "speaker": speaker}
            )

    links = set()
    for name, source in sources.items():
        if name == "O":
            continue
        entry = parsed[source["question"]]
        for a, b in entry["links"]:
            links.add((name + "/" + a, name + "/" + b, "related_to"))
    if full:
        own = {m: sorted(label for label in content if label.startswith(m + "/")
                         and label != m + "/canary") for m in members}
        shared = sorted(label for label in content if label.startswith("S/") and "canary" not in label)
        for number, member in enumerate(members):
            links.add((rng.choice(own[member]), rng.choice(shared), "related_to"))
            links.add((rng.choice(own[member]), rng.choice(own[members[(number + 1) % len(members)]]),
                       "related_to"))
            links.add((rng.choice(shared), rng.choice(own[member]), "contradicts"))
    links = sorted(links)

    channels = {GENERAL: set(members) - {joiner}, TEAM: set(TEAM_MEMBERS)} if full else {}
    for member in members:
        channels[private(member)] = {member}
    pre = {name: set(people) for name, people in channels.items()}
    post = {name: set(people) for name, people in channels.items()}
    if full:
        post[GENERAL] = (post[GENERAL] | {joiner}) - {leaver}

    gold = {
        member: {member + "/" + key for key in parsed[question]["evidence"]}
        for member, question in zip(members, owners)
    }
    if full:
        gold["S"] = {"S/" + key for key in parsed[spec["extras"][0]]["evidence"]}
        gold["T"] = {"T/" + key for key in parsed[spec["extras"][1]]["evidence"]}
    question_text = {
        member: parsed[question]["question"] for member, question in zip(members, owners)
    }
    if full:
        question_text["S"] = parsed[spec["extras"][0]]["question"]
        question_text["T"] = parsed[spec["extras"][1]]["question"]

    plan = {
        "spec": spec, "mode": mode, "members": members, "joiner": joiner, "leaver": leaver,
        "writes": writes, "links": links, "channel_of": channel_of, "content": content,
        "speaker_of": speaker_of,
        "pre": pre, "post": post, "gold": gold, "question_text": question_text,
        "owner_question": dict(zip(members, owners)),
        "pre_items": [], "post_items": [],
    }
    plan["pre_items"], plan["post_items"] = plan_items(plan, rng, items_extra, full)
    return plan


def plan_items(plan: dict, rng: random.Random, canaries: dict, full: bool) -> tuple:
    members, leaver, joiner = plan["members"], plan["leaver"], plan["joiner"]
    pre, post = [], []

    def item(bucket, kind, audience, query, gold=None, owner=None, canary=None):
        bucket.append({
            "id": "{}#{}".format(kind, len(bucket)), "kind": kind, "audience": tuple(audience),
            "query": query, "gold": gold, "owner": owner, "canary": canary,
        })

    for member in members:
        text = plan["question_text"][member]
        gold = plan["gold"][member]
        item(post, "owner", [member], text, gold, member)
        askers = rng.sample([m for m in members if m != member], min(3, len(members) - 1))
        for asker in askers:
            item(post, "other", [asker], text, gold, member)
        if len(members) > 1:
            item(post, "mixed", [member, rng.choice(askers)], text, gold, member)
    if not full:
        return pre, post

    current = [m for m in members if m != leaver]
    for member in current:
        item(post, "shared", [member], plan["question_text"]["S"], plan["gold"]["S"])
    item(pre, "shared-pre", [leaver], plan["question_text"]["S"], plan["gold"]["S"])
    item(pre, "shared-joiner-pre", [joiner], plan["question_text"]["S"], plan["gold"]["S"])
    item(post, "shared-left", [leaver], plan["question_text"]["S"], plan["gold"]["S"])
    item(post, "shared-mixed", current[:2], plan["question_text"]["S"], plan["gold"]["S"])
    for member in TEAM_MEMBERS:
        item(post, "team", [member], plan["question_text"]["T"], plan["gold"]["T"])
    for member in MEMBERS[4:7]:
        item(post, "team-outside", [member], plan["question_text"]["T"], plan["gold"]["T"])
    item(post, "team-mixed", [MEMBERS[0], MEMBERS[1]], plan["question_text"]["T"], plan["gold"]["T"])
    item(post, "team-mixed-out", [MEMBERS[0], MEMBERS[4]], plan["question_text"]["T"], plan["gold"]["T"])
    for member in members:
        label, query = canaries[member]
        item(post, "canary-own", [member], query, owner=member, canary=label)
        for asker in rng.sample([m for m in members if m != member], 3):
            item(post, "canary-other", [asker], query, owner=member, canary=label)
    label, query = canaries["general"]
    for member in current:
        item(post, "canary-general", [member], query, canary=label)
    item(post, "canary-general-left", [leaver], query, canary=label)
    label, query, speaker, holder = canaries["outsider"]
    item(post, "outsider-self", [speaker], query, canary=label)
    item(post, "outsider-owner", [holder], query, canary=label)
    return pre, post


def visible_labels(plan: dict, record: dict, audience: tuple) -> set:
    return {
        label
        for label, channel in plan["channel_of"].items()
        if all(person in record[channel] for person in audience)
    }


def check_plan(plan: dict) -> list:
    """The intended placement, checked against the plan only (control C5)."""
    errors = []
    for member in plan["members"]:
        for label in plan["gold"][member]:
            if plan["channel_of"][label] != private(member):
                errors.append(label)
    for name, channel in (("S", GENERAL), ("T", TEAM)):
        for label in plan["gold"].get(name, ()):
            if plan["channel_of"][label] != channel:
                errors.append(label)
    if len(plan["writes"]) != len(plan["channel_of"]):
        errors.append("write count")
    return errors


# --- faults ----------------------------------------------------------------------


@contextlib.contextmanager
def fault(name):
    if name is None:
        yield
        return
    if name not in FAULTS:
        raise ValueError("unknown fault: {}".format(name))
    saved = {
        key: MemoryStore.__dict__[key]
        for key in ("_visible", "_links_touching", "_active_ids", "search")
    }
    original_visible = MemoryStore.__dict__["_visible"].__func__
    original_links = MemoryStore.__dict__["_links_touching"]
    original_active = MemoryStore.__dict__["_active_ids"]
    original_search = MemoryStore.__dict__["search"]

    def visible(viewers, column="memories.channel"):
        sql, params = original_visible(viewers, column)
        if viewers is None:
            return sql, params
        count = len(viewers)
        if name == "ignore-private":
            sql = sql.replace(column + " IN (", "({} = '{}' OR {} IN (".format(
                column, private(MEMBERS[0]), column), 1)
            sql = sql.replace("HAVING COUNT(*) = ?)", "HAVING COUNT(*) = ?))", 1)
        elif name == "union":
            sql = sql.replace("= ?)", ">= ?)")
            params = list(params)
            params[count] = 1
            params[2 * count + 1] = 1
        elif name == "overblock-joiner" and CURRENT["joiner"] in viewers:
            sql += " AND {} != '{}'".format(column, GENERAL)
        elif name == "speaker-sees":
            speakers = ",".join("?" for _ in viewers)
            sql = sql.replace(column + " IN (", "({} IN (".format(column), 1)
            sql = sql.replace(
                "HAVING COUNT(*) = ?)",
                "HAVING COUNT(*) = ?) OR memories.speaker IN ({}))".format(speakers), 1)
            params = [*params[: count + 1], *viewers, *params[count + 1:]]
        return sql, params

    def links(self, connection, memory_ids, viewers=None):
        return original_links(self, connection, memory_ids, None)

    def active(self, connection, memory_ids, viewers, view):
        return original_active(self, connection, memory_ids, None, view)

    def search(self, query, *, kind=None, limit=10, record_access=True, viewers=None):
        if viewers is None:
            return original_search(self, query, kind=kind, limit=limit,
                                   record_access=record_access, viewers=None)
        viewer_ids = self._read_viewers(viewers)
        results = original_search(self, query, kind=kind, limit=BIG,
                                  record_access=record_access, viewers=None)
        kept = [r for r in results if self.get_memory(r.memory.id, viewers=viewer_ids) is not None]
        return kept[:limit]

    try:
        if name in ("ignore-private", "union", "overblock-joiner", "speaker-sees"):
            MemoryStore._visible = staticmethod(visible)
        elif name == "links-ignore":
            MemoryStore._links_touching = links
            MemoryStore._active_ids = active
        elif name == "filter-after":
            MemoryStore.search = search
        yield
    finally:
        for key, value in saved.items():
            setattr(MemoryStore, key, value)


# --- running one workspace ---------------------------------------------------------


def view_of(results, labels):
    return [
        {
            "label": labels.get(r.memory.id, "?" + r.memory.id),
            "score": r.score, "lexical": r.lexical_score, "vector": r.vector_score,
            "coverage": r.coverage, "matched_by": r.matched_by, "claim_status": r.claim_status,
            "links": sorted(
                (labels.get(l.source_id, "?" + l.source_id),
                 labels.get(l.target_id, "?" + l.target_id), l.relation.value)
                for l in r.links
            ),
        }
        for r in results
    ]


def compare(built, reference, strip=None):
    """Classify built against a reference: exact, order, links or score-only."""
    def name(label):
        return label.split("/", 1)[1] if strip and label.startswith(strip) else label

    if [name(v["label"]) for v in built] != [name(v["label"]) for v in reference]:
        return "order", None
    worst = 0.0
    for one, other in zip(built, reference):
        if one["matched_by"] != other["matched_by"] or one["claim_status"] != other["claim_status"]:
            return "score-only", None
        mine = sorted((name(a), name(b), r) for a, b, r in one["links"])
        theirs = sorted((name(a), name(b), r) for a, b, r in other["links"])
        if mine != theirs:
            return "links", None
        for field in ("score", "lexical", "vector", "coverage"):
            worst = max(worst, abs(one[field] - other[field]))
    return ("exact" if worst <= TOLERANCE else "score-only"), worst


def rank_metrics(top, gold):
    for rank, label in enumerate(top[:DEPTH], start=1):
        if label in gold:
            return {"rr": 1.0 / rank, "hit10": 1.0, "hit5": 1.0 if rank <= 5 else 0.0}
    return {"rr": 0.0, "hit10": 0.0, "hit5": 0.0}


def reference_store(plan, visible, embedder):
    store = MemoryStore(":memory:")
    ids = {}
    for write in plan["writes"]:
        if write["label"] in visible:
            ids[write["label"]] = store.add_memory(write["content"], "episode").id
    for a, b, relation in plan["links"]:
        if a in visible and b in visible:
            store.relate(ids[a], ids[b], relation)
    store.embedder = embedder
    return store, {memory_id: label for label, memory_id in ids.items()}


def run_workspace(
    parsed: dict, spec: dict, mode: str, family: str = None, hybrid: bool = False, flaw=None
) -> dict:
    plan = plan_workspace(parsed, spec, mode)
    embedder = NGramHashEmbedder() if hybrid else None
    result = {
        "family": family or ("hybrid" if hybrid else mode), "seed": spec["seed"], "index": spec["index"],
        "plan_errors": len(check_plan(plan)), "placement_errors": 0, "leak": 0, "link_leak": 0,
        "list_in": 0, "get_in": 0, "list_out": 0, "get_out": 0, "canary_missed": 0,
        "canary_items": 0, "classes": {}, "reads": 0, "leak_kinds": {}, "leak_channels": {},
        "leak_foreign": 0,
        "leakers": [], "exact_kinds": {}, "owner_rows": [], "pooled_other": 0,
        "pooled_other_hidden": 0, "pooled_other_informative": 0, "pooled_canary": 0,
        "pooled_canary_rank1": 0, "answer": {"items": 0, "built": 0, "pooled": 0},
        "holds_copy": 0, "other_items": 0, "kinds": {}, "failures": [],
    }
    if COLLECT_HITS:
        result["hits"] = []
    CURRENT["joiner"] = plan["joiner"]
    store = MemoryStore(":memory:")
    pooled_cache, ref_cache, built_cache, audit_done = {}, {}, {}, set()
    try:
        ids = {}
        for member in plan["members"]:
            store.join_channel(private(member), member)
        if mode == "full":
            for member in plan["members"]:
                if member != plan["joiner"]:
                    store.join_channel(GENERAL, member)
            for member in TEAM_MEMBERS:
                store.join_channel(TEAM, member)
        for write in plan["writes"]:
            ids[write["label"]] = store.add_memory(
                write["content"], "episode", channel=write["channel"], speaker=write["speaker"]
            ).id
        for a, b, relation in plan["links"]:
            store.relate(ids[a], ids[b], relation)
        store.embedder = embedder
        labels = {memory_id: label for label, memory_id in ids.items()}

        for memory in store.list_memories(limit=BIG):
            if memory.channel != plan["channel_of"][labels[memory.id]]:
                result["placement_errors"] += 1
        if len(store.list_memories(limit=BIG)) != len(plan["writes"]):
            result["placement_errors"] += 1

        def pooled_for(query):
            if query not in pooled_cache:
                pooled_cache[query] = view_of(
                    store.search(query, limit=BIG, record_access=False), labels)
            return pooled_cache[query]

        def reference_for(visible_key, visible, query):
            if visible_key not in ref_cache:
                ref_cache[visible_key] = reference_store(plan, visible, embedder)
            ref, ref_labels = ref_cache[visible_key]
            return view_of(ref.search(query, limit=BIG, record_access=False), ref_labels)

        def fail(item, what):
            if len(result["failures"]) < 5:
                result["failures"].append({"item": item["id"], "what": what})

        def audit_audience(audience, record, tag, items):
            key = (audience, tag)
            if key in audit_done:
                return
            audit_done.add(key)
            visible = visible_labels(plan, record, audience)
            listed = {labels[m.id] for m in store.list_memories(
                limit=BIG, viewers=list(audience))}
            result["list_in"] += len(listed - visible)
            result["list_out"] += len(visible - listed)
            sample_rng = random.Random("stageB-audit-{}-{}-{}".format(
                spec["seed"], spec["index"], "+".join(audience) + tag))
            hidden = sorted(set(plan["channel_of"]) - visible)
            wanted = {label for it in items for label in (it["gold"] or ()) if label in hidden}
            probe = set(sample_rng.sample(hidden, min(SAMPLE, len(hidden)))) | wanted
            for label in sorted(probe):
                if store.get_memory(ids[label], viewers=list(audience)) is not None:
                    result["get_in"] += 1
            shown = sorted(visible)
            for label in sample_rng.sample(shown, min(SAMPLE, len(shown))):
                if store.get_memory(ids[label], viewers=list(audience)) is None:
                    result["get_out"] += 1

        def read_items(items, record, tag):
            by_audience = {}
            for it in items:
                by_audience.setdefault(it["audience"], []).append(it)
            for audience, group in by_audience.items():
                audit_audience(audience, record, tag, group)
            for it in items:
                result["kinds"][it["kind"]] = result["kinds"].get(it["kind"], 0) + 1
                visible = visible_labels(plan, record, it["audience"])
                key = (it["audience"], it["query"], tag)
                if key not in built_cache:
                    built_cache[key] = view_of(store.search(
                        it["query"], limit=BIG, record_access=False,
                        viewers=list(it["audience"])), labels)
                built = built_cache[key]
                ranked = [v["label"] for v in built]
                result["reads"] += 1
                hidden = [label for label in ranked if label not in visible]
                ends = {end for v in built for a, b, _ in v["links"] for end in (a, b)}
                link_hidden = [label for label in ends if label not in visible]
                if hidden or link_hidden:
                    result["leak"] += len(hidden)
                    result["link_leak"] += len(link_hidden)
                    result["leak_kinds"][it["kind"]] = result["leak_kinds"].get(it["kind"], 0) + 1
                    for label in hidden + link_hidden:
                        if plan["speaker_of"].get(label) not in it["audience"]:
                            result["leak_foreign"] += 1
                        channel = plan["channel_of"].get(label, "?")
                        result["leak_channels"][channel] = result["leak_channels"].get(channel, 0) + 1
                    if it["audience"][0] not in result["leakers"]:
                        result["leakers"].append(it["audience"][0])
                    fail(it, "leak {} link {}".format(len(hidden), len(link_hidden)))
                reference = reference_for((tag, it["audience"]), visible, it["query"])
                if plan["mode"] == "bare":
                    solo_key = ("solo", it["owner"] or "", it["query"])
                    if it["kind"] == "owner":
                        if solo_key not in built_cache:
                            owner_set = {
                                "memories": {
                                    key: {"content": content, "kind": "episode"}
                                    for key, content in parsed[plan["owner_question"][it["owner"]]]
                                    ["memories"].items()
                                },
                                "links": [
                                    {"source": a, "target": b}
                                    for a, b in parsed[plan["owner_question"][it["owner"]]]["links"]
                                ],
                            }
                            solo, solo_ids = _build_store(owner_set, owner_set["links"], embedder,
                                                          apply_updates=False)
                            solo_labels = {mid: key for key, mid in solo_ids.items()}
                            built_cache[solo_key] = view_of(
                                solo.search(it["query"], limit=BIG, record_access=False), solo_labels)
                            solo.close()
                        klass, _ = compare(built, built_cache[solo_key], strip=it["owner"] + "/")
                        result["classes"]["solo-" + klass] = result["classes"].get("solo-" + klass, 0) + 1
                        if klass != "exact":
                            fail(it, "differs from the single-user store: " + klass)
                klass, _ = compare(built, reference)
                result["classes"][klass] = result["classes"].get(klass, 0) + 1
                if klass != "exact":
                    result["exact_kinds"][it["kind"]] = result["exact_kinds"].get(it["kind"], 0) + 1
                    fail(it, "not exact: " + klass)

                pooled = pooled_for(it["query"])
                pooled_labels = [v["label"] for v in pooled]
                if it["canary"]:
                    result["pooled_canary"] += 1
                    result["pooled_canary_rank1"] += pooled_labels[:1] == [it["canary"]]
                    if it["canary"] in visible:
                        result["canary_items"] += 1
                        if ranked[:1] != [it["canary"]]:
                            result["canary_missed"] += 1
                            fail(it, "visible canary not at rank 1")
                if it["kind"] == "other":
                    result["pooled_other"] += 1
                    result["other_items"] += 1
                    top = pooled_labels[:DEPTH]
                    result["pooled_other_hidden"] += any(label not in visible for label in top)
                    result["pooled_other_informative"] += any(label in it["gold"] for label in top)
                    evidence_text = {plan["content"][label] for label in it["gold"]}
                    result["holds_copy"] += any(plan["content"][label] in evidence_text
                                                for label in visible)
                    answer = parsed[plan["owner_question"][it["owner"]]]["answer"]
                    if isinstance(answer, str) and len(answer) >= 3 and not answer.strip().isdigit():
                        needle = answer.lower().strip()
                        result["answer"]["items"] += 1
                        result["answer"]["built"] += any(
                            needle in plan["content"][label].lower() for label in ranked[:DEPTH])
                        result["answer"]["pooled"] += any(
                            needle in plan["content"][label].lower() for label in pooled_labels[:DEPTH])
                        if COLLECT_HITS:
                            result["hits"].append(hit_record(
                                plan, it, parsed, needle, ranked[:DEPTH], pooled_labels[:DEPTH], visible))
                if it["kind"] == "owner":
                    after_all = [label for label in pooled_labels if label in visible][:DEPTH]
                    after_k = [label for label in pooled_labels[:DEPTH] if label in visible]
                    gold = it["gold"]
                    result["owner_rows"].append({
                        "question": plan["owner_question"][it["owner"]], "seed": spec["seed"],
                        "built": rank_metrics(ranked, gold), "after_all": rank_metrics(after_all, gold),
                        "after_k": rank_metrics(after_k, gold),
                        "pooled": rank_metrics(pooled_labels, gold),
                        "differs": ranked[:DEPTH] != after_all,
                    })

        with fault(flaw):
            read_items(plan["pre_items"], plan["pre"], "pre")
            if mode == "full":
                store.join_channel(GENERAL, plan["joiner"])
                store.leave_channel(GENERAL, plan["leaver"])
            read_items(plan["post_items"], plan["post"], "post")
    finally:
        for ref, _ in ref_cache.values():
            ref.close()
        store.close()
        CURRENT["joiner"] = None
    return result


COLLECT_HITS = False  # the spot-check builder switches this on; results are unchanged


def hit_record(plan, it, parsed, needle, built_top, pooled_top, visible) -> dict:
    """What a human needs to judge one other-user item with a text answer."""
    owner_question = parsed[plan["owner_question"][it["owner"]]]
    def first_match(top):
        for rank, label in enumerate(top, start=1):
            if needle in plan["content"][label].lower():
                return {"rank": rank, "label": label, "channel": plan["channel_of"][label],
                        "speaker": plan["speaker_of"].get(label), "visible": label in visible,
                        "text": plan["content"][label]}
        return None
    return {
        "item": it["id"], "asker": it["audience"][0], "owner": it["owner"],
        "question": owner_question["question"], "answer": owner_question["answer"],
        "evidence": [plan["content"][label] for label in it["gold"]],
        "built": first_match(built_top), "pooled": first_match(pooled_top),
    }


def digest(result: dict) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


# --- jobs, summary, decision ------------------------------------------------------

PARSED = {}
MIXED_KINDS = {"mixed", "shared-mixed", "team-mixed", "team-mixed-out"}


def run_job(job: dict) -> dict:
    return run_workspace(
        PARSED, job["spec"], job["mode"], family=job["family"], hybrid=job["hybrid"],
        flaw=job["flaw"],
    )


def make_jobs(pool, eligible, arguments) -> list:
    jobs = []
    flaw = arguments.fault
    for seed in range(arguments.seeds):
        for spec in workspaces_of(pool, eligible, seed, arguments.workspaces):
            jobs.append({"spec": spec, "mode": "full", "family": "full", "hybrid": False, "flaw": flaw})
    if arguments.seeds:
        zero = workspaces_of(pool, eligible, 0, max(arguments.bare, arguments.hybrid))
        for spec in zero[: arguments.bare]:
            jobs.append({"spec": spec, "mode": "bare", "family": "bare8", "hybrid": False, "flaw": flaw})
        for spec in zero[: arguments.hybrid]:
            jobs.append({"spec": spec, "mode": "full", "family": "hybrid", "hybrid": True, "flaw": flaw})
    for number, question_id in enumerate(sorted(eligible)[: arguments.k1]):
        spec = {"seed": 0, "index": number, "owners": [question_id], "extras": []}
        jobs.append({"spec": spec, "mode": "bare", "family": "k1", "hybrid": False, "flaw": flaw})
    return jobs


def bootstrap(deltas: list) -> tuple:
    rng = random.Random("stageB-boot")
    count = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(BOOTSTRAPS)
    )
    return means[int(0.025 * BOOTSTRAPS)], means[int(0.975 * BOOTSTRAPS) - 1]


def utility_of(results: list):
    rows = [row for r in results if r["family"] == "full" for row in r["owner_rows"]]
    if not rows:
        return None
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question"], []).append(row)
    arms = ("built", "after_all", "after_k", "pooled")

    def mean(items, arm, metric):
        return sum(item[arm][metric] for item in items) / len(items)

    questions = sorted(by_question)
    summary = {
        "questions": len(questions), "items": len(rows),
        "arms": {
            arm: {metric: sum(mean(by_question[q], arm, metric) for q in questions) / len(questions)
                  for metric in ("rr", "hit10", "hit5")}
            for arm in arms
        },
        "differs_share": sum(row["differs"] for row in rows) / len(rows),
    }
    for other in ("after_all", "after_k"):
        deltas = [mean(by_question[q], "built", "rr") - mean(by_question[q], other, "rr")
                  for q in questions]
        low, high = bootstrap(deltas)
        summary["built_minus_" + other] = {
            "mean": sum(deltas) / len(deltas), "low": low, "high": high,
        }
    return summary


def summarise(results: list, determinism, vocabulary_clashes: int) -> dict:
    keys = ("leak", "link_leak", "list_in", "get_in", "list_out", "get_out", "canary_missed",
            "canary_items", "reads", "placement_errors", "plan_errors", "pooled_other",
            "pooled_other_hidden", "pooled_other_informative", "pooled_canary",
            "pooled_canary_rank1", "holds_copy", "other_items")
    total = {key: sum(r[key] for r in results) for key in keys}
    classes, kinds, families = {}, {}, {}
    for r in results:
        families[r["family"]] = families.get(r["family"], 0) + 1
        for name, count in r["classes"].items():
            classes[name] = classes.get(name, 0) + count
        for name, count in r["kinds"].items():
            kinds[name] = kinds.get(name, 0) + count
    answer = {key: sum(r["answer"][key] for r in results) for key in ("items", "built", "pooled")}
    first = {}
    for rule, bad in (
        (1, lambda r: r["leak"] + r["link_leak"] + r["list_in"] + r["get_in"]),
        (2, lambda r: sum(c for n, c in r["classes"].items() if not n.endswith("exact"))),
        (3, lambda r: r["list_out"] + r["get_out"] + r["canary_missed"]),
    ):
        for r in results:
            if bad(r):
                first[rule] = {
                    "family": r["family"], "seed": r["seed"], "workspace": r["index"],
                    "item": r["failures"][0]["item"] if r["failures"] else "(audit)",
                }
                break
    k1 = [r for r in results if r["family"] == "k1"]
    solo = sum(r["classes"].get("solo-exact", 0) for r in k1)
    solo_total = sum(r["classes"].get("solo-exact", 0) + sum(
        c for n, c in r["classes"].items() if n.startswith("solo-") and n != "solo-exact")
        for r in k1)
    other = total["pooled_other"]
    controls = {
        1: (solo == solo_total and solo_total > 0) if k1 else None,
        2: (total["pooled_other_hidden"] >= LEAK_POWER * other
            and total["pooled_other_informative"] >= INFORMATIVE_SHARE * other) if other else None,
        3: (total["pooled_canary_rank1"] == total["pooled_canary"]) if total["pooled_canary"] else None,
        4: determinism,
        5: total["plan_errors"] == 0 and total["placement_errors"] == 0 and vocabulary_clashes == 0,
    }
    return {
        "totals": total, "classes": classes, "kinds": kinds, "families": families,
        "answer": answer, "first_failure": first, "controls": controls,
        "k1_harness": {"exact": solo, "total": solo_total},
        "utility": utility_of(results), "vocabulary_clashes": vocabulary_clashes,
    }


def utility_reading(utility) -> str:
    if utility is None:
        return "not measured"
    if utility["differs_share"] < INSENSITIVE_SHARE:
        return "insensitive (built and after-all rarely differ)"
    delta = utility["built_minus_after_all"]
    if delta["low"] > -MARGIN:
        return "no cost"
    if delta["high"] < -MARGIN:
        return "cost"
    return "unclear"


def decide(summary: dict) -> dict:
    totals = summary["totals"]
    holds = {
        1: totals["leak"] + totals["link_leak"] + totals["list_in"] + totals["get_in"] == 0,
        2: all(name.endswith("exact") for name in summary["classes"]),
        3: totals["list_out"] + totals["get_out"] + totals["canary_missed"] == 0,
    }
    failed = [number for number in holds if not holds[number]]
    broken = [number for number, value in summary["controls"].items() if value is False]
    skipped = [number for number, value in summary["controls"].items() if value is None]
    reading = utility_reading(summary["utility"])
    if failed:
        where = summary["first_failure"][failed[0]]
        verdict = "fails (rules {} failed, first failing {} seed {} workspace {} item {})".format(
            ", ".join(map(str, failed)), where["family"], where["seed"], where["workspace"],
            where["item"])
    elif broken:
        verdict = "inconclusive (control {} failed)".format(", ".join(map(str, broken)))
    else:
        verdict = "zero leakage, exactness and no over-blocking supported; utility {}".format(reading)
        if skipped:
            verdict += " (controls not run: {})".format(", ".join(map(str, skipped)))
    decision = {"rule_{}".format(number): holds[number] for number in holds}
    decision["utility"] = reading
    decision["controls"] = {str(number): value for number, value in summary["controls"].items()}
    decision["verdict"] = verdict
    return decision


def validate_fault(name: str, results: list) -> dict:
    """The pre-registered pattern of each fault, checked in every full workspace."""
    full = [r for r in results if r["family"] == "full"]

    def exact_only(r, kinds):
        return set(r["exact_kinds"]) <= kinds

    rules = {
        "ignore-private": lambda r: r["leak"] + r["link_leak"] > 0
        and set(r["leak_channels"]) <= {private(MEMBERS[0])}
        and set(MEMBERS[1:]) <= set(r["leakers"]),
        "union": lambda r: r["leak"] > 0 and set(r["leak_kinds"]) <= MIXED_KINDS
        and exact_only(r, MIXED_KINDS),
        "overblock-joiner": lambda r: (r["list_out"] + r["canary_missed"]) > 0
        and r["leak"] + r["link_leak"] + r["list_in"] == 0,
        "links-ignore": lambda r: r["link_leak"] > 0 and r["leak"] == 0,
        "filter-after": lambda r: any(not n.endswith("exact") for n in r["classes"]),
        "speaker-sees": lambda r: r["leak"] > 0 and r["leak_foreign"] == 0
        and "outsider-self" in r["leak_kinds"],
    }
    held = [rules[name](r) for r in full]
    return {"fault": name, "workspaces": len(full), "pattern_held": sum(held),
            "holds": bool(full) and all(held)}


# --- report -------------------------------------------------------------------------


def _share(count, total):
    return "{} of {} ({:.1f}%)".format(count, total, 100.0 * count / total) if total else "-"


def render(report: dict) -> str:
    summary, decision = report["summary"], report["decision"]
    totals, utility = summary["totals"], summary["utility"]
    lines = [
        "# LongMemEval as a shared workspace (Stage B)",
        "",
        "Real conversation text; no LLM. Store schema version {}{}. Families (workspaces): {}.".format(
            report["store_schema_version"],
            "; injected fault: {}".format(report["fault"]) if report["fault"] else "",
            ", ".join("{} {}".format(k, v) for k, v in sorted(summary["families"].items())),
        ),
        "See the docstring of `experiments/longmemeval_shared_workspace.py` for the design,",
        "the decision rule and the validation criteria.",
        "",
        "| Rule | Check | Count | Holds |",
        "| --- | --- | --- | --- |",
        "| 1 | Leakage (ranking {}, links {}, list_memories {}, get_memory {}) | {} reads | {} |".format(
            totals["leak"], totals["link_leak"], totals["list_in"], totals["get_in"],
            totals["reads"], decision["rule_1"]),
        "| 2 | Exactness against the reference: {} | {} reads | {} |".format(
            ", ".join("{} {}".format(k, v) for k, v in sorted(summary["classes"].items())),
            totals["reads"], decision["rule_2"]),
        "| 3 | Over-blocking (list_memories {}, get_memory {}, canaries missed {} of {}) | | {} |".format(
            totals["list_out"], totals["get_out"], totals["canary_missed"], totals["canary_items"],
            decision["rule_3"]),
        "",
        "Controls (a failure gates only a claim of no leak or no change):",
        "",
    ]
    names = {
        "1": "C1 harness: single-user stores equal the `_build_store` ranking ({} of {})".format(
            summary["k1_harness"]["exact"], summary["k1_harness"]["total"]),
        "2": "C2 leak probe: pooled top 10 shows a hidden memory in {}, an owner evidence turn in {}".format(
            _share(totals["pooled_other_hidden"], totals["pooled_other"]),
            _share(totals["pooled_other_informative"], totals["pooled_other"])),
        "3": "C3 canary probe: pooled finds every canary at rank 1 ({})".format(
            _share(totals["pooled_canary_rank1"], totals["pooled_canary"])),
        "4": "C4 determinism: two processes, different PYTHONHASHSEED, same digest",
        "5": "C5 placement: plan errors {}, store audit errors {}, canary-word clashes {}".format(
            totals["plan_errors"], totals["placement_errors"], summary["vocabulary_clashes"]),
    }
    for key in ("1", "2", "3", "4", "5"):
        value = decision["controls"][key]
        lines.append("- {}: {}".format(names[key], "not run" if value is None else value))
    lines += ["", "**Verdict: {}**".format(decision["verdict"]), ""]
    if utility:
        lines += [
            "Utility on owner items ({} questions, {} items; any evidence turn, depth 10):".format(
                utility["questions"], utility["items"]),
            "",
            "| Arm | RR@10 | hit@10 | hit@5 |",
            "| --- | --- | --- | --- |",
        ]
        for arm, label in (("built", "built (filter before)"), ("after_all", "after-all"),
                           ("after_k", "after-k"), ("pooled", "pooled (no access control)")):
            values = utility["arms"][arm]
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                label, values["rr"], values["hit10"], values["hit5"]))
        for other, label in (("after_all", "after-all"), ("after_k", "after-k")):
            delta = utility["built_minus_" + other]
            lines.append("")
            lines.append("Built minus {} in RR@10: {:+.4f} (95% interval {:+.4f} to {:+.4f}).".format(
                label, delta["mean"], delta["low"], delta["high"]))
        lines += [
            "",
            "Built and after-all give a different top 10 in {:.1f}% of owner items (needed for a".format(
                100 * utility["differs_share"]),
            "reading: at least 5%). Utility reading: **{}**.".format(decision["utility"]),
            "",
        ]
    lines += [
        "Reported, not part of the verdict:",
        "",
        "- Items by kind: {}.".format(", ".join(
            "{} {}".format(k, v) for k, v in sorted(summary["kinds"].items()))),
        "- Other-user items whose asker holds a verbatim copy of an owner evidence turn: {}.".format(
            _share(totals["holds_copy"], totals["other_items"])),
        "- Other-user items whose top 10 contains the (non-numeric) answer text: built {}, pooled {}"
        " (of {} items with a text answer).".format(
            summary["answer"]["built"], summary["answer"]["pooled"], summary["answer"]["items"]),
        "",
        "Limits: real text but invented access structure (private channels, one shared and one"
        " team history, canaries). Exact-text copies are allowed by design; near-duplicate text,"
        " retraction, rebuild and the `record_access` side channel are not tested here."
        " A human spot-check of other-user results is still required before anyone relies on this.",
        "",
    ]
    if report.get("validation"):
        validation = report["validation"]
        lines += ["Fault validation: pattern held in {} of {} workspaces ({}).".format(
            validation["pattern_held"], validation["workspaces"],
            "as predicted" if validation["holds"] else "NOT as predicted"), ""]
    return "\n".join(lines)


# --- main ---------------------------------------------------------------------------


def determinism_check(data: Path) -> bool:
    digests = []
    for hash_seed in ("1", "2"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        output = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--digest", "--data", str(data)],
            capture_output=True, text=True, env=environment, check=True,
        ).stdout.strip()
        digests.append(output)
    return len(digests[0]) == 64 and digests[0] == digests[1]


def load(data: Path):
    questions = json.loads(data.read_text(encoding="utf-8"))
    parsed = parse_questions(questions)
    del questions
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workspaces", type=int, default=52)
    parser.add_argument("--bare", type=int, default=13)
    parser.add_argument("--hybrid", type=int, default=13)
    parser.add_argument("--k1", type=int, default=10 ** 6)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fault", choices=FAULTS)
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--digest", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "results" / "longmemeval_shared_workspace"
    )
    arguments = parser.parse_args(argv)
    install_clock()
    global PARSED
    PARSED = load(arguments.data)
    eligible = sorted(PARSED)
    pool = pool_of(eligible)
    if arguments.digest:
        spec = workspaces_of(pool, eligible, 0, 1)[0]
        print(digest(run_workspace(PARSED, spec, "full")))
        return 0

    clashes = len(set(INVENTED) & vocabulary(PARSED))
    clashes += any(word.startswith("zq") for word in vocabulary(PARSED))
    jobs = make_jobs(pool, eligible, arguments)
    started = time.time()
    results = []
    if arguments.jobs > 1:
        context = multiprocessing.get_context("fork")
        with context.Pool(arguments.jobs) as workers:
            for number, result in enumerate(workers.imap(run_job, jobs, chunksize=1), start=1):
                results.append(result)
                if number % 10 == 0:
                    print("{}/{} workspaces, {:.0f}s".format(number, len(jobs), time.time() - started),
                          file=sys.stderr, flush=True)
    else:
        for number, job in enumerate(jobs, start=1):
            results.append(run_job(job))
            if number % 10 == 0:
                print("{}/{} workspaces, {:.0f}s".format(number, len(jobs), time.time() - started),
                      file=sys.stderr, flush=True)
    determinism = None
    if not arguments.no_determinism:
        determinism = determinism_check(arguments.data)
    summary = summarise(results, determinism, clashes)
    report = {
        "store_schema_version": SCHEMA_VERSION, "fault": arguments.fault,
        "arguments": {k: str(v) for k, v in vars(arguments).items() if k != "output"},
        "summary": summary, "decision": decide(summary),
        "validation": validate_fault(arguments.fault, results) if arguments.fault else None,
        "per_workspace": [
            {k: r[k] for k in ("family", "seed", "index", "reads", "leak", "link_leak", "list_in",
                               "get_in", "list_out", "get_out", "canary_missed", "classes",
                               "failures")}
            for r in results
        ],
    }
    text = render(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    arguments.output.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
