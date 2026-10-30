"""Measure retrieval and pruning against a hand-labelled evaluation set.

An evaluation set is a JSON file; ``eval/synthetic_set.json`` shows the
format. The labels are the researcher's judgement. This module only measures
how well the store reproduces them, next to negative controls built to carry
no real signal:

- wrong links: each link moved to a memory its source is not linked to;
- permuted vectors: each memory embedded as if it were a different memory.

A mechanism is only supported by the data when it beats both the
configuration without it and its control. Beating the control alone is a weak
bar: wrong vectors push relevant memories down, so the permuted-vector control
can score below plain lexical search.
"""

from __future__ import annotations

import inspect
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from .embeddings import Embedder
from .store import LinkCarryover, MemoryStore, MemoryType, Relationship, retention_score

MRR_DEPTH = 10
PRUNE_SETTING_NAMES = (
    "threshold",
    "min_age_days",
    "decay_days",
    "decay_days_by_kind",
    "protect_importance",
)


def _check_member(value: object, enum: type, where: str) -> None:
    try:
        enum(value)
    except ValueError as error:
        raise ValueError("{}: unsupported value {!r}".format(where, value)) from error


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_number(value: object, where: str, expected: str, test) -> None:
    if not _is_number(value) or not test(value):
        raise ValueError("{}: expected {}".format(where, expected))


def _check_memory_key(value: object, memories: dict, where: str) -> None:
    if not isinstance(value, str) or value not in memories:
        raise ValueError("{}: unknown memory {!r}".format(where, value))


def _check_prune(prune: object) -> None:
    if not isinstance(prune, dict):
        raise ValueError("prune: expected an object")
    settings = prune.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("prune.settings: expected an object")
    unknown = sorted(set(settings) - set(PRUNE_SETTING_NAMES))
    if unknown:
        raise ValueError("prune.settings: unknown setting(s) {}".format(", ".join(unknown)))
    for name, expected, test in (
        ("threshold", "a number between 0 and 1", lambda value: 0 <= value <= 1),
        ("min_age_days", "a non-negative number", lambda value: value >= 0),
        ("decay_days", "a positive number", lambda value: value > 0),
    ):
        if name in settings:
            _check_number(settings[name], "prune.settings." + name, expected, test)
    if settings.get("protect_importance") is not None:
        _check_number(
            settings["protect_importance"],
            "prune.settings.protect_importance",
            "null or a number between 0 and 1",
            lambda value: 0 <= value <= 1,
        )
    by_kind = settings.get("decay_days_by_kind")
    if by_kind is not None:
        where = "prune.settings.decay_days_by_kind"
        if not isinstance(by_kind, dict):
            raise ValueError("{}: expected an object of kind -> days".format(where))
        for kind, days in by_kind.items():
            _check_member(kind, MemoryType, where)
            _check_number(
                days, "{}.{}".format(where, kind), "a positive number", lambda value: value > 0
            )

    cases = prune.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("prune.cases: expected a non-empty object")
    for key, case in cases.items():
        where = "prune.cases.{}".format(key)
        if not isinstance(case, dict):
            raise ValueError("{}: expected an object".format(where))
        _check_member(case.get("kind", MemoryType.FACT.value), MemoryType, where + ".kind")
        for field in ("importance", "confidence"):
            _check_number(
                case.get(field),
                "{}.{}".format(where, field),
                "a number between 0 and 1",
                lambda value: 0 <= value <= 1,
            )
        _check_number(
            case.get("idle_days"),
            where + ".idle_days",
            "a non-negative number",
            lambda value: value >= 0,
        )
        access_count = case.get("access_count", 0)
        if not isinstance(access_count, int) or isinstance(access_count, bool) or access_count < 0:
            raise ValueError("{}.access_count: expected a non-negative integer".format(where))
        if case.get("expected") not in ("prune", "keep"):
            raise ValueError("{}.expected: must be 'prune' or 'keep'".format(where))


def load_eval_set(source: Union[str, Path, dict]) -> dict:
    """Read and validate an evaluation set from a JSON file or a dict."""
    data = (
        source
        if isinstance(source, dict)
        else json.loads(Path(source).read_text(encoding="utf-8"))
    )
    if not isinstance(data, dict):
        raise ValueError("an evaluation set must be a JSON object")

    memories = data.get("memories")
    if not isinstance(memories, dict) or not memories:
        raise ValueError("memories: expected a non-empty object of key -> memory")
    for key, memory in memories.items():
        where = "memories.{}".format(key)
        if not isinstance(memory, dict):
            raise ValueError("{}: expected an object".format(where))
        content = memory.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("{}.content: expected a non-empty string".format(where))
        _check_member(memory.get("kind"), MemoryType, where + ".kind")

    links = data.get("links", [])
    if not isinstance(links, list):
        raise ValueError("links: expected a list")
    for index, link in enumerate(links):
        where = "links[{}]".format(index)
        if not isinstance(link, dict):
            raise ValueError("{}: expected an object".format(where))
        for end in ("source", "target"):
            _check_memory_key(link.get(end), memories, "{}.{}".format(where, end))
        if link["source"] == link["target"]:
            raise ValueError("{}: a memory cannot be linked to itself".format(where))
        relation = link.get("relation", Relationship.RELATED_TO.value)
        _check_member(relation, Relationship, where + ".relation")
        if relation == Relationship.SUPERSEDES.value:
            raise ValueError(
                "{}.relation: 'supersedes' would archive its target in the "
                "configurations that use links only; leave superseded memories "
                "out of the set instead".format(where)
            )

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries: expected a non-empty list")
    for index, query in enumerate(queries):
        where = "queries[{}]".format(index)
        if not isinstance(query, dict):
            raise ValueError("{}: expected an object".format(where))
        text = query.get("query")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("{}.query: expected a non-empty string".format(where))
        relevant = query.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError("{}.relevant: expected a non-empty list".format(where))
        for key in relevant:
            _check_memory_key(key, memories, where + ".relevant")
        tags = query.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("{}.tags: expected a list of strings".format(where))
        if "all" in tags:
            raise ValueError("{}.tags: 'all' is reserved for the overall score".format(where))
        stale = query.get("stale", [])
        if not isinstance(stale, list):
            raise ValueError("{}.stale: expected a list".format(where))
        for key in stale:
            _check_memory_key(key, memories, where + ".stale")
            if key in relevant:
                raise ValueError(
                    "{}.stale: {!r} is also listed as relevant".format(where, key)
                )

    updates = data.get("updates", [])
    if not isinstance(updates, list):
        raise ValueError("updates: expected a list")
    position = {key: index for index, key in enumerate(memories)}
    replaced = set()
    for index, update in enumerate(updates):
        where = "updates[{}]".format(index)
        if not isinstance(update, dict):
            raise ValueError("{}: expected an object".format(where))
        for field in ("old", "new"):
            _check_memory_key(update.get(field), memories, "{}.{}".format(where, field))
        if position[update["new"]] <= position[update["old"]]:
            raise ValueError(
                "{}: the new memory must be listed after the old one".format(where)
            )
        if update["old"] in replaced:
            raise ValueError("{}: {!r} is already replaced".format(where, update["old"]))
        replaced.add(update["old"])

    if data.get("prune") is not None:
        _check_prune(data["prune"])
    return data


def load_eval_suite(source: Union[str, Path]) -> list[dict]:
    """Read evaluation sets: one per line of a ``.jsonl`` file, else one ``.json``."""
    path = Path(source)
    if path.suffix != ".jsonl":
        return [load_eval_set(path)]
    eval_sets = []
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                eval_sets.append(load_eval_set(json.loads(line)))
            except ValueError as error:
                raise ValueError("{} line {}: {}".format(path, number, error)) from error
    if not eval_sets:
        raise ValueError("{}: no evaluation sets".format(path))
    return eval_sets


def wrong_links(eval_set: dict, seed: int) -> list[dict]:
    """Negative control: move each link to a memory its source is not linked to.

    A link whose source is already linked to every other memory cannot be
    moved and is dropped.
    """
    rng = random.Random(seed)
    keys = list(eval_set["memories"])
    links = eval_set.get("links", [])
    adjacent = {key: {key} for key in keys}
    for link in links:
        adjacent[link["source"]].add(link["target"])
        adjacent[link["target"]].add(link["source"])
    moved = []
    for link in links:
        candidates = [key for key in keys if key not in adjacent[link["source"]]]
        if candidates:
            moved.append(dict(link, target=rng.choice(candidates)))
    return moved


class PermutedEmbedder:
    """Negative control: embed each memory as if it were a different memory.

    Keeps the real model's similarity scale and number of matches, but breaks
    the link between a memory and its vector. Queries are embedded normally.
    """

    def __init__(self, base: Embedder, contents: Sequence[str], seed: int) -> None:
        contents = list(dict.fromkeys(contents))
        if len(contents) < 2:
            raise ValueError(
                "the permuted-vector control needs at least two distinct memory texts"
            )
        shuffled = list(contents)
        rng = random.Random(seed)
        # Sattolo's algorithm: a random single cycle, so no text keeps its own vector.
        for index in range(len(shuffled) - 1, 0, -1):
            other = rng.randrange(index)
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        self._swap = dict(zip(contents, shuffled))
        self._base = base
        self.name = "{}|permuted:{}".format(base.name, seed)
        self.min_similarity = base.min_similarity

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._base([self._swap.get(text, text) for text in texts])


class CachedEmbedder:
    """Remember each text's vector, so repeated runs embed every text once."""

    def __init__(self, base: Embedder) -> None:
        self._base = base
        self._vectors = {}
        self.name = base.name
        self.min_similarity = base.min_similarity

    def __call__(self, texts: Sequence[str]) -> list[Sequence[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if missing:
            vectors = list(self._base(missing))
            if len(vectors) != len(missing):
                raise ValueError(
                    "embedder returned {} vectors for {} texts".format(
                        len(vectors), len(missing)
                    )
                )
            self._vectors.update(zip(missing, vectors))
        return [self._vectors[text] for text in texts]


def _ordered_updates(eval_set: dict) -> list[dict]:
    position = {key: index for index, key in enumerate(eval_set["memories"])}
    return sorted(
        eval_set.get("updates", []),
        key=lambda update: (position[update["new"]], position[update["old"]]),
    )


def _build_store(
    eval_set: dict,
    links: Sequence[dict],
    embedder: Optional[Embedder],
    *,
    link_carryover: str = LinkCarryover.KEEP.value,
    apply_updates: bool = True,
) -> tuple[MemoryStore, dict[str, str]]:
    """Add the memories in order, then the links, then the set's updates."""
    store = MemoryStore(":memory:", link_carryover=link_carryover)
    ids = {
        key: store.add_memory(memory["content"], memory["kind"]).id
        for key, memory in eval_set["memories"].items()
    }
    for link in links:
        store.relate(
            ids[link["source"]],
            ids[link["target"]],
            link.get("relation", Relationship.RELATED_TO.value),
        )
    if apply_updates:
        for update in _ordered_updates(eval_set):
            store.relate(ids[update["new"]], ids[update["old"]], Relationship.SUPERSEDES)
    # Attached last, so the first search embeds every memory in one batch.
    store.embedder = embedder
    return store, ids


def query_rows(
    eval_sets: Sequence[dict],
    *,
    k: int,
    links_for: Callable[[dict], Sequence[dict]],
    embedder_for: Callable[[dict], Optional[Embedder]],
    link_carryover: str = LinkCarryover.KEEP.value,
    apply_updates: bool = True,
) -> list[dict]:
    """Search every query of every set once; one row of metrics per query.

    ``links_for`` and ``embedder_for`` choose each set's links and embedder,
    which is how configurations and controls differ.
    """
    rows = []
    for set_index, eval_set in enumerate(eval_sets):
        store, ids = _build_store(
            eval_set,
            links_for(eval_set),
            embedder_for(eval_set),
            link_carryover=link_carryover,
            apply_updates=apply_updates,
        )
        keys = {memory_id: key for key, memory_id in ids.items()}
        with store:
            for query in eval_set["queries"]:
                relevant = set(query["relevant"])
                stale = set(query.get("stale", []))
                retrieved = [
                    keys[result.memory.id]
                    for result in store.search(
                        query["query"], limit=max(k, MRR_DEPTH), record_access=False
                    )
                ]
                first_answer = next(
                    (key for key in retrieved if key in relevant or key in stale), None
                )
                rows.append(
                    {
                        "set": eval_set.get("id", set_index),
                        "query": query["query"],
                        "tags": list(query.get("tags", [])),
                        "relevant": sorted(relevant),
                        "retrieved": retrieved[:k],
                        "recall": len(relevant.intersection(retrieved[:k])) / len(relevant),
                        "reciprocal_rank": next(
                            (
                                1.0 / rank
                                for rank, key in enumerate(retrieved, start=1)
                                if key in relevant
                            ),
                            0.0,
                        ),
                        "stale_first": (first_answer in stale) if stale else None,
                    }
                )
    return rows


def mean_rows(row_lists: Sequence[Sequence[dict]]) -> list[dict]:
    """Average each query's metrics over repeated runs, such as control seeds."""
    averaged = []
    for runs in zip(*row_lists):
        row = dict(runs[0], retrieved=[])
        for field in ("recall", "reciprocal_rank"):
            row[field] = sum(run[field] for run in runs) / len(runs)
        if runs[0]["stale_first"] is not None:
            row["stale_first"] = sum(run["stale_first"] for run in runs) / len(runs)
        averaged.append(row)
    return averaged


def summarize(rows: Sequence[dict]) -> dict:
    """Mean recall, MRR and stale-first rate, overall and for each tag."""
    groups = {"all": list(rows)}
    for row in rows:
        for tag in dict.fromkeys(row["tags"]):
            groups.setdefault(tag, []).append(row)
    summary = {}
    for name, group in groups.items():
        with_stale = [row["stale_first"] for row in group if row["stale_first"] is not None]
        summary[name] = {
            "queries": len(group),
            "recall": sum(row["recall"] for row in group) / len(group),
            "mrr": sum(row["reciprocal_rank"] for row in group) / len(group),
            "stale_first": sum(with_stale) / len(with_stale) if with_stale else None,
        }
    return summary


def _configuration(name: str, rows: list[dict], control: bool = False) -> dict:
    configuration = {"name": name, "control": control, "metrics": summarize(rows)}
    if not control:
        configuration["per_query"] = rows
    return configuration


def _contents(eval_set: dict) -> list[str]:
    return [memory["content"].strip() for memory in eval_set["memories"].values()]


def retrieval_report(
    eval_sets: Union[dict, Sequence[dict]],
    *,
    k: int = 3,
    embedder: Optional[Embedder] = None,
    seeds: int = 20,
    link_carryover: str = LinkCarryover.KEEP.value,
) -> dict:
    """Recall@k and MRR for each retrieval configuration and its control.

    Accepts one evaluation set or several (a suite); each set gets its own
    store, and the metrics are pooled over all queries.
    """
    eval_sets = [eval_sets] if isinstance(eval_sets, dict) else list(eval_sets)
    if not eval_sets:
        raise ValueError("no evaluation sets")
    if k < 1:
        raise ValueError("k must be positive")
    if seeds < 1:
        raise ValueError("seeds must be positive")
    try:
        policy = LinkCarryover(link_carryover).value
    except ValueError as error:
        raise ValueError("unsupported link carryover: {}".format(link_carryover)) from error
    has_updates = any(eval_set.get("updates") for eval_set in eval_sets)
    graph = "graph ({})".format(policy) if has_updates else "graph"

    def run(links_for, embedder_for, apply_updates=True):
        return query_rows(
            eval_sets,
            k=k,
            links_for=links_for,
            embedder_for=embedder_for,
            link_carryover=policy,
            apply_updates=apply_updates,
        )

    def no_links(eval_set):
        return []

    def set_links(eval_set):
        return eval_set.get("links", [])

    def no_embedder(eval_set):
        return None

    configurations = [_configuration("lexical", run(no_links, no_embedder))]
    if has_updates:
        configurations.append(
            _configuration(
                "lexical, updates not applied", run(no_links, no_embedder, apply_updates=False)
            )
        )
    configurations.append(_configuration("lexical + " + graph, run(set_links, no_embedder)))
    configurations.append(
        _configuration(
            "lexical + {}, wrong links".format(graph),
            mean_rows(
                [
                    run(lambda eval_set, seed=seed: wrong_links(eval_set, seed), no_embedder)
                    for seed in range(seeds)
                ]
            ),
            control=True,
        )
    )

    if embedder is not None:
        cached = CachedEmbedder(embedder)

        def same_embedder(eval_set):
            return cached

        configurations.append(_configuration("lexical + vector", run(no_links, same_embedder)))
        configurations.append(
            _configuration(
                "lexical + vector, permuted vectors",
                mean_rows(
                    [
                        run(
                            no_links,
                            lambda eval_set, seed=seed: PermutedEmbedder(
                                cached, _contents(eval_set), seed
                            ),
                        )
                        for seed in range(seeds)
                    ]
                ),
                control=True,
            )
        )
        configurations.append(
            _configuration("lexical + vector + " + graph, run(set_links, same_embedder))
        )

    return {
        "k": k,
        "mrr_depth": max(k, MRR_DEPTH),
        "seeds": seeds,
        "sets": len(eval_sets),
        "embedder": getattr(embedder, "name", None),
        "link_carryover": policy if has_updates else None,
        "configurations": configurations,
    }


def _prune_settings(overrides: dict) -> dict:
    parameters = inspect.signature(MemoryStore.prune).parameters
    settings = {name: parameters[name].default for name in PRUNE_SETTING_NAMES}
    settings.update(overrides)
    return settings


def prune_report(eval_set: dict, *, as_of: Optional[datetime] = None) -> Optional[dict]:
    """Compare a dry-run prune with each labelled case's expected outcome."""
    prune = eval_set.get("prune")
    if not prune:
        return None
    settings = _prune_settings(prune.get("settings", {}))
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cases = prune["cases"]

    with MemoryStore(":memory:") as store:
        keys = {}
        for key, case in cases.items():
            memory = store.add_memory(
                "prune case {}".format(key),
                case.get("kind", MemoryType.FACT.value),
                importance=case["importance"],
                confidence=case["confidence"],
            )
            keys[memory.id] = key
            access_count = case.get("access_count", 0)
            stamp = (now - timedelta(days=case["idle_days"])).isoformat(
                timespec="microseconds"
            )
            store._set_activity(
                memory.id,
                created_at=stamp,
                last_accessed_at=stamp if access_count else None,
                access_count=access_count,
            )
        flagged = {
            keys[candidate.memory.id]
            for candidate in store.prune(dry_run=True, as_of=now, **settings)
        }

    kind_decay_days = {
        MemoryType(kind).value: days
        for kind, days in (settings["decay_days_by_kind"] or {}).items()
    }
    rows = []
    for key, case in cases.items():
        kind = MemoryType(case.get("kind", MemoryType.FACT.value)).value
        if (
            settings["protect_importance"] is not None
            and case["importance"] >= settings["protect_importance"]
        ):
            note = "protected"
        elif case["idle_days"] < settings["min_age_days"]:
            note = "too recent"
        else:
            note = ""
        rows.append(
            {
                "case": key,
                "kind": kind,
                "expected": case["expected"],
                "outcome": "prune" if key in flagged else "keep",
                "score": retention_score(
                    case["importance"],
                    case["confidence"],
                    case["idle_days"],
                    case.get("access_count", 0),
                    kind_decay_days.get(kind, settings["decay_days"]),
                ),
                "note": note,
            }
        )
    expected_prune = {row["case"] for row in rows if row["expected"] == "prune"}
    return {
        "settings": settings,
        "cases": rows,
        "correct": sum(1 for row in rows if row["expected"] == row["outcome"]),
        "precision": len(expected_prune & flagged) / len(flagged) if flagged else None,
        "recall": len(expected_prune & flagged) / len(expected_prune) if expected_prune else None,
    }


def _is_numeric_text(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    columns = list(zip(header, *rows))
    widths = [max(len(cell) for cell in column) for column in columns]
    numeric = [
        bool(rows) and all(_is_numeric_text(cell) for cell in column[1:])
        for column in columns
    ]
    return [
        "  ".join(
            cell.rjust(width) if is_numeric else cell.ljust(width)
            for cell, width, is_numeric in zip(line, widths, numeric)
        ).rstrip()
        for line in [header, *rows]
    ]


def format_report(retrieval: Optional[dict], prune: Optional[dict]) -> str:
    lines = []
    if retrieval is not None:
        configurations = retrieval["configurations"]
        k = retrieval["k"]
        tags = [name for name in configurations[0]["metrics"] if name != "all"]
        total = configurations[0]["metrics"]["all"]["queries"]
        show_stale = any(
            configuration["metrics"]["all"]["stale_first"] is not None
            for configuration in configurations
        )
        details = [
            "{} queries in {} set{}".format(
                total, retrieval["sets"], "" if retrieval["sets"] == 1 else "s"
            ),
            "controls averaged over {} seeds".format(retrieval["seeds"]),
        ]
        if retrieval["embedder"]:
            details.append("embedder {}".format(retrieval["embedder"]))
        if retrieval["link_carryover"]:
            details.append("link carryover {}".format(retrieval["link_carryover"]))
        lines.append("Retrieval: " + "; ".join(details))
        header = ["configuration", "recall@{}".format(k), "mrr@{}".format(retrieval["mrr_depth"])]
        if show_stale:
            header.append("stale first")
        header += [
            "{} ({})".format(tag, configurations[0]["metrics"][tag]["queries"]) for tag in tags
        ]
        table_rows = []
        for configuration in configurations:
            metrics = configuration["metrics"]
            row = [
                configuration["name"] + (" [control]" if configuration["control"] else ""),
                "{:.2f}".format(metrics["all"]["recall"]),
                "{:.2f}".format(metrics["all"]["mrr"]),
            ]
            if show_stale:
                row.append("{:.2f}".format(metrics["all"]["stale_first"]))
            table_rows.append(row + ["{:.2f}".format(metrics[tag]["recall"]) for tag in tags])
        lines.extend(_table(header, table_rows))
        if tags:
            lines.append("(tag columns show recall@{})".format(k))
        if show_stale:
            lines.append(
                "(stale first: share of queries where an outdated version ranks "
                "above the current one; 0 by construction once updates archive "
                "the outdated versions)"
            )

        final = [configuration for configuration in configurations if not configuration["control"]][-1]
        misses = [row for row in final["per_query"] if row["recall"] < 1]
        lines.append("")
        lines.append("Misses in '{}':".format(final["name"]))
        if not misses:
            lines.append("  none")
        shown = 10
        if len(misses) > shown:
            lines.append("  ({} of {} shown)".format(shown, len(misses)))
        for row in misses[:shown]:
            lines.append(
                "  {!r}: relevant {}; top {}: {}".format(
                    row["query"],
                    ", ".join(row["relevant"]),
                    k,
                    ", ".join(row["retrieved"]) or "nothing",
                )
            )

    if prune is not None:
        if lines:
            lines.append("")
        settings = ", ".join(
            "{}={}".format(name, value)
            for name, value in prune["settings"].items()
            if value is not None
        )
        lines.append("Pruning: {} cases; {}".format(len(prune["cases"]), settings))
        lines.extend(
            _table(
                ["case", "kind", "expected", "outcome", "score", "note"],
                [
                    [
                        row["case"],
                        row["kind"],
                        row["expected"],
                        row["outcome"],
                        "{:.4f}".format(row["score"]),
                        row["note"],
                    ]
                    for row in prune["cases"]
                ],
            )
        )
        lines.append(
            "{}/{} match the labels; prune precision {}, recall {}".format(
                prune["correct"],
                len(prune["cases"]),
                "n/a" if prune["precision"] is None else "{:.2f}".format(prune["precision"]),
                "n/a" if prune["recall"] is None else "{:.2f}".format(prune["recall"]),
            )
        )
    return "\n".join(lines)
