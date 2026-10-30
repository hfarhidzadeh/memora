"""Command-line interface for the Memora memory prototype."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from typing import Optional, Sequence

from .embeddings import Embedder, NGramHashEmbedder, SentenceTransformerEmbedder
from .evaluation import format_report, load_eval_suite, prune_report, retrieval_report
from .store import LinkCarryover, Memory, MemoryStore, MemoryType, Relationship


def _memory_json(memory: Memory) -> dict:
    return asdict(memory)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2))


def _kind_days(text: str) -> tuple[str, float]:
    kind, separator, days = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected KIND=DAYS, got {!r}".format(text))
    try:
        return MemoryType(kind.strip()).value, float(days)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _number_or_none(text: str) -> Optional[float]:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a number or 'none', got {!r}".format(text)
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memora",
        description="A local-first typed memory store for agent prototypes.",
    )
    parser.add_argument(
        "--db",
        default=".memora.sqlite3",
        help="SQLite database path (default: .memora.sqlite3)",
    )
    parser.add_argument(
        "--embedder",
        choices=["none", "ngram", "sentence-transformers"],
        default="none",
        help="vector backend for add, search and eval (default: none)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="model for the sentence-transformers backend",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        help="cosine similarity a memory needs to count as a vector match "
        "(default depends on the backend)",
    )
    parser.add_argument(
        "--link-carryover",
        choices=[policy.value for policy in LinkCarryover],
        default=LinkCarryover.KEEP.value,
        help="what happens to a memory's links when a newer version replaces it: "
        "keep them on the old version, copy them to the new one, or treat them "
        "as links to the topic's current version (default: keep)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="initialize the database")

    add = commands.add_parser("add", help="save a typed memory")
    add.add_argument("content")
    add.add_argument("--type", choices=[kind.value for kind in MemoryType], required=True)
    add.add_argument("--key", dest="memory_key")
    add.add_argument("--source", default="conversation")
    add.add_argument("--confidence", type=float, default=0.8)
    add.add_argument("--importance", type=float, default=0.5)

    link = commands.add_parser(
        "link",
        help="connect two memories (reads SOURCE RELATION TARGET)",
        description="Connect two active memories. The link reads SOURCE RELATION "
        "TARGET; 'supersedes' also archives TARGET.",
    )
    link.add_argument("source_id")
    link.add_argument("target_id")
    link.add_argument(
        "--relation",
        choices=[relation.value for relation in Relationship],
        default=Relationship.RELATED_TO.value,
    )

    search = commands.add_parser("search", help="retrieve memories by text and links")
    search.add_argument("query")
    search.add_argument("--type", choices=[kind.value for kind in MemoryType])
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--peek",
        action="store_true",
        help="do not count this search as an access",
    )

    listing = commands.add_parser("list", help="list active memories")
    listing.add_argument("--include-archived", action="store_true")
    listing.add_argument("--limit", type=int, default=100)

    history = commands.add_parser("history", help="show every version of a memory")
    history.add_argument("memory_id")

    restore = commands.add_parser("restore", help="reactivate a pruned memory")
    restore.add_argument("memory_id")

    prune = commands.add_parser("prune", help="preview or archive stale memories")
    prune.add_argument("--threshold", type=float, default=0.1)
    prune.add_argument("--min-age-days", type=float, default=30)
    prune.add_argument("--decay-days", type=float, default=90)
    prune.add_argument(
        "--decay",
        type=_kind_days,
        action="append",
        default=[],
        metavar="KIND=DAYS",
        help="decay days for one memory type, e.g. episode=14 (repeatable)",
    )
    prune.add_argument(
        "--protect-importance",
        type=_number_or_none,
        default=0.9,
        help="never prune memories at or above this importance; "
        "'none' protects nothing (default: 0.9)",
    )
    prune.add_argument(
        "--apply",
        action="store_true",
        help="archive matching memories (default is a preview)",
    )

    evaluate = commands.add_parser(
        "eval", help="score retrieval and pruning against a labelled set"
    )
    evaluate.add_argument(
        "eval_set", help="evaluation set (.json) or suite of sets (.jsonl, one per line)"
    )
    evaluate.add_argument("--k", type=int, default=3, help="recall cutoff (default: 3)")
    evaluate.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="random seeds averaged for each negative control (default: 20)",
    )
    evaluate.add_argument("--json", action="store_true", help="print raw JSON")

    return parser


def _embedder(args: argparse.Namespace) -> Optional[Embedder]:
    options = {}
    if args.min_similarity is not None:
        options["min_similarity"] = args.min_similarity
    if args.embedder == "ngram":
        return NGramHashEmbedder(**options)
    if args.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(args.embedding_model, **options)
    return None


def _run(args: argparse.Namespace) -> int:
    if args.command == "eval":
        eval_sets = load_eval_suite(args.eval_set)
        retrieval = retrieval_report(
            eval_sets,
            k=args.k,
            embedder=_embedder(args),
            seeds=args.seeds,
            link_carryover=args.link_carryover,
        )
        pruning = prune_report(eval_sets[0]) if len(eval_sets) == 1 else None
        if args.json:
            _print_json({"retrieval": retrieval, "prune": pruning})
        else:
            print(format_report(retrieval, pruning))
        return 0

    store = MemoryStore(
        args.db, embedder=_embedder(args), link_carryover=args.link_carryover
    )
    if args.command == "init":
        print("Initialized {}".format(args.db))
    elif args.command == "add":
        memory = store.add_memory(
            args.content,
            args.type,
            memory_key=args.memory_key,
            source=args.source,
            confidence=args.confidence,
            importance=args.importance,
        )
        _print_json(_memory_json(memory))
    elif args.command == "link":
        store.relate(args.source_id, args.target_id, args.relation)
        if args.relation == Relationship.SUPERSEDES.value:
            print("Archived {}; superseded by {}".format(args.target_id, args.source_id))
        else:
            print("Linked {} -> {} ({})".format(args.source_id, args.target_id, args.relation))
    elif args.command == "search":
        results = store.search(
            args.query,
            kind=args.type,
            limit=args.limit,
            record_access=not args.peek,
        )
        _print_json(
            [
                {
                    "score": result.score,
                    "matched_by": result.matched_by,
                    "links": [asdict(link) for link in result.links],
                    "memory": _memory_json(result.memory),
                }
                for result in results
            ]
        )
    elif args.command == "list":
        memories = store.list_memories(
            include_archived=args.include_archived, limit=args.limit
        )
        _print_json([_memory_json(memory) for memory in memories])
    elif args.command == "history":
        versions = store.history(args.memory_id)
        if not versions:
            raise ValueError("memory not found: {}".format(args.memory_id))
        _print_json([_memory_json(memory) for memory in versions])
    elif args.command == "restore":
        _print_json(_memory_json(store.restore(args.memory_id)))
    elif args.command == "prune":
        candidates = store.prune(
            threshold=args.threshold,
            min_age_days=args.min_age_days,
            decay_days=args.decay_days,
            decay_days_by_kind=dict(args.decay),
            protect_importance=args.protect_importance,
            dry_run=not args.apply,
        )
        _print_json(
            {
                "mode": "preview" if not args.apply else "archived",
                "count": len(candidates),
                "memories": [
                    {
                        "score": candidate.score,
                        "memory": _memory_json(candidate.memory),
                    }
                    for candidate in candidates
                ],
            }
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, ImportError, OSError, sqlite3.Error) as error:
        print("memora: error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
