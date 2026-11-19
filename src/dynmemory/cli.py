"""Command-line interface for the Memvane memory prototype.

The CLI is an administrative tool for someone who can open the database file:
it authenticates no one. ``--viewers`` scopes reads to an audience and ``--as``
writes as a bound member (with membership checks), but anyone running the CLI
can choose them. Untrusted users belong behind ``memvane serve`` with tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .embeddings import Embedder, NGramHashEmbedder, SentenceTransformerEmbedder
from .evaluation import format_report, load_eval_suite, prune_report, retrieval_report
from .store import LinkCarryover, Memory, MemoryStore, MemoryType, Relationship


def _memory_json(memory: Memory) -> dict:
    return asdict(memory)


def _print_json(value: object) -> None:
    from .server import to_json

    print(json.dumps(to_json(value), indent=2))


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


def _names(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected one or more comma-separated names")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memvane",
        description="A local-first typed memory store for agent prototypes.",
    )
    parser.add_argument(
        "--db",
        default=".memora.sqlite3",
        help="SQLite database path (default: .memora.sqlite3)",
    )
    parser.add_argument(
        "--key-file",
        help="encrypt text at rest with one-time pads kept in this key database",
    )
    parser.add_argument(
        "--viewers",
        type=_names,
        help="read as this audience (comma-separated member IDs)",
    )
    parser.add_argument(
        "--as",
        dest="as_member",
        help="write as this member, with channel membership checks",
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
    add.add_argument("--channel", help="channel the memory was said in")
    add.add_argument("--speaker", help="who said it (ignored with --as)")
    add.add_argument("--subject", dest="fact_subject", help="person a personal fact is about")
    add.add_argument(
        "--sources", type=_names, help="IDs of the source messages it was derived from"
    )

    link = commands.add_parser(
        "link",
        help="connect two memories (reads SOURCE RELATION TARGET)",
        description="Connect two memories. The link reads SOURCE RELATION TARGET. "
        "'supersedes' archives TARGET at once without a channel; in a channel, "
        "reads apply it for viewers who can see SOURCE.",
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

    get = commands.add_parser("get", help="show one memory")
    get.add_argument("memory_id")

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

    channel = commands.add_parser("channel", help="manage channel membership")
    channel_actions = channel.add_subparsers(dest="action", required=True)
    for action in ("join", "leave"):
        command = channel_actions.add_parser(action, help="{} a channel".format(action))
        command.add_argument("channel")
        command.add_argument("member")
    members = channel_actions.add_parser("members", help="list a channel's members")
    members.add_argument("channel")

    admin = commands.add_parser("admin", help="register a workspace admin")
    admin_actions = admin.add_subparsers(dest="action", required=True)
    admin_actions.add_parser("add").add_argument("member")

    delegation = commands.add_parser("delegation", help="grant or revoke one team key")
    delegation_actions = delegation.add_subparsers(dest="action", required=True)
    for action in ("grant", "revoke"):
        command = delegation_actions.add_parser(action)
        command.add_argument("channel")
        command.add_argument("key")
        command.add_argument("member")
        command.add_argument("--by", required=True, help="the admin acting")

    message = commands.add_parser("message", help="source messages")
    message_actions = message.add_subparsers(dest="action", required=True)
    message_add = message_actions.add_parser("add", help="record a source message")
    message_add.add_argument("content")
    message_add.add_argument("--channel", required=True)
    message_add.add_argument("--speaker", help="who said it (ignored with --as)")
    message_show = message_actions.add_parser("show", help="show one message")
    message_show.add_argument("message_id")
    message_retract = message_actions.add_parser(
        "retract", help="retract a message and clear what was derived from it"
    )
    message_retract.add_argument("message_id")
    message_retract.add_argument("--actor", help="speaker or admin (ignored with --as)")
    message_ack = message_actions.add_parser("ack", help="record a reply or reaction")
    message_ack.add_argument("message_id")
    message_ack.add_argument("--member", help="who acknowledged (ignored with --as)")
    message_ack.add_argument("--kind", choices=["replied", "reacted"], default="reacted")

    disputes = commands.add_parser("disputes", help="show disagreements as tables")
    disputes.add_argument("--channel")
    disputes.add_argument("--key", dest="memory_key")

    who = commands.add_parser("who-knows", help="who stated, acknowledged or had access")
    who.add_argument("memory_id")

    commands.add_parser("stale", help="list stale memories waiting to be rebuilt")
    rebuild = commands.add_parser(
        "rebuild", help="rebuild stale memories from their remaining sources (extractive)"
    )
    rebuild.add_argument("--limit", type=int, default=100)

    secret = commands.add_parser("new-secret", help="write a random token secret to a file")
    secret.add_argument("secret_file")
    token = commands.add_parser("token", help="issue a signed member token")
    token.add_argument("member")
    token.add_argument("--secret-file", required=True)
    token.add_argument("--ttl", type=float, default=3600, help="seconds (default: 3600)")
    serve = commands.add_parser(
        "serve", help="serve the store to token holders over a Unix socket"
    )
    serve.add_argument("--socket", required=True)
    serve.add_argument("--secret-file", required=True)

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


def _read_secret(path: str) -> bytes:
    return bytes.fromhex(Path(path).read_text().strip())


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
    if args.command == "new-secret":
        from .auth import TokenAuthority

        path = Path(args.secret_file)
        if path.exists():
            raise ValueError("secret file already exists: {}".format(path))
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(TokenAuthority.generate_secret().hex() + "\n")
        print("Wrote a new secret to {}".format(path))
        return 0
    if args.command == "token":
        from .auth import TokenAuthority

        print(TokenAuthority(_read_secret(args.secret_file)).issue(args.member, ttl_seconds=args.ttl))
        return 0

    store = MemoryStore(
        args.db,
        embedder=_embedder(args),
        link_carryover=args.link_carryover,
        key_path=args.key_file,
    )
    viewers = args.viewers
    writer = store.for_writer(args.as_member) if args.as_member else None

    if args.command == "init":
        print("Initialized {}".format(args.db))
    elif args.command == "add":
        if writer is not None:
            if args.channel is None:
                raise ValueError("--as needs --channel")
            memory = writer.add_memory(
                args.content, args.type, channel=args.channel,
                memory_key=args.memory_key, fact_subject=args.fact_subject,
                source=args.source, confidence=args.confidence,
                importance=args.importance, source_message_ids=args.sources,
            )
        else:
            memory = store.add_memory(
                args.content,
                args.type,
                memory_key=args.memory_key,
                source=args.source,
                confidence=args.confidence,
                importance=args.importance,
                channel=args.channel,
                speaker=args.speaker,
                fact_subject=args.fact_subject,
                source_message_ids=args.sources,
            )
        _print_json(_memory_json(memory))
    elif args.command == "link":
        store.relate(args.source_id, args.target_id, args.relation)
        if args.relation == Relationship.SUPERSEDES.value:
            target = store.get_memory(args.target_id)
            if target is not None and target.channel is not None:
                print("{} supersedes {} (applied at read time, for viewers who can see it)".format(
                    args.source_id, args.target_id
                ))
            else:
                print("Archived {}; superseded by {}".format(args.target_id, args.source_id))
        else:
            print("Linked {} -> {} ({})".format(args.source_id, args.target_id, args.relation))
    elif args.command == "search":
        results = store.search(
            args.query,
            kind=args.type,
            limit=args.limit,
            record_access=not args.peek,
            viewers=viewers,
        )
        _print_json(
            [
                {
                    "score": result.score,
                    "matched_by": result.matched_by,
                    "claim_status": result.claim_status,
                    "links": [asdict(link) for link in result.links],
                    "memory": _memory_json(result.memory),
                }
                for result in results
            ]
        )
    elif args.command == "get":
        memory = store.get_memory(args.memory_id, viewers=viewers)
        if memory is None:
            raise ValueError("memory not found: {}".format(args.memory_id))
        _print_json(_memory_json(memory))
    elif args.command == "list":
        memories = store.list_memories(
            include_archived=args.include_archived, limit=args.limit, viewers=viewers
        )
        _print_json([_memory_json(memory) for memory in memories])
    elif args.command == "history":
        versions = store.history(args.memory_id, viewers=viewers)
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
    elif args.command == "channel":
        if args.action == "join":
            store.join_channel(args.channel, args.member)
            print("{} joined {}".format(args.member, args.channel))
        elif args.action == "leave":
            store.leave_channel(args.channel, args.member)
            print("{} left {}".format(args.member, args.channel))
        else:
            _print_json(store.members_of(args.channel))
    elif args.command == "admin":
        store.add_admin(args.member)
        print("{} is an admin".format(args.member))
    elif args.command == "delegation":
        if args.action == "grant":
            store.grant_delegation(args.channel, args.key, args.member, granted_by=args.by)
            print("{} holds {} in {}".format(args.member, args.key, args.channel))
        else:
            store.revoke_delegation(args.channel, args.key, args.member, revoked_by=args.by)
            print("Revoked {} on {} in {}".format(args.member, args.key, args.channel))
    elif args.command == "message":
        if args.action == "add":
            if writer is not None:
                result = writer.add_message(args.content, channel=args.channel)
            else:
                if args.speaker is None:
                    raise ValueError("message add needs --speaker or --as")
                result = store.add_message(args.content, channel=args.channel, speaker=args.speaker)
            _print_json(result)
        elif args.action == "show":
            result = store.get_message(args.message_id, viewers=viewers)
            if result is None:
                raise ValueError("message not found: {}".format(args.message_id))
            _print_json(result)
        elif args.action == "retract":
            if writer is not None:
                writer.retract_message(args.message_id)
            else:
                if args.actor is None:
                    raise ValueError("message retract needs --actor or --as")
                store.retract_message(args.message_id, actor_id=args.actor)
            print("Retracted {}".format(args.message_id))
        else:
            if writer is not None:
                writer.acknowledge(args.message_id, args.kind)
            else:
                if args.member is None:
                    raise ValueError("message ack needs --member or --as")
                store.acknowledge(args.message_id, args.member, args.kind)
            print("Recorded {} on {}".format(args.kind, args.message_id))
    elif args.command == "disputes":
        _print_json(store.disputes(viewers=viewers, channel=args.channel, memory_key=args.memory_key))
    elif args.command == "who-knows":
        knowledge = store.who_knows(args.memory_id, viewers=viewers)
        if knowledge is None:
            raise ValueError("memory not found in a channel: {}".format(args.memory_id))
        _print_json(knowledge)
    elif args.command == "stale":
        _print_json([_memory_json(memory) for memory in store.stale_memories()])
    elif args.command == "rebuild":
        from .consolidation import ExtractiveConsolidator

        _print_json(store.rebuild_stale(ExtractiveConsolidator(), limit=args.limit))
    elif args.command == "serve":
        from .auth import TokenAuthority
        from .server import MemvaneServer

        server = MemvaneServer(args.socket, store, TokenAuthority(_read_secret(args.secret_file)))
        print("Serving {} on {} (Ctrl-C to stop)".format(args.db, args.socket), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, ImportError, OSError, sqlite3.Error) as error:
        print("memvane: error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
