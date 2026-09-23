"""Command line interface for the personal memory agent."""

import argparse
import logging
import os
import sys

from pymongo.errors import PyMongoError

from . import config, seed
from .agent import PersonalAgent
from .memory import MemoryStore


def _print_score_trace(label: str, memory: dict) -> None:
    trace = memory["score_trace"]
    print(f"\n{label}: {memory.get('text', '')}")
    for name, component in trace["components"].items():
        print(
            f"  {name:10} {component['weight']:.2f} × "
            f"{component['value']:.6f} = {component['contribution']:.6f}"
        )
    print(
        f"  total      = {trace['composite_score']:.6f} "
        f"(uses={trace['use_count']}, successes={trace['success_count']}, "
        f"failures={trace['failure_count']}, age={trace['hours_since_access']:.3f}h)"
    )


def _print_score_changes(before: list[dict], after: list[dict]) -> None:
    after_by_id = {memory["_id"]: memory for memory in after}
    print("\nComposite score changes:")
    for memory in before:
        updated = after_by_id.get(memory["_id"])
        if not updated:
            continue
        delta = updated["composite_score"] - memory["composite_score"]
        print(
            f"  {memory['composite_score']:.6f} -> "
            f"{updated['composite_score']:.6f} ({delta:+.6f}) :: "
            f"{memory.get('text', '')}"
        )
        _print_score_trace("Before", memory)
        _print_score_trace("After", updated)


def _cmd_seed(args: argparse.Namespace) -> int:
    inserted = seed.load(
        count=args.count, session_id=args.session, reset=args.reset
    )
    print(f"Inserted {inserted} memories into {config.DB_NAME}.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    agent = PersonalAgent(session_id=args.session)
    response = agent.run(args.query, top_k=args.top_k, trace=args.trace)
    print(response.answer)
    if args.show_memories and response.memories:
        print("\nRetrieved memories:")
        for memory in response.memories:
            print(
                f"  [{memory.get('kind')}] score={memory['composite_score']:.3f} "
                f"relevance={float(memory.get('score', 0)):.3f} "
                f"importance={float(memory.get('importance', 5)):.2f} "
                f"uses={int(memory.get('use_count', 1))} :: {memory.get('text', '')}"
            )
    if args.feedback:
        agent.give_feedback(response, success=args.feedback == "good")
        print(f"\nFeedback '{args.feedback}' applied to {len(response.memories)} memories.")
    else:
        print(f"\nAuto-scored outcome: {'success' if response.success else 'weak match'}")
    if args.trace:
        if args.feedback:
            response.updated_memories = agent.memory.rescore_memories(
                response.memories, trace=True
            )
        _print_score_changes(response.memories, response.updated_memories)
    return 0


def _cmd_remember(args: argparse.Namespace) -> int:
    store = MemoryStore(session_id=args.session)
    memory_id = store.add_memory(
        args.text, importance=args.importance, tags=args.tags or []
    )
    print(f"Stored memory {memory_id}")
    if args.trace:
        created = store.score_memory(memory_id, relevance=1.0, trace=True)
        print("\nComposite score change: not stored -> stored")
        if created:
            _print_score_trace("After remember", created)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    store = MemoryStore(session_id=args.session)
    for key, value in store.stats().items():
        print(f"{key:20} {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    parser.add_argument("--session", default="default", help="session id")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("AGENT_LOG_LEVEL", "INFO").upper(),
        help="operational log verbosity (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="generate and load demo memories")
    seed_parser.add_argument("--count", type=int, default=60)
    seed_parser.add_argument("--reset", action="store_true", help="drop existing data")
    seed_parser.set_defaults(func=_cmd_seed)

    ask_parser = subparsers.add_parser("ask", help="run a query through the agent")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=config.RECALL_AT_K)
    ask_parser.add_argument("--show-memories", action="store_true")
    ask_parser.add_argument("--trace", action="store_true", help="explain score changes")
    ask_parser.add_argument("--feedback", choices=["good", "bad"])
    ask_parser.set_defaults(func=_cmd_ask)

    remember_parser = subparsers.add_parser("remember", help="store a new memory")
    remember_parser.add_argument("text")
    remember_parser.add_argument("--importance", type=float, default=5.0)
    remember_parser.add_argument("--tags", nargs="*")
    remember_parser.add_argument("--trace", action="store_true", help="explain initial score")
    remember_parser.set_defaults(func=_cmd_remember)

    stats_parser = subparsers.add_parser("stats", help="show memory statistics")
    stats_parser.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return args.func(args)
    except PyMongoError as error:
        print(f"MongoDB error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
