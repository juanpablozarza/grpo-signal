"""Command-line entry point for grpo-signal."""

from __future__ import annotations

import argparse
from pathlib import Path

_NOT_IMPLEMENTED = "not implemented yet"


def _parse_steps(value: str) -> slice:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected START:STOP")
    start_s, stop_s = value.split(":", 1)
    if ":" in stop_s:
        raise argparse.ArgumentTypeError("expected START:STOP")
    try:
        start = int(start_s) if start_s else None
        stop = int(stop_s) if stop_s else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected START:STOP") from exc
    return slice(start, stop)


def _parse_group_sizes(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("expected G or G,G,...")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected G or G,G,...") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grpo-signal",
        description="Estimate GRPO training signal from rollout traces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser(
        "audit",
        help="estimate dead rates and family mix from a run or traces",
    )
    source = audit.add_mutually_exclusive_group()
    source.add_argument(
        "--from-run",
        type=Path,
        metavar="DIR",
        help="prime-rl run directory with monitors/file traces",
    )
    source.add_argument(
        "--from-traces",
        type=Path,
        metavar="FILE",
        help="vf-run traces.jsonl",
    )
    source.add_argument(
        "--rollouts",
        action="store_true",
        help="later milestone: opt-in live rollouts (requires a cap)",
    )
    audit.add_argument(
        "--steps",
        type=_parse_steps,
        metavar="START:STOP",
        help="dispatch/arrival step range, e.g. 0:50",
    )
    family = audit.add_mutually_exclusive_group()
    family.add_argument(
        "--family-field",
        metavar="FIELD",
        help="TaskData field to use as the family name",
    )
    family.add_argument(
        "--family-map",
        type=Path,
        metavar="CSV",
        help="CSV of task_key,family",
    )
    audit.add_argument(
        "-G",
        "--group-size",
        dest="group_sizes",
        type=_parse_group_sizes,
        metavar="G[,G...]",
        help="group size(s) to score, e.g. 8 or 4,8,16",
    )
    audit.add_argument("--taskset", metavar="ID", help="taskset id for --rollouts")
    audit.add_argument(
        "-N",
        dest="n_rollouts",
        type=int,
        metavar="N",
        help="rollouts per task for --rollouts",
    )
    audit.add_argument(
        "--base-url",
        metavar="URL",
        help="inference endpoint for --rollouts",
    )
    audit.add_argument(
        "--max-tokens",
        type=float,
        metavar="N",
        help="token cap for --rollouts",
    )
    audit.add_argument(
        "--max-usd",
        type=float,
        metavar="USD",
        help="dollar cap for --rollouts",
    )
    audit.add_argument(
        "--fail-if",
        action="append",
        metavar="EXPR",
        help='CI gate, e.g. "dead_rate>0.5"',
    )
    audit.add_argument(
        "--include-text",
        action="store_true",
        help="include prompts and replies in the report (off by default)",
    )

    simulate = sub.add_parser(
        "simulate",
        help="predict trainer-batch mix under a prime-rl config",
    )
    simulate.add_argument("report", type=Path, help="path to report.json")
    simulate.add_argument(
        "--config",
        type=Path,
        metavar="TOML",
        help="prime-rl rl.toml",
    )

    card = sub.add_parser("card", help="write a Hub env signal card from a report")
    card.add_argument("report", type=Path, help="path to report.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command in {"audit", "simulate", "card"}:
        print(_NOT_IMPLEMENTED)
        return 1
    parser.error(f"unknown command: {args.command}")
