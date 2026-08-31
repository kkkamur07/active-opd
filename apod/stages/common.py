"""Shared helpers for the pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path


def stage_parser(*, description: str | None, needs_shards: bool = True) -> argparse.ArgumentParser:
    """The flags every stage takes: launch coordinates, not configuration.

    Config comes from ``<run-dir>/resolved_config.yaml`` (frozen by the
    driver); these arguments are what the driver varies per launched process
    (arm, round, GPU shard).
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_index")
    if needs_shards:
        parser.add_argument("--shard", type=int, required=True)
        parser.add_argument("--num-shards", type=int, required=True)
    return parser


def parse_stage_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse plus the validation shared by every stage."""

    args = parser.parse_args(argv)
    if args.round_index < 0:
        parser.error(f"--round must be >= 0; got {args.round_index}")
    if hasattr(args, "num_shards"):
        if args.num_shards < 1:
            parser.error(f"--num-shards must be >= 1; got {args.num_shards}")
        if not 0 <= args.shard < args.num_shards:
            parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    return args
