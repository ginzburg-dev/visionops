from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

COMMANDS = (
    "ingest",
    "prelabel",
    "select",
    "export-labelstudio",
    "train",
    "evaluate",
    "export-onnx",
    "serve",
    "benchmark",
    "make-showcase",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionops",
        description="Prepare, evaluate, and serve crowd-person detectors.",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--input", nargs="+", type=Path, required=True)
    ingest_parser.add_argument("--out", type=Path, required=True)
    ingest_parser.add_argument("--stride", type=int, default=1)
    ingest_parser.add_argument("--max-frames", type=int)
    ingest_parser.add_argument("--seed", type=int, default=0)
    ingest_parser.add_argument("--group")
    for command in COMMANDS[1:]:
        subparsers.add_parser(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        from visionops.ingest import ingest

        manifest = ingest(
            args.input,
            args.out,
            args.stride,
            args.max_frames,
            args.seed,
            group=args.group,
        )
        print(f"manifest={manifest}")
        return 0
    parser.print_help()
    return 0
