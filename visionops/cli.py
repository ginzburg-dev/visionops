from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    for command in COMMANDS:
        subparsers.add_parser(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
