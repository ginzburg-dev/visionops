from visionops.cli import build_parser


def test_cli_registers_complete_mvp_commands() -> None:
    parser = build_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if action.dest == "command"  # noqa: SLF001
    )

    assert set(subparsers_action.choices) == {
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
    }
