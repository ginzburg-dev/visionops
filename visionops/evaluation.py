from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionops.hashing import file_sha256
from visionops.training import (
    DatasetConfig,
    environment_versions,
    validate_group_splits,
    write_json,
)

METRIC_FIELDS = ("precision", "recall", "map50", "map50_95")


def _metric(source: Any, *names: str) -> float:
    for name in names:
        if isinstance(source, dict) and name in source:
            return float(source[name])
        value = getattr(source, name, None)
        if value is not None:
            return float(value)
    raise ValueError(f"evaluator result is missing metric {names[0]}")


def evaluate_detector(
    config: DatasetConfig,
    model_path: str | Path,
    out_dir: str | Path,
    evaluator: Callable[..., Any] | None = None,
) -> Path:
    validate_group_splits(config.group_manifest)
    model = Path(model_path)
    if evaluator is None and not model.is_file():
        raise FileNotFoundError(f"model weights not found: {model}")

    arguments = {"data": str(config.data_yaml), "split": "test"}
    run = evaluator if evaluator is not None else _ultralytics_evaluator(model)
    result = run(**arguments)
    metrics = {
        "precision": _metric(result, "precision", "mp", "box_precision"),
        "recall": _metric(result, "recall", "mr", "box_recall"),
        "map50": _metric(result, "map50", "map_50"),
        "map50_95": _metric(result, "map50_95", "map"),
    }
    return write_json(
        Path(out_dir) / "metrics.json",
        {
            "arguments": arguments,
            "metrics": metrics,
            "model_path": str(model),
            "model_sha256": file_sha256(model) if model.is_file() else None,
            "config_sha256": config.config_sha256,
            "split": "test",
            "versions": environment_versions(),
            "evaluated_at": datetime.now(UTC).isoformat(),
        },
    )


def _ultralytics_evaluator(model: Path) -> Callable[..., Any]:
    def run(**kwargs: Any) -> Any:
        from ultralytics import YOLO  # lazy: local training dependency only

        return YOLO(str(model)).val(**kwargs)

    return run
