from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionops.hashing import file_sha256

REQUIRED_SPLITS = ("train", "val", "test")
UNREVIEWED_MARKERS = ("prediction", "prelabel", "auto_label", "pseudo_label")


class DatasetConfigError(ValueError):
    """Raised when a VisionOps dataset configuration cannot be trusted for training."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    config_path: Path
    data_yaml: Path
    group_manifest: Path
    labels_reviewed: bool
    class_names: tuple[str, ...]
    config_sha256: str
    raw: dict[str, Any]


def _reject_unreviewed(path: Path, field: str) -> None:
    lowered = path.as_posix().lower()
    for marker in UNREVIEWED_MARKERS:
        if marker in lowered:
            raise DatasetConfigError(
                f"{field} points at unreviewed model output ({marker}): {path}"
            )


def load_dataset_config(path: str | Path) -> DatasetConfig:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise DatasetConfigError(f"dataset config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetConfigError(f"dataset config must be a mapping: {config_path}")
    if raw.get("labels_reviewed") is not True:
        raise DatasetConfigError(
            f"dataset config must set labels_reviewed to true after review: {config_path}"
        )
    for field in ("data_yaml", "group_manifest"):
        if not raw.get(field):
            raise DatasetConfigError(f"dataset config is missing {field}: {config_path}")

    base = config_path.parent
    data_yaml = Path(raw["data_yaml"])
    data_yaml = data_yaml if data_yaml.is_absolute() else base / data_yaml
    group_manifest = Path(raw["group_manifest"])
    group_manifest = group_manifest if group_manifest.is_absolute() else base / group_manifest
    _reject_unreviewed(data_yaml, "data_yaml")
    _reject_unreviewed(group_manifest, "group_manifest")
    for field, resolved in (("data_yaml", data_yaml), ("group_manifest", group_manifest)):
        if not resolved.is_file():
            raise DatasetConfigError(f"{field} does not exist: {resolved}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    for split in REQUIRED_SPLITS:
        if not data.get(split):
            raise DatasetConfigError(f"data yaml is missing the {split} split: {data_yaml}")
        _reject_unreviewed(Path(str(data[split])), f"{split} split")
    names = data.get("names") or {}
    if isinstance(names, dict):
        class_names = tuple(str(names[key]) for key in sorted(names))
    else:
        class_names = tuple(str(name) for name in names)
    if class_names != ("person",):
        raise DatasetConfigError(f"data yaml must declare exactly one person class: {data_yaml}")

    return DatasetConfig(
        config_path=config_path,
        data_yaml=data_yaml,
        group_manifest=group_manifest,
        labels_reviewed=True,
        class_names=class_names,
        config_sha256=file_sha256(config_path),
        raw=raw,
    )


def validate_group_splits(path: str | Path) -> None:
    """Fail when one source group is assigned to more than one split."""
    manifest = Path(path)
    assignments: dict[str, str] = {}
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        group = str(payload.get("split_group", ""))
        split = str(payload.get("split", ""))
        if not group or split not in REQUIRED_SPLITS:
            raise DatasetConfigError(
                f"invalid group assignment at {manifest}:{line_number}: {payload}"
            )
        previous = assignments.setdefault(group, split)
        if previous != split:
            raise DatasetConfigError(
                f"split_group {group} is assigned to both {previous} and {split}: {manifest}"
            )
    missing = [split for split in REQUIRED_SPLITS if split not in set(assignments.values())]
    if missing:
        raise DatasetConfigError(f"group manifest has no {', '.join(missing)} split: {manifest}")


def environment_versions() -> dict[str, str]:
    import cv2
    import numpy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return destination


def train_detector(
    config: DatasetConfig,
    out_dir: str | Path,
    seed: int,
    weights: str | Path,
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
    trainer: Callable[..., Any] | None = None,
) -> Path:
    if not config.labels_reviewed:
        raise DatasetConfigError("refusing to train on unreviewed labels")
    validate_group_splits(config.group_manifest)
    weights_path = Path(weights)
    if trainer is None and not weights_path.is_file():
        raise FileNotFoundError(f"local starting weights not found: {weights_path}")

    destination = Path(out_dir)
    arguments = {
        "data": str(config.data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "device": device,
        "seed": int(seed),
        "project": str(destination),
        "name": "train",
        "deterministic": True,
        "pretrained": str(weights_path),
    }
    run = trainer if trainer is not None else _ultralytics_trainer(weights_path)
    result = run(**arguments)
    return write_json(
        destination / "training-run.json",
        {
            "arguments": arguments,
            "config_path": str(config.config_path),
            "config_sha256": config.config_sha256,
            "data_yaml": str(config.data_yaml),
            "group_manifest": str(config.group_manifest),
            "class_names": list(config.class_names),
            "labels_reviewed": True,
            "seed": int(seed),
            "weights": str(weights_path),
            "weights_sha256": file_sha256(weights_path) if weights_path.is_file() else None,
            "versions": environment_versions(),
            "finished_at": datetime.now(UTC).isoformat(),
            "result": getattr(result, "save_dir", result),
        },
    )


def _ultralytics_trainer(weights_path: Path) -> Callable[..., Any]:
    def run(**kwargs: Any) -> Any:
        from ultralytics import YOLO  # lazy: local training dependency only

        return YOLO(str(weights_path)).train(**kwargs)

    return run
