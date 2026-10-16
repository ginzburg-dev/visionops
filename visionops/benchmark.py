from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionops.training import environment_versions


def _load_images(images: Sequence[Any]) -> list[np.ndarray]:
    loaded: list[np.ndarray] = []
    for item in images:
        if isinstance(item, np.ndarray):
            loaded.append(item)
            continue
        path = Path(item)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"could not decode benchmark image: {path}")
        loaded.append(image)
    if not loaded:
        raise ValueError("at least one benchmark image is required")
    return loaded


def benchmark_model(
    detector: Any,
    images: Sequence[Any],
    warmup: int = 3,
    iterations: int = 20,
) -> dict[str, Any]:
    """Measure single-image CPU latency. Warm-up runs are excluded from every statistic."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    loaded = _load_images(images)

    for index in range(warmup):
        detector.detect(loaded[index % len(loaded)])

    durations_ms: list[float] = []
    person_counts: list[int] = []
    for index in range(iterations):
        image = loaded[index % len(loaded)]
        started = time.perf_counter()
        detections = detector.detect(image)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        person_counts.append(len(detections))

    ordered = sorted(durations_ms)
    mean_ms = statistics.fmean(durations_ms)
    return {
        "samples": len(durations_ms),
        "warmup": int(warmup),
        "images": len(loaded),
        "latency_ms": {
            "mean": mean_ms,
            "p50": float(np.percentile(ordered, 50)),
            "p95": float(np.percentile(ordered, 95)),
            "min": ordered[0],
            "max": ordered[-1],
        },
        "throughput_fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "person_counts": person_counts,
        "model_id": getattr(detector, "model_id", None),
        "model_sha256": getattr(detector, "model_sha256", None),
        "input_size": getattr(detector, "input_size", None),
        "providers": list(getattr(detector, "providers", []) or []),
        "versions": environment_versions(),
        "measured_at": datetime.now(UTC).isoformat(),
    }
