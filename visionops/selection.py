from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from visionops.schemas import PredictionRecord, SelectionRecord

HASH_BITS = 64
DUPLICATE_DISTANCE = 4
EMPTY_SCENE_UNCERTAINTY = 0.75
UNCERTAINTY_WEIGHT = 0.7
DIVERSITY_WEIGHT = 0.3
STRATEGIES = ("random", "uncertainty")


def dhash(image_path: str | Path) -> int:
    """Return a 64-bit difference hash of an image."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"could not decode image for hashing: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int32)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def frame_uncertainty(record: PredictionRecord) -> float:
    """Confidence ambiguity around 0.5; empty scenes are worth reviewing on their own."""
    if not record.detections:
        return EMPTY_SCENE_UNCERTAINTY
    scores = [1.0 - 2.0 * abs(item.confidence - 0.5) for item in record.detections]
    return float(min(max(sum(scores) / len(scores), 0.0), 1.0))


def _image_hash(record: PredictionRecord, root: Path | None) -> int:
    candidate = Path(record.image.image_path)
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    if candidate.is_file():
        return dhash(candidate)
    return int(record.image.content_sha256[:16], 16)


def _deduplicate(
    records: Sequence[PredictionRecord], root: Path | None
) -> list[tuple[PredictionRecord, int]]:
    kept: list[tuple[PredictionRecord, int]] = []
    for record in records:
        value = _image_hash(record, root)
        if any(hamming_distance(value, other) <= DUPLICATE_DISTANCE for _, other in kept):
            continue
        kept.append((record, value))
    return kept


def _min_distance(value: int, selected: Sequence[int]) -> int:
    if not selected:
        return HASH_BITS
    return min(hamming_distance(value, other) for other in selected)


def select_records(
    predictions: Sequence[PredictionRecord],
    budget: int,
    strategy: str,
    seed: int,
    root: str | Path | None = None,
) -> list[SelectionRecord]:
    if budget <= 0:
        raise ValueError("budget must be greater than zero")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown selection strategy: {strategy!r}")
    base = Path(root) if root is not None else None
    candidates = _deduplicate(predictions, base)

    if strategy == "random":
        order = list(candidates)
        random.Random(seed).shuffle(order)
        ordered = order[:budget]
    else:
        remaining = sorted(
            candidates, key=lambda item: (-frame_uncertainty(item[0]), item[0].image.image_id)
        )
        ordered = []
        chosen_hashes: list[int] = []
        while remaining and len(ordered) < budget:
            if not ordered:
                best = remaining[0]
            else:
                best = max(
                    remaining,
                    key=lambda item: (
                        UNCERTAINTY_WEIGHT * frame_uncertainty(item[0])
                        + DIVERSITY_WEIGHT * _min_distance(item[1], chosen_hashes) / HASH_BITS
                    ),
                )
            remaining.remove(best)
            ordered.append(best)
            chosen_hashes.append(best[1])

    results: list[SelectionRecord] = []
    seen: list[int] = []
    for rank, (record, value) in enumerate(ordered, start=1):
        results.append(
            SelectionRecord(
                prediction=record,
                rank=rank,
                strategy=strategy,
                uncertainty_score=frame_uncertainty(record),
                diversity_score=_min_distance(value, seen) / HASH_BITS,
            )
        )
        seen.append(value)
    return results
