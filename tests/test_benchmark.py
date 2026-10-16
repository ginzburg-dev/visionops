import numpy as np
import pytest

from tests.fixtures import write_image
from visionops.benchmark import benchmark_model


class _Detector:
    model_id = "person.onnx"
    model_sha256 = "f" * 64
    input_size = 640
    providers = ["CPUExecutionProvider"]

    def __init__(self):
        self.calls = 0

    def detect(self, image_bgr):
        self.calls += 1
        return []


def test_warmup_runs_are_excluded_from_statistics(tmp_path):
    write_image(tmp_path / "a.png", 10)
    detector = _Detector()

    report = benchmark_model(detector, [tmp_path / "a.png"], warmup=4, iterations=6)

    assert detector.calls == 10
    assert report["samples"] == 6
    assert report["warmup"] == 4
    assert report["images"] == 1


def test_report_contains_latency_throughput_and_provenance(tmp_path):
    write_image(tmp_path / "a.png", 10)
    write_image(tmp_path / "b.png", 20)

    report = benchmark_model(_Detector(), [tmp_path / "a.png", tmp_path / "b.png"], 1, 5)

    latency = report["latency_ms"]
    assert set(latency) == {"mean", "p50", "p95", "min", "max"}
    assert latency["min"] <= latency["p50"] <= latency["p95"] <= latency["max"]
    assert report["throughput_fps"] > 0
    assert report["model_sha256"] == "f" * 64
    assert report["providers"] == ["CPUExecutionProvider"]
    assert report["versions"]["python"]
    assert report["versions"]["platform"]
    assert report["images"] == 2


def test_arrays_are_accepted_and_empty_input_is_rejected():
    report = benchmark_model(_Detector(), [np.zeros((8, 8, 3), dtype=np.uint8)], 0, 2)
    assert report["samples"] == 2
    with pytest.raises(ValueError, match="at least one benchmark image"):
        benchmark_model(_Detector(), [], 0, 1)


def test_undecodable_benchmark_image_is_reported(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"nope")
    with pytest.raises(ValueError, match="could not decode benchmark image"):
        benchmark_model(_Detector(), [broken], 0, 1)
