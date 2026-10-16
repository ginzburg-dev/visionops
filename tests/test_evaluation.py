import json

import pytest

from tests.test_training import _dataset
from visionops.evaluation import evaluate_detector
from visionops.training import load_dataset_config


class _Metrics:
    mp = 0.81
    mr = 0.74
    map50 = 0.78
    map = 0.52


def test_evaluator_receives_test_split_and_metrics_are_stored(tmp_path):
    config = load_dataset_config(_dataset(tmp_path))
    seen = {}

    def evaluator(**kwargs):
        seen.update(kwargs)
        return _Metrics()

    out = evaluate_detector(config, tmp_path / "model.pt", tmp_path / "eval", evaluator)

    assert seen == {"data": str(tmp_path / "data.yaml"), "split": "test"}
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"] == {
        "precision": 0.81,
        "recall": 0.74,
        "map50": 0.78,
        "map50_95": 0.52,
    }
    assert payload["split"] == "test"
    assert payload["config_sha256"] == config.config_sha256


def test_missing_metric_is_reported(tmp_path):
    config = load_dataset_config(_dataset(tmp_path))
    with pytest.raises(ValueError, match="missing metric"):
        evaluate_detector(config, tmp_path / "m.pt", tmp_path / "eval", lambda **kwargs: object())


def test_missing_model_without_injection_is_reported(tmp_path):
    config = load_dataset_config(_dataset(tmp_path))
    with pytest.raises(FileNotFoundError, match="model weights not found"):
        evaluate_detector(config, tmp_path / "absent.pt", tmp_path / "eval")
