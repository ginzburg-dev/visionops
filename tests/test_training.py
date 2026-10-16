import json

import pytest

from visionops.training import (
    DatasetConfigError,
    load_dataset_config,
    train_detector,
    validate_group_splits,
)

DATA_YAML = """
path: .
train: images/train
val: images/val
test: images/test
names:
  0: person
"""


def _dataset(tmp_path, *, reviewed=True, data_yaml=DATA_YAML, groups=None):
    (tmp_path / "data.yaml").write_text(data_yaml, encoding="utf-8")
    assignments = (
        groups
        if groups is not None
        else [
            {"split_group": "a", "split": "train"},
            {"split_group": "b", "split": "val"},
            {"split_group": "c", "split": "test"},
        ]
    )
    (tmp_path / "groups.jsonl").write_text(
        "\n".join(json.dumps(item) for item in assignments) + "\n", encoding="utf-8"
    )
    marker = "true" if reviewed else "false"
    (tmp_path / "dataset.yaml").write_text(
        f"data_yaml: data.yaml\ngroup_manifest: groups.jsonl\nlabels_reviewed: {marker}\n",
        encoding="utf-8",
    )
    return tmp_path / "dataset.yaml"


def test_rejects_missing_reviewed_label_marker(tmp_path):
    with pytest.raises(DatasetConfigError, match="labels_reviewed"):
        load_dataset_config(_dataset(tmp_path, reviewed=False))


def test_rejects_unreviewed_prediction_paths(tmp_path):
    data_yaml = DATA_YAML.replace("images/train", "runs/predictions/train")
    with pytest.raises(DatasetConfigError, match="unreviewed model output"):
        load_dataset_config(_dataset(tmp_path, data_yaml=data_yaml))


def test_rejects_absent_test_split(tmp_path):
    data_yaml = DATA_YAML.replace("test: images/test\n", "")
    with pytest.raises(DatasetConfigError, match="missing the test split"):
        load_dataset_config(_dataset(tmp_path, data_yaml=data_yaml))


def test_rejects_group_in_two_splits(tmp_path):
    groups = [
        {"split_group": "a", "split": "train"},
        {"split_group": "a", "split": "test"},
        {"split_group": "b", "split": "val"},
    ]
    config_path = _dataset(tmp_path, groups=groups)
    with pytest.raises(DatasetConfigError, match="assigned to both"):
        validate_group_splits(load_dataset_config(config_path).group_manifest)


def test_training_stops_before_calling_the_trainer_on_split_leakage(tmp_path):
    groups = [
        {"split_group": "a", "split": "train"},
        {"split_group": "a", "split": "val"},
        {"split_group": "b", "split": "test"},
    ]
    config = load_dataset_config(_dataset(tmp_path, groups=groups))
    calls = []
    with pytest.raises(DatasetConfigError):
        train_detector(config, tmp_path / "runs", 1, tmp_path / "w.pt", trainer=calls.append)
    assert calls == []


def test_trainer_receives_exact_arguments_and_run_is_recorded(tmp_path):
    config = load_dataset_config(_dataset(tmp_path))
    seen = {}

    def trainer(**kwargs):
        seen.update(kwargs)
        return type("Result", (), {"save_dir": str(tmp_path / "runs" / "train")})()

    out = train_detector(
        config,
        tmp_path / "runs",
        seed=11,
        weights=tmp_path / "weights.pt",
        epochs=3,
        imgsz=320,
        device="cpu",
        trainer=trainer,
    )

    assert seen["data"] == str(tmp_path / "data.yaml")
    assert seen["epochs"] == 3
    assert seen["imgsz"] == 320
    assert seen["seed"] == 11
    assert seen["deterministic"] is True
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed"] == 11
    assert payload["labels_reviewed"] is True
    assert payload["config_sha256"] == config.config_sha256
    assert payload["versions"]["python"]
    assert payload["result"].endswith("train")
