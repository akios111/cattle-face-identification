from pathlib import Path

import pytest
import yaml

from cattle_id.colab_runtime import write_holstein_runtime_configs


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_write_holstein_runtime_configs_only_overrides_dataset_root(tmp_path):
    readiness_source = tmp_path / "project" / "configs" / "holstein2025_open_set.yaml"
    training_source = tmp_path / "project" / "configs" / "holstein2025_colab_proplus.yaml"
    readiness_payload = {
        "dataset": {
            "name": "holstein2025",
            "raw_dir": "data/external_public/Holstein2025",
            "source_commit": "pinned-commit",
        },
        "outputs": {"metadata": "artifacts/metadata/holstein2025_open_set.csv"},
    }
    training_payload = {
        "dataset": {"name": "holstein2025", "raw_dir": "data/external_public/Holstein2025"},
        "training": {"epochs": 35},
        "output": {"run_dir": "artifacts/runs"},
    }
    _write_yaml(readiness_source, readiness_payload)
    _write_yaml(training_source, training_payload)
    readiness_before = readiness_source.read_bytes()
    training_before = training_source.read_bytes()

    dataset_root = tmp_path / "content" / "Holstein2025"
    result = write_holstein_runtime_configs(
        readiness_source=readiness_source,
        training_source=training_source,
        runtime_dir=tmp_path / "content" / "runtime-configs",
        dataset_root=dataset_root,
    )

    readiness_runtime = yaml.safe_load(result.readiness.read_text(encoding="utf-8"))
    training_runtime = yaml.safe_load(result.training.read_text(encoding="utf-8"))
    expected_root = str(dataset_root.resolve())
    assert readiness_runtime["dataset"]["raw_dir"] == expected_root
    assert training_runtime["dataset"]["raw_dir"] == expected_root
    assert readiness_runtime["outputs"] == readiness_payload["outputs"]
    assert training_runtime["output"] == training_payload["output"]
    assert training_runtime["training"] == training_payload["training"]
    assert readiness_source.read_bytes() == readiness_before
    assert training_source.read_bytes() == training_before


@pytest.mark.parametrize("missing_source", ["readiness", "training"])
def test_write_holstein_runtime_configs_rejects_missing_dataset_mapping(tmp_path, missing_source):
    readiness_source = tmp_path / "readiness.yaml"
    training_source = tmp_path / "training.yaml"
    _write_yaml(readiness_source, {"dataset": {"raw_dir": "old"}})
    _write_yaml(training_source, {"dataset": {"raw_dir": "old"}})
    source = readiness_source if missing_source == "readiness" else training_source
    _write_yaml(source, {"not_dataset": {}})

    with pytest.raises(ValueError, match="dataset mapping"):
        write_holstein_runtime_configs(
            readiness_source=readiness_source,
            training_source=training_source,
            runtime_dir=tmp_path / "runtime",
            dataset_root=tmp_path / "dataset",
        )
