from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cattle_id.evaluate import (
    _prediction_export_frame,
    evaluation_output_paths,
    evaluation_sample_set_sha256,
    load_evaluation_metadata,
)


def test_evaluation_output_paths_preserve_default_names(tmp_path: Path):
    paths = evaluation_output_paths(tmp_path)

    assert paths.metrics == tmp_path / "metrics.json"
    assert paths.confusion_matrix == tmp_path / "confusion_matrix.csv"
    assert paths.predictions == tmp_path / "predictions.csv"


def test_evaluation_output_paths_add_suffix_for_external_or_masked_runs(tmp_path: Path):
    paths = evaluation_output_paths(tmp_path, output_suffix="external_holdout")

    assert paths.metrics == tmp_path / "metrics_external_holdout.json"
    assert paths.confusion_matrix == tmp_path / "confusion_matrix_external_holdout.csv"
    assert paths.predictions == tmp_path / "predictions_external_holdout.csv"


def test_evaluation_sample_set_hash_is_order_independent_and_content_sensitive():
    frame = pd.DataFrame(
        [
            {"source_file": "a.jpg", "class_id": 0, "augmentation_id": "original", "split": "test"},
            {"source_file": "b.jpg", "class_id": 1, "augmentation_id": "blur", "split": "test"},
        ]
    )

    first, columns = evaluation_sample_set_sha256(frame)
    reordered, _ = evaluation_sample_set_sha256(frame.iloc[::-1])
    changed, _ = evaluation_sample_set_sha256(frame.assign(augmentation_id=["flip", "blur"]))

    assert first == reordered
    assert first != changed
    assert columns == ["source_file", "class_id", "augmentation_id", "split"]


def test_load_evaluation_metadata_uses_override_and_split_filter(tmp_path: Path):
    training_metadata = tmp_path / "training.csv"
    override_metadata = tmp_path / "external.csv"
    pd.DataFrame(
        [
            {"image_path": "train.png", "class_id": 0, "split": "test"},
        ]
    ).to_csv(training_metadata, index=False)
    pd.DataFrame(
        [
            {"image_path": "external-test.png", "class_id": 1, "split": "test"},
            {"image_path": "external-val.png", "class_id": 1, "split": "val"},
        ]
    ).to_csv(override_metadata, index=False)
    manifest = {"metadata_path": str(training_metadata)}

    loaded = load_evaluation_metadata(manifest, split="test", metadata_path=override_metadata)

    assert loaded["image_path"].tolist() == ["external-test.png"]


def test_load_evaluation_metadata_defaults_to_manifest_metadata(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("image_path,class_id,split\nsample.png,0,test\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata_path": str(metadata_path)}), encoding="utf-8")

    loaded = load_evaluation_metadata(json.loads(manifest_path.read_text(encoding="utf-8")), split="test")

    assert loaded["image_path"].tolist() == ["sample.png"]


def test_prediction_export_preserves_robustness_group_columns():
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "cow-1::severity::blur_pos_0.7",
                "image_path": "sample.png",
                "class_id": 0,
                "split": "test",
                "region_variant": "central_region_only",
                "severity_family": "blur",
                "severity_value": 0.7,
                "severity_direction": 1,
            }
        ]
    )

    exported = _prediction_export_frame(metadata)

    assert exported["region_variant"].tolist() == ["central_region_only"]
    assert exported["severity_family"].tolist() == ["blur"]
    assert exported["severity_value"].tolist() == [0.7]
    assert exported["severity_direction"].tolist() == [1]
