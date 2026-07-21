from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cattle_id.data import (
    build_external_acquisition_metadata,
    validate_external_acquisition_manifest,
)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 80, 40)).save(path)


def _manifest(tmp_path: Path) -> pd.DataFrame:
    image_a = tmp_path / "external" / "animal-001-a.jpg"
    image_b = tmp_path / "external" / "animal-001-b.jpg"
    _image(image_a)
    _image(image_b)
    return pd.DataFrame(
        [
            {
                "image_path": str(image_a),
                "class_id": 1,
                "animal_id": "animal-001",
                "acquisition_id": "external-day-1-a",
                "capture_date": "2026-07-07",
                "camera_id": "phone-a",
                "lighting": "shade",
                "pose": "front",
                "operator": "operator-a",
                "split": "test",
                "notes": "new acquisition",
            },
            {
                "image_path": str(image_b),
                "class_id": 1,
                "animal_id": "animal-001",
                "acquisition_id": "external-day-1-b",
                "capture_date": "2026-07-07",
                "camera_id": "phone-a",
                "lighting": "sun",
                "pose": "three-quarter",
                "operator": "operator-a",
                "split": "test",
                "notes": "new acquisition",
            },
        ]
    )


def test_external_manifest_accepts_required_schema_and_test_only_split(tmp_path):
    manifest = _manifest(tmp_path)

    validated = validate_external_acquisition_manifest(manifest, allowed_class_ids={1, 2})

    assert validated["split"].tolist() == ["test", "test"]
    assert validated["protocol"].tolist() == [
        "external_acquisition_holdout",
        "external_acquisition_holdout",
    ]
    assert validated["image_path"].map(lambda value: Path(value).is_absolute()).all()


def test_external_manifest_rejects_missing_columns(tmp_path):
    manifest = _manifest(tmp_path).drop(columns=["camera_id"])

    with pytest.raises(ValueError, match="missing columns.*camera_id"):
        validate_external_acquisition_manifest(manifest, allowed_class_ids={1})


def test_external_manifest_rejects_duplicate_image_paths(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[1, "image_path"] = manifest.loc[0, "image_path"]

    with pytest.raises(ValueError, match="duplicate image_path"):
        validate_external_acquisition_manifest(manifest, allowed_class_ids={1})


def test_external_manifest_rejects_class_ids_not_in_training_map(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[1, "class_id"] = 99

    with pytest.raises(ValueError, match="unknown class_id"):
        validate_external_acquisition_manifest(manifest, allowed_class_ids={1})


def test_external_manifest_rejects_train_or_validation_rows(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[0, "split"] = "train"

    with pytest.raises(ValueError, match="external acquisition rows must use split=test"):
        validate_external_acquisition_manifest(manifest, allowed_class_ids={1})


def test_external_manifest_rejects_acquisition_overlap_with_training_metadata(tmp_path):
    manifest = _manifest(tmp_path)
    training_metadata = pd.DataFrame(
        [{"source_file": "animal-001-a.jpg", "acquisition_id": "external-day-1-a"}]
    )

    with pytest.raises(ValueError, match="acquisition leakage"):
        validate_external_acquisition_manifest(
            manifest,
            allowed_class_ids={1},
            training_metadata=training_metadata,
        )


def test_build_external_acquisition_metadata_reads_csv_and_writes_validated_csv(tmp_path):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "external_manifest.csv"
    output_path = tmp_path / "metadata_external.csv"
    manifest.to_csv(manifest_path, index=False)

    written = build_external_acquisition_metadata(
        manifest_path,
        output_path=output_path,
        allowed_class_ids={1},
    )

    frame = pd.read_csv(written)
    assert written == output_path
    assert frame["protocol"].unique().tolist() == ["external_acquisition_holdout"]
    assert frame["augmentation_id"].unique().tolist() == ["external_real"]
