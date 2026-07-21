from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cattle_id.masking import apply_masking_annotations


def _sample_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (200, 160, 120)).save(path)


def _metadata(tmp_path: Path) -> pd.DataFrame:
    image_path = tmp_path / "input" / "animal-001.jpg"
    _sample_image(image_path)
    return pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "class_id": 1,
                "animal_id": "animal-001",
                "augmentation_id": "external_real",
                "split": "test",
            }
        ]
    )


def _annotations(image_path: str, *, ear_tag: bool = True) -> pd.DataFrame:
    ear_values = (1, 1, 6, 6) if ear_tag else ("", "", "", "")
    return pd.DataFrame(
        [
            {
                "image_path": image_path,
                "ear_tag_x1": ear_values[0],
                "ear_tag_y1": ear_values[1],
                "ear_tag_x2": ear_values[2],
                "ear_tag_y2": ear_values[3],
                "paint_mark_x1": "",
                "paint_mark_y1": "",
                "paint_mark_x2": "",
                "paint_mark_y2": "",
                "face_bbox_x1": 2,
                "face_bbox_y1": 2,
                "face_bbox_x2": 18,
                "face_bbox_y2": 18,
                "background_x1": 0,
                "background_y1": 0,
                "background_x2": 20,
                "background_y2": 20,
            }
        ]
    )


def test_masking_preserves_identity_columns_and_writes_deterministic_outputs(tmp_path):
    metadata = _metadata(tmp_path)
    annotations = _annotations(metadata.loc[0, "image_path"])
    out_dir = tmp_path / "masked"

    first = apply_masking_annotations(metadata, annotations, out_dir, variant="mask_ear_tag")
    second = apply_masking_annotations(metadata, annotations, out_dir, variant="mask_ear_tag")

    assert first[["class_id", "animal_id", "split"]].to_dict("records") == [
        {"class_id": 1, "animal_id": "animal-001", "split": "test"}
    ]
    assert first.loc[0, "mask_status"] == "applied"
    assert first.loc[0, "augmentation_id"] == "mask_ear_tag"
    assert Path(first.loc[0, "image_path"]).read_bytes() == Path(second.loc[0, "image_path"]).read_bytes()


def test_masking_marks_absent_optional_region_as_not_present(tmp_path):
    metadata = _metadata(tmp_path)
    annotations = _annotations(metadata.loc[0, "image_path"], ear_tag=False)

    result = apply_masking_annotations(metadata, annotations, tmp_path / "masked", variant="mask_ear_tag")

    assert result.loc[0, "mask_status"] == "not_present"
    assert Path(result.loc[0, "image_path"]).exists()


def test_masking_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown masking variant"):
        apply_masking_annotations(_metadata(tmp_path), pd.DataFrame(), tmp_path / "masked", variant="mask_unknown")
