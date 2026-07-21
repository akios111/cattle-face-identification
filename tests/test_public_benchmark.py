from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from cattle_id.public_benchmark import (
    assign_closed_set_splits,
    build_external_public_benchmark_metadata,
    scan_cattely_benchmark,
)


def _image(path: Path, color: tuple[int, int, int] = (90, 120, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _cattely_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "Cattely"
    for animal in ("Cattle_2", "Cattle_10"):
        for index in range(5):
            _image(root / animal / f"{animal}_{index}.jpg", (index * 10, 80, 120))
    _image(root / "Extra_Folder_1" / "ignore.jpg")
    (root / "Cattle_2" / "notes.txt").write_text("not an image", encoding="utf-8")
    return root


def test_scan_cattely_benchmark_builds_identity_metadata_and_ignores_extra_folders(tmp_path):
    root = _cattely_fixture(tmp_path)

    metadata = scan_cattely_benchmark(root, source_url="https://github.com/example/cattely")

    assert len(metadata) == 10
    assert metadata["animal_id"].unique().tolist() == ["Cattle_2", "Cattle_10"]
    assert metadata.groupby("animal_id")["class_id"].first().to_dict() == {
        "Cattle_2": 0,
        "Cattle_10": 1,
    }
    assert metadata["image_path"].map(lambda value: Path(value).is_absolute()).all()
    assert metadata["external_dataset"].unique().tolist() == ["cattely"]
    assert metadata["protocol"].unique().tolist() == ["external_public_face_benchmark"]
    assert metadata["source_url"].unique().tolist() == ["https://github.com/example/cattely"]


def test_scan_cattely_benchmark_accepts_public_identity_ids_and_ignores_detection_valid(tmp_path):
    root = tmp_path / "Cattely"
    for animal in ("s114", "n2005"):
        for index in range(3):
            _image(root / animal / f"{animal}_{index}.jpg")
    _image(root / "valid" / "images" / "detection.jpg")

    metadata = scan_cattely_benchmark(root, source_url="https://github.com/example/cattely")

    assert set(metadata["animal_id"]) == {"s114", "n2005"}
    assert "valid" not in set(metadata["animal_id"])
    assert len(metadata) == 6


def test_assign_closed_set_splits_is_deterministic_and_keeps_each_animal_in_all_splits(tmp_path):
    metadata = scan_cattely_benchmark(_cattely_fixture(tmp_path), source_url="https://github.com/example/cattely")

    first = assign_closed_set_splits(metadata, seed=7, train_ratio=0.6, validation_ratio=0.2)
    second = assign_closed_set_splits(metadata, seed=7, train_ratio=0.6, validation_ratio=0.2)

    assert first["split"].tolist() == second["split"].tolist()
    counts = first.groupby(["animal_id", "split"]).size().unstack(fill_value=0)
    assert counts.loc["Cattle_2"].to_dict() == {"test": 1, "train": 3, "validation": 1}
    assert counts.loc["Cattle_10"].to_dict() == {"test": 1, "train": 3, "validation": 1}


def test_assign_closed_set_splits_marks_identities_with_too_few_images_as_excluded(tmp_path):
    root = tmp_path / "Cattely"
    _image(root / "s1806" / "only.jpg")

    metadata = scan_cattely_benchmark(root, source_url="https://github.com/example/cattely")
    assigned = assign_closed_set_splits(metadata)

    assert assigned["split"].tolist() == ["excluded_low_image_count"]
    assert "too few images" in assigned["notes"].iloc[0]


def test_build_external_public_benchmark_metadata_writes_split_csv(tmp_path):
    root = _cattely_fixture(tmp_path)
    output_path = tmp_path / "metadata.csv"

    written = build_external_public_benchmark_metadata(
        root,
        output_path,
        source_url="https://github.com/example/cattely",
        seed=11,
    )

    frame = pd.read_csv(written)
    assert written == output_path
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame["protocol"].unique().tolist() == ["external_public_face_benchmark"]
