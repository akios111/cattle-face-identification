from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cattle_id.holstein2025 import (
    HOLSTEIN2025_PROTOCOL,
    build_holstein2025_metadata,
    scan_holstein2025,
    validate_holstein2025_metadata,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color).save(path)


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "Holstein2025"
    for identity_index, animal_id in enumerate(("train_a", "train_b")):
        for image_index in range(5):
            _image(
                root / "datasets_v2" / animal_id / f"{image_index}.jpg",
                (identity_index * 70, image_index * 20, 90),
            )
    for image_index in range(2):
        _image(root / "gallery1" / "open_a" / f"g{image_index}.jpg", (10, image_index * 30, 20))
    for image_index in range(3):
        _image(root / "query1" / "open_a" / f"q{image_index}.jpg", (20, image_index * 30, 10))
    return root


def test_scan_holstein2025_preserves_official_open_set_split_and_hashes(tmp_path: Path):
    metadata = scan_holstein2025(
        _fixture(tmp_path),
        source_url="https://github.com/example/holstein2025",
        source_commit="abc123",
        seed=7,
        validation_ratio=0.2,
    )

    assert len(metadata) == 15
    assert set(metadata["split"]) == {"train", "validation", "gallery", "probe"}
    assert set(metadata.loc[metadata["split"] == "gallery", "animal_id"]) == {"open_a"}
    assert set(metadata.loc[metadata["split"] == "probe", "animal_id"]) == {"open_a"}
    assert metadata.groupby(["animal_id", "split"]).size().loc[("train_a", "validation")] == 1
    assert metadata["image_path"].map(lambda value: Path(value).is_absolute()).all()
    assert metadata["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert metadata["protocol"].unique().tolist() == [HOLSTEIN2025_PROTOCOL]
    assert metadata["source_commit"].unique().tolist() == ["abc123"]


def test_validate_holstein2025_metadata_reports_no_identity_or_hash_leakage(tmp_path: Path):
    metadata = scan_holstein2025(
        _fixture(tmp_path),
        source_url="https://github.com/example/holstein2025",
        source_commit="abc123",
    )

    summary = validate_holstein2025_metadata(
        metadata,
        expected_development_identities=2,
        expected_open_set_identities=1,
        expected_images=15,
    )

    assert summary["data_integrity_ready"] is True
    assert summary["development_identities"] == 2
    assert summary["open_set_identities"] == 1
    assert summary["identity_overlap"] == 0
    assert summary["duplicate_hashes"] == 0
    assert summary["issues"] == []


def test_validate_holstein2025_metadata_rejects_open_identity_in_development(tmp_path: Path):
    metadata = scan_holstein2025(
        _fixture(tmp_path),
        source_url="https://github.com/example/holstein2025",
        source_commit="abc123",
    )
    gallery_index = metadata.index[metadata["split"] == "gallery"][0]
    metadata.loc[gallery_index, "animal_id"] = "train_a"

    summary = validate_holstein2025_metadata(metadata)

    assert summary["data_integrity_ready"] is False
    assert summary["identity_overlap"] == 1
    assert any("identity leakage" in issue for issue in summary["issues"])


def test_validate_holstein2025_metadata_rejects_gallery_probe_identity_mismatch(tmp_path: Path):
    metadata = scan_holstein2025(
        _fixture(tmp_path),
        source_url="https://github.com/example/holstein2025",
        source_commit="abc123",
    )
    probe_index = metadata.index[metadata["split"] == "probe"][0]
    metadata.loc[probe_index, "animal_id"] = "different_open_id"

    summary = validate_holstein2025_metadata(metadata)

    assert summary["data_integrity_ready"] is False
    assert any("gallery/probe identity sets differ" in issue for issue in summary["issues"])


def test_build_holstein2025_metadata_writes_metadata_and_hash_manifest(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    hashes_path = tmp_path / "hashes.csv"

    summary = build_holstein2025_metadata(
        _fixture(tmp_path),
        metadata_path,
        hashes_path,
        source_url="https://github.com/example/holstein2025",
        source_commit="abc123",
        expected_development_identities=2,
        expected_open_set_identities=1,
        expected_images=15,
    )

    metadata = pd.read_csv(metadata_path)
    hashes = pd.read_csv(hashes_path)
    assert summary["data_integrity_ready"] is True
    assert len(metadata) == 15
    assert hashes.columns.tolist() == ["relative_path", "sha256"]
    assert len(hashes) == 15


def test_scan_holstein2025_requires_gallery_and_probe_directories(tmp_path: Path):
    root = tmp_path / "Holstein2025"
    _image(root / "datasets_v2" / "train_a" / "0.jpg", (1, 2, 3))

    with pytest.raises(FileNotFoundError, match="gallery1"):
        scan_holstein2025(root, source_url="https://example.test", source_commit="abc")
