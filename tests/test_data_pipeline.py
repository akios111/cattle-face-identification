from pathlib import Path

from PIL import Image
import pytest

from cattle_id.data import (
    assign_splits,
    build_augmented_metadata,
    build_source_manifest,
    materialize_augmented_images,
)


def _make_sources(tmp_path: Path, count: int = 311) -> Path:
    image_dir = tmp_path / "cattle_images"
    image_dir.mkdir()
    for idx in range(count):
        image = Image.new(
            "RGB",
            (16, 16),
            ((idx * 3) % 255, (idx * 5) % 255, (idx * 7) % 255),
        )
        image.save(image_dir / f"IMG_{idx:04d}.png")
    return image_dir


def test_source_manifest_assigns_sequential_classes_from_sorted_filenames(tmp_path):
    image_dir = _make_sources(tmp_path, count=311)

    manifest = build_source_manifest(image_dir)

    assert len(manifest) == 311
    assert manifest["class_id"].tolist() == list(range(311))
    assert manifest.iloc[0]["source_file"] == "IMG_0000.png"
    assert manifest.iloc[-1]["source_file"] == "IMG_0310.png"


def test_full_augmentation_metadata_matches_cattlessfr_paper_size(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=311))

    metadata = build_augmented_metadata(manifest, profile="all")

    assert len(metadata) == 6220
    assert metadata["class_id"].nunique() == 311
    assert metadata.groupby("source_file")["augmentation_id"].nunique().eq(20).all()


def test_paper_random_split_matches_reported_counts_and_keeps_all_classes(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=311))
    metadata = build_augmented_metadata(manifest, profile="all")

    split = assign_splits(metadata, protocol="paper_random", seed=2026)

    assert split["split"].value_counts().to_dict() == {
        "train": 3918,
        "test": 1866,
        "validation": 436,
    }
    assert split.groupby("split")["class_id"].nunique().to_dict() == {
        "test": 311,
        "train": 311,
        "validation": 311,
    }


def test_transform_holdout_uses_disjoint_augmentation_families(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=4))
    metadata = build_augmented_metadata(manifest, profile="all")

    split = assign_splits(metadata, protocol="transform_holdout", seed=2026)

    ids_by_split = {
        name: set(group["augmentation_id"])
        for name, group in split.groupby("split", sort=False)
    }
    assert ids_by_split["train"].isdisjoint(ids_by_split["validation"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["validation"].isdisjoint(ids_by_split["test"])
    assert split.groupby("split")["class_id"].nunique().to_dict() == {
        "test": 4,
        "train": 4,
        "validation": 4,
    }


def test_paper_random_rejects_unestimable_single_sample_per_class(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=4))
    metadata = build_augmented_metadata(manifest, profile="none")

    with pytest.raises(ValueError, match="at least 3 samples per class"):
        assign_splits(metadata, protocol="paper_random", seed=1)


def test_small_ablation_profile_gets_one_or_more_samples_per_class_in_every_split(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=4))
    metadata = build_augmented_metadata(manifest, profile="cutout_only")

    first = assign_splits(metadata, protocol="paper_random", seed=1)
    second = assign_splits(metadata, protocol="paper_random", seed=1)

    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("split")["class_id"].nunique().to_dict() == {
        "test": 4,
        "train": 4,
        "validation": 4,
    }
    assert first["split"].value_counts().to_dict() == {
        "train": 4,
        "validation": 4,
        "test": 4,
    }


def test_hardening_materialization_records_source_and_image_byte_hashes(tmp_path):
    manifest = build_source_manifest(_make_sources(tmp_path, count=2), include_sha256=True)
    metadata = build_augmented_metadata(manifest, profile="all")
    metadata = metadata[metadata["augmentation_id"].isin(["original", "gaussian_noise"])]
    metadata = assign_splits(metadata, protocol="transform_holdout", seed=1)

    materialized = materialize_augmented_images(
        metadata,
        tmp_path / "processed",
        image_size=(24, 24),
        seed=1,
        protocol_version="hardening_v2",
        materialization_id="fixture_v2",
    )

    assert materialized["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert materialized["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(materialized["protocol_version"]) == {"hardening_v2"}
    assert set(materialized["materialization_id"]) == {"fixture_v2"}
    noisy = materialized[materialized["augmentation_id"] == "gaussian_noise"]
    assert noisy["image_sha256"].nunique() == 2
