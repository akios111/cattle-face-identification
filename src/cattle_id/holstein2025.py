from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import IMAGE_EXTENSIONS


HOLSTEIN2025_PROTOCOL = "holstein2025_unseen_identity_reid"
HOLSTEIN2025_DATASET = "holstein2025"
LICENSE_STATUS = "repository_license_not_declared"


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _images(path: Path) -> list[Path]:
    return sorted(
        [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS],
        key=_natural_key,
    )


def _identity_directories(path: Path) -> list[Path]:
    return sorted([item for item in path.iterdir() if item.is_dir()], key=_natural_key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_splits(
    image_paths: list[Path],
    *,
    rng: np.random.Generator,
    validation_ratio: float,
) -> dict[Path, str]:
    if len(image_paths) < 2:
        raise ValueError("Each Holstein2025 development identity needs at least two images")
    validation_count = max(1, int(np.ceil(len(image_paths) * validation_ratio)))
    validation_count = min(validation_count, len(image_paths) - 1)
    order = rng.permutation(len(image_paths))
    validation_indices = set(order[:validation_count].tolist())
    return {
        image_path: "validation" if index in validation_indices else "train"
        for index, image_path in enumerate(image_paths)
    }


def scan_holstein2025(
    dataset_root: str | Path,
    *,
    source_url: str,
    source_commit: str,
    seed: int = 2026,
    validation_ratio: float = 0.2,
) -> pd.DataFrame:
    root = Path(dataset_root).resolve()
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    split_dirs = {
        "development": root / "datasets_v2",
        "gallery": root / "gallery1",
        "probe": root / "query1",
    }
    for split_dir in split_dirs.values():
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(f"Holstein2025 directory not found: {split_dir}")

    development_dirs = _identity_directories(split_dirs["development"])
    gallery_dirs = _identity_directories(split_dirs["gallery"])
    probe_dirs = _identity_directories(split_dirs["probe"])
    if not development_dirs or not gallery_dirs or not probe_dirs:
        raise FileNotFoundError("Holstein2025 identity folders are incomplete")

    development_ids = [path.name for path in development_dirs]
    open_ids = sorted({path.name for path in gallery_dirs + probe_dirs}, key=lambda value: _natural_key(Path(value)))
    class_mapping = {
        animal_id: class_id
        for class_id, animal_id in enumerate([*development_ids, *open_ids])
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    def append_row(image_path: Path, animal_id: str, split: str) -> None:
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "relative_path": image_path.relative_to(root).as_posix(),
                "source_file": image_path.name,
                "class_id": int(class_mapping[animal_id]),
                "animal_id": animal_id,
                "external_dataset": HOLSTEIN2025_DATASET,
                "protocol": HOLSTEIN2025_PROTOCOL,
                "split": split,
                "source_url": source_url,
                "source_commit": source_commit,
                "sha256": _sha256(image_path),
                "capture_date": "unknown",
                "camera_id": "fixed_side_view_surveillance",
                "lighting": "field_variable",
                "pose": "side_view_variable",
                "operator": "public_dataset",
                "license_status": LICENSE_STATUS,
                "notes": (
                    "Official Holstein2025 identification image. Open-set identities are disjoint "
                    "from development identities; repository has no explicit license file."
                ),
            }
        )

    for identity_dir in development_dirs:
        image_paths = _images(identity_dir)
        assignments = _development_splits(
            image_paths,
            rng=rng,
            validation_ratio=validation_ratio,
        )
        for image_path in image_paths:
            append_row(image_path, identity_dir.name, assignments[image_path])

    for split, identity_dirs in (("gallery", gallery_dirs), ("probe", probe_dirs)):
        for identity_dir in identity_dirs:
            for image_path in _images(identity_dir):
                append_row(image_path, identity_dir.name, split)

    if not rows:
        raise FileNotFoundError(f"No Holstein2025 images found in {root}")
    return pd.DataFrame(rows)


def validate_holstein2025_metadata(
    metadata: pd.DataFrame,
    *,
    expected_development_identities: int | None = None,
    expected_open_set_identities: int | None = None,
    expected_images: int | None = None,
) -> dict[str, object]:
    required = {
        "image_path",
        "relative_path",
        "class_id",
        "animal_id",
        "protocol",
        "split",
        "sha256",
        "license_status",
    }
    issues: list[str] = []
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        issues.append(f"metadata missing required columns: {', '.join(missing_columns)}")
        return {
            "data_integrity_ready": False,
            "license_declared": False,
            "redistribution_ready": False,
            "images": int(len(metadata)),
            "development_identities": 0,
            "open_set_identities": 0,
            "identity_overlap": 0,
            "duplicate_paths": 0,
            "duplicate_hashes": 0,
            "split_counts": {},
            "issues": issues,
        }

    unexpected_protocols = set(metadata["protocol"].astype(str)) - {HOLSTEIN2025_PROTOCOL}
    unexpected_splits = set(metadata["split"].astype(str)) - {"train", "validation", "gallery", "probe"}
    if unexpected_protocols:
        issues.append(f"unexpected protocol values: {sorted(unexpected_protocols)}")
    if unexpected_splits:
        issues.append(f"unexpected split values: {sorted(unexpected_splits)}")

    train_ids = set(metadata.loc[metadata["split"] == "train", "animal_id"].astype(str))
    validation_ids = set(metadata.loc[metadata["split"] == "validation", "animal_id"].astype(str))
    gallery_ids = set(metadata.loc[metadata["split"] == "gallery", "animal_id"].astype(str))
    probe_ids = set(metadata.loc[metadata["split"] == "probe", "animal_id"].astype(str))
    development_ids = train_ids | validation_ids
    open_ids = gallery_ids | probe_ids
    identity_overlap = len(development_ids & open_ids)
    duplicate_paths = int(metadata["image_path"].duplicated().sum())
    duplicate_hashes = int(metadata["sha256"].duplicated().sum())

    if train_ids != validation_ids:
        issues.append("train/validation identity sets differ")
    if gallery_ids != probe_ids:
        issues.append("gallery/probe identity sets differ")
    if identity_overlap:
        issues.append(f"identity leakage between development and open set: {identity_overlap}")
    if duplicate_paths:
        issues.append(f"duplicate image paths: {duplicate_paths}")
    if duplicate_hashes:
        issues.append(f"duplicate image hashes: {duplicate_hashes}")
    if metadata["image_path"].isna().any() or (metadata["image_path"].astype(str).str.strip() == "").any():
        issues.append("blank image paths")

    images = int(len(metadata))
    if expected_development_identities is not None and len(development_ids) != expected_development_identities:
        issues.append(
            f"development identity count mismatch: {len(development_ids)} != {expected_development_identities}"
        )
    if expected_open_set_identities is not None and len(open_ids) != expected_open_set_identities:
        issues.append(f"open-set identity count mismatch: {len(open_ids)} != {expected_open_set_identities}")
    if expected_images is not None and images != expected_images:
        issues.append(f"image count mismatch: {images} != {expected_images}")

    license_declared = not (metadata["license_status"].astype(str) == LICENSE_STATUS).any()
    data_ready = not issues
    return {
        "data_integrity_ready": data_ready,
        "license_declared": license_declared,
        "redistribution_ready": data_ready and license_declared,
        "images": images,
        "development_identities": len(development_ids),
        "open_set_identities": len(open_ids),
        "identity_overlap": identity_overlap,
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": duplicate_hashes,
        "split_counts": {
            str(split): int(count)
            for split, count in metadata["split"].value_counts().sort_index().items()
        },
        "issues": issues,
    }


def build_holstein2025_metadata(
    dataset_root: str | Path,
    metadata_path: str | Path,
    hash_manifest_path: str | Path,
    *,
    source_url: str,
    source_commit: str,
    seed: int = 2026,
    validation_ratio: float = 0.2,
    expected_development_identities: int | None = None,
    expected_open_set_identities: int | None = None,
    expected_images: int | None = None,
) -> dict[str, object]:
    metadata = scan_holstein2025(
        dataset_root,
        source_url=source_url,
        source_commit=source_commit,
        seed=seed,
        validation_ratio=validation_ratio,
    )
    summary = validate_holstein2025_metadata(
        metadata,
        expected_development_identities=expected_development_identities,
        expected_open_set_identities=expected_open_set_identities,
        expected_images=expected_images,
    )
    metadata_output = Path(metadata_path)
    hash_output = Path(hash_manifest_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    hash_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_output, index=False)
    metadata[["relative_path", "sha256"]].to_csv(hash_output, index=False)
    summary.update(
        {
            "metadata_path": str(metadata_output),
            "hash_manifest_path": str(hash_output),
            "source_url": source_url,
            "source_commit": source_commit,
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build validated Holstein2025 open-set metadata.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--hashes-out", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)
    summary = build_holstein2025_metadata(
        args.dataset_root,
        args.metadata_out,
        args.hashes_out,
        source_url=args.source_url,
        source_commit=args.source_commit,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    print(summary)
    return 0 if summary["data_integrity_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
