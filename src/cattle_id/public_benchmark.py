from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IMAGE_EXTENSIONS


CATTELY_IDENTITY_RE = re.compile(r"^Cattle_(\d+)$", re.IGNORECASE)
PUBLIC_BENCHMARK_PROTOCOL = "external_public_face_benchmark"
IGNORED_TOP_LEVEL_DIRS = {"valid", "train", "test", "validation", "images", "labels"}


def _natural_key(path: Path) -> tuple[int, str]:
    match = CATTELY_IDENTITY_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name.lower()
    number_match = re.search(r"\d+", path.name)
    if number_match:
        return int(number_match.group(0)), path.name.lower()
    return 10**9, path.name.lower()


def _image_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _direct_image_paths(path: Path) -> list[Path]:
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _is_identity_dir(path: Path) -> bool:
    name = path.name.lower()
    if name in IGNORED_TOP_LEVEL_DIRS or name.startswith("extra_") or name.startswith("."):
        return False
    return bool(_direct_image_paths(path))


def scan_cattely_benchmark(dataset_root: str | Path, *, source_url: str) -> pd.DataFrame:
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Cattely dataset root not found: {root}")

    identity_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and _is_identity_dir(path)],
        key=_natural_key,
    )
    if not identity_dirs:
        raise FileNotFoundError(f"No Cattely identity folders found in {root}")

    rows: list[dict[str, object]] = []
    for class_id, identity_dir in enumerate(identity_dirs):
        image_paths = sorted(
            _direct_image_paths(identity_dir),
            key=_image_key,
        )
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_file": image_path.name,
                    "class_id": int(class_id),
                    "animal_id": identity_dir.name,
                    "external_dataset": "cattely",
                    "protocol": PUBLIC_BENCHMARK_PROTOCOL,
                    "split": "",
                    "source_url": source_url,
                    "capture_date": "unknown",
                    "camera_id": "unknown",
                    "lighting": "unknown",
                    "pose": "front_profile",
                    "operator": "public_dataset",
                    "notes": "Public Cattely cattle face benchmark; identities are disjoint from CattleSSFR.",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No Cattely image files found in {root}")
    return pd.DataFrame(rows)


def _split_labels_for_group(n: int, *, rng: np.random.Generator, train_ratio: float, validation_ratio: float) -> list[str]:
    if n < 3:
        raise ValueError("Each public benchmark identity needs at least three images for train/validation/test splits")
    train_count = max(1, int(n * train_ratio))
    validation_count = max(1, int(n * validation_ratio))
    if train_count + validation_count >= n:
        train_count = max(1, n - validation_count - 1)
    if train_count + validation_count >= n:
        validation_count = max(1, n - train_count - 1)
    test_count = n - train_count - validation_count
    if test_count < 1:
        raise ValueError("Split ratios leave no test images for at least one public benchmark identity")

    labels = np.array(["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count)
    rng.shuffle(labels)
    return labels.tolist()


def assign_closed_set_splits(
    metadata: pd.DataFrame,
    *,
    seed: int = 2026,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> pd.DataFrame:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")
    if "animal_id" not in metadata.columns:
        raise ValueError("metadata is missing required column: animal_id")

    output = metadata.copy()
    output["split"] = ""
    rng = np.random.default_rng(seed)
    for _animal_id, group in output.groupby("animal_id", sort=True):
        if len(group) < 3:
            output.loc[group.index, "split"] = "excluded_low_image_count"
            output.loc[group.index, "notes"] = (
                output.loc[group.index, "notes"].astype(str)
                + " Excluded from closed-set split: too few images for train/validation/test."
            )
            continue
        labels = _split_labels_for_group(
            len(group),
            rng=rng,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        output.loc[group.index, "split"] = labels
    return output


def build_external_public_benchmark_metadata(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    source_url: str,
    seed: int = 2026,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> Path:
    metadata = scan_cattely_benchmark(dataset_root, source_url=source_url)
    metadata = assign_closed_set_splits(
        metadata,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build metadata for an external public cattle-face benchmark.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)

    output = build_external_public_benchmark_metadata(
        args.dataset_root,
        args.out,
        source_url=args.source_url,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
