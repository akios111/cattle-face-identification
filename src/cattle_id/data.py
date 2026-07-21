from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .augmentation import apply_augmentation, get_augmentation_specs, resize_image
from .config import (
    HARDENING_PROTOCOL_VERSION,
    config_for_run,
    image_size_from_config,
    load_config,
    metadata_filename,
    run_seeds_from_config,
)
from .hashing import sha256_file
from .logging_utils import log_line

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTERNAL_ACQUISITION_REQUIRED_COLUMNS = {
    "image_path",
    "class_id",
    "animal_id",
    "acquisition_id",
    "capture_date",
    "camera_id",
    "lighting",
    "pose",
    "operator",
    "split",
    "notes",
}


def build_source_manifest(
    image_dir: str | Path,
    *,
    include_sha256: bool = False,
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    files = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    rows = []
    for class_id, path in enumerate(files):
        row: dict[str, object] = {
            "source_file": path.name,
            "source_path": str(path.resolve()),
            "class_id": class_id,
        }
        if include_sha256:
            row["source_sha256"] = sha256_file(path)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_external_acquisition_manifest(
    manifest: pd.DataFrame,
    *,
    allowed_class_ids: set[int],
    training_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    missing = sorted(EXTERNAL_ACQUISITION_REQUIRED_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(f"external acquisition manifest missing columns: {', '.join(missing)}")

    output = manifest.copy()
    output["class_id"] = output["class_id"].astype(int)
    output["split"] = output["split"].astype(str).str.lower()
    bad_splits = sorted(set(output.loc[output["split"] != "test", "split"].tolist()))
    if bad_splits:
        raise ValueError("external acquisition rows must use split=test")

    unknown_classes = sorted(set(output["class_id"].tolist()).difference(allowed_class_ids))
    if unknown_classes:
        raise ValueError(f"external acquisition manifest contains unknown class_id values: {unknown_classes}")

    resolved_paths = []
    for value in output["image_path"].astype(str):
        path = Path(value)
        resolved = path if path.is_absolute() else path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"external acquisition image not found: {resolved}")
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"external acquisition image has unsupported extension: {resolved}")
        resolved_paths.append(str(resolved))
    output["image_path"] = resolved_paths

    if output["image_path"].duplicated().any():
        raise ValueError("external acquisition manifest contains duplicate image_path values")
    if output["acquisition_id"].astype(str).duplicated().any():
        raise ValueError("external acquisition manifest contains duplicate acquisition_id values")

    if training_metadata is not None and "acquisition_id" in training_metadata.columns:
        train_acquisitions = set(training_metadata["acquisition_id"].dropna().astype(str).tolist())
        external_acquisitions = set(output["acquisition_id"].astype(str).tolist())
        overlap = sorted(train_acquisitions.intersection(external_acquisitions))
        if overlap:
            raise ValueError(f"external acquisition leakage: acquisition_id overlap {overlap}")

    output["protocol"] = "external_acquisition_holdout"
    output["augmentation_id"] = "external_real"
    output["augmentation_family"] = "external_real"
    return output


def build_external_acquisition_metadata(
    manifest_path: str | Path,
    *,
    output_path: str | Path,
    allowed_class_ids: set[int],
    training_metadata: pd.DataFrame | None = None,
) -> Path:
    manifest = pd.read_csv(manifest_path)
    metadata = validate_external_acquisition_manifest(
        manifest,
        allowed_class_ids=allowed_class_ids,
        training_metadata=training_metadata,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    return output_path


def build_augmented_metadata(manifest: pd.DataFrame, profile: str = "all") -> pd.DataFrame:
    specs = get_augmentation_specs(profile)
    rows: list[dict[str, object]] = []
    for source in manifest.to_dict(orient="records"):
        for spec in specs:
            row = {
                    "source_file": source["source_file"],
                    "source_path": source["source_path"],
                    "class_id": int(source["class_id"]),
                    "augmentation_id": spec.identifier,
                    "augmentation_family": spec.family,
                    "sample_id": (
                        f"{int(source['class_id']):03d}:"
                        f"{source['source_file']}:{spec.identifier}"
                    ),
                }
            if source.get("source_sha256"):
                row["source_sha256"] = source["source_sha256"]
            rows.append(row)
    return pd.DataFrame(rows)


def assign_splits(metadata: pd.DataFrame, protocol: str, seed: int = 2026) -> pd.DataFrame:
    protocol = protocol.lower()
    output = metadata.copy()
    output["split"] = "train"

    if protocol in {"paper_random", "augmentation_ablation"}:
        class_counts = output.groupby("class_id").size()
        minimum_count = int(class_counts.min())
        if minimum_count < 3:
            raise ValueError(
                "paper_random requires at least 3 samples per class for disjoint "
                "train/validation/test splits; the no-augmentation CattleSSFR "
                f"ablation is not estimable because the minimum is {minimum_count}"
            )

        if not (len(output) == 6220 and class_counts.eq(20).all()):
            for class_id, group in output.groupby("class_id", sort=True):
                indices = group.index.to_numpy(copy=True)
                rng = np.random.default_rng(seed + int(class_id) * 1_000_003)
                rng.shuffle(indices)
                count = len(indices)
                test_count = max(1, int(round(count * 0.30)))
                validation_count = max(1, int(round(count * 0.10)))
                if test_count + validation_count >= count:
                    test_count = 1
                    validation_count = 1
                output.loc[indices[:test_count], "split"] = "test"
                output.loc[
                    indices[test_count : test_count + validation_count], "split"
                ] = "validation"
                output.loc[indices[test_count + validation_count :], "split"] = "train"
            output["protocol"] = protocol
            return output

        indices = output.index.to_numpy()
        labels = output["class_id"].to_numpy()
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed,
            stratify=labels,
        )
        train_idx, validation_idx = train_test_split(
            trainval_idx,
            test_size=436 if len(output) == 6220 else 0.10,
            random_state=seed,
            stratify=output.loc[trainval_idx, "class_id"].to_numpy(),
        )
        output.loc[test_idx, "split"] = "test"
        output.loc[validation_idx, "split"] = "validation"
        output.loc[train_idx, "split"] = "train"
        output["protocol"] = protocol
        return output

    if protocol == "transform_holdout":
        validation_ids = {"brightness_up", "contrast_down"}
        test_ids = {"gaussian_noise", "blur", "cutout_center", "cutout_random"}
        output.loc[output["augmentation_id"].isin(validation_ids), "split"] = "validation"
        output.loc[output["augmentation_id"].isin(test_ids), "split"] = "test"
        output["protocol"] = protocol
        return output

    raise ValueError(f"Unknown split protocol: {protocol}")


def materialize_augmented_images(
    metadata: pd.DataFrame,
    output_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 2026,
    log_path: str | Path | None = None,
    protocol_version: str = "legacy",
    materialization_id: str = "legacy",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.identifier: spec for spec in get_augmentation_specs("all")}
    rows = []
    records = metadata.to_dict(orient="records")
    total = len(records)
    for index, row in enumerate(records, start=1):
        class_dir = output_dir / f"class_{int(row['class_id']):03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(row["source_file"])).stem
        filename = f"{stem}__{row['augmentation_id']}.png"
        image_path = class_dir / filename
        if not image_path.exists():
            image = Image.open(row["source_path"])
            image = apply_augmentation(
                image,
                specs[str(row["augmentation_id"])],
                seed=seed,
                source_id=str(row["source_file"]),
                protocol_version=protocol_version,
            )
            image = resize_image(image, image_size)
            image.save(image_path)
        elif protocol_version == HARDENING_PROTOCOL_VERSION:
            with Image.open(image_path) as existing:
                if existing.size != image_size:
                    raise ValueError(
                        f"Frozen materialization has unexpected size: {image_path} "
                        f"is {existing.size}, expected {image_size}"
                    )
        updated = dict(row)
        updated["image_path"] = str(image_path.resolve())
        if protocol_version == HARDENING_PROTOCOL_VERSION:
            updated["source_sha256"] = str(
                row.get("source_sha256") or sha256_file(row["source_path"])
            )
            updated["image_sha256"] = sha256_file(image_path)
            updated["protocol_version"] = protocol_version
            updated["augmentation_seed"] = int(seed)
            updated["materialization_id"] = materialization_id
        rows.append(updated)
        if index == 1 or index % 500 == 0 or index == total:
            message = f"materialized {index}/{total} images into {output_dir}"
            print(message, flush=True)
            if log_path is not None:
                log_line(log_path, "prepare", message)
    return pd.DataFrame(rows)


def clone_or_update_dataset(
    repo_url: str,
    dataset_dir: str | Path,
    commit_sha: str | None = None,
) -> Path:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prepare] cloning dataset into {dataset_dir}", flush=True)
        subprocess.run(["git", "clone", repo_url, str(dataset_dir)], check=True)
    else:
        print(f"[prepare] using existing dataset at {dataset_dir}", flush=True)
    if commit_sha:
        print(f"[prepare] checking out dataset commit {commit_sha}", flush=True)
        subprocess.run(["git", "checkout", commit_sha], cwd=dataset_dir, check=True)
    return dataset_dir


def resolve_repo_commit(repo_dir: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_from_config(
    config_path: str | Path,
    protocol: str | None = None,
    seed: int | None = None,
    *,
    training_seed: int | None = None,
    split_seed: int | None = None,
    augmentation_seed: int | None = None,
) -> Path:
    print(f"[prepare] loading config {config_path}", flush=True)
    loaded_config = load_config(config_path)
    requested_protocol = protocol or loaded_config.get("split", {}).get("protocol", "paper_random")
    config = config_for_run(
        loaded_config,
        protocol=requested_protocol,
        seed=seed,
        training_seed=training_seed,
        split_seed=split_seed,
        augmentation_seed=augmentation_seed,
    )
    dataset_cfg = config["dataset"]
    output_cfg = config["output"]
    split_cfg = config.get("split", {})
    augmentation_cfg = config.get("augmentation", {})
    seeds = run_seeds_from_config(config)
    effective_training_seed = seeds["training_seed"]
    effective_split_seed = seeds["split_seed"]
    effective_augmentation_seed = seeds["augmentation_seed"]
    protocol_version = str(config.get("protocol_version", "legacy"))
    split_protocol = split_cfg.get("protocol", "paper_random")
    log_path = (
        Path(output_cfg.get("metadata_dir", "artifacts/metadata")).parent
        / "logs"
        / (
            f"prepare_{requested_protocol}_split{effective_split_seed}_"
            f"aug{effective_augmentation_seed}.log"
        )
    )

    def emit(message: str) -> None:
        log_line(log_path, "prepare", message)

    raw_dir = Path(dataset_cfg.get("raw_dir", "data/raw/CattleSSFR"))
    emit(f"config_path={config_path}")
    emit(f"requested_protocol={requested_protocol}")
    emit(f"split_protocol={split_protocol}")
    emit(f"protocol_version={protocol_version}")
    emit(f"training_seed={effective_training_seed}")
    emit(f"split_seed={effective_split_seed}")
    emit(f"augmentation_seed={effective_augmentation_seed}")
    emit(f"raw_dir={raw_dir}")
    configured_commit = dataset_cfg.get("commit_sha") or None
    if protocol_version == HARDENING_PROTOCOL_VERSION and not configured_commit:
        raise ValueError("hardening_v2 requires dataset.commit_sha")
    repo_dir = clone_or_update_dataset(
        dataset_cfg["repo_url"],
        raw_dir,
        configured_commit,
    )
    commit_sha = resolve_repo_commit(repo_dir)
    emit(f"dataset_commit_sha={commit_sha}")
    image_dir = repo_dir / dataset_cfg.get("image_subdir", "cattle_images")
    emit(f"building source manifest from {image_dir}")
    manifest = build_source_manifest(
        image_dir,
        include_sha256=protocol_version == HARDENING_PROTOCOL_VERSION,
    )
    emit(f"source_images={len(manifest)}")
    if len(manifest) != int(dataset_cfg.get("expected_images", 311)):
        raise ValueError(f"Expected 311 source images, found {len(manifest)}")

    profile = augmentation_cfg.get("profile", "all")
    emit(f"building augmentation metadata profile={profile}")
    metadata = build_augmented_metadata(manifest, profile=profile)
    emit(f"augmented_rows={len(metadata)}")
    emit(f"assigning split protocol={split_protocol}")
    metadata = assign_splits(metadata, protocol=split_protocol, seed=effective_split_seed)
    metadata["protocol"] = requested_protocol
    metadata["protocol_version"] = protocol_version
    metadata["split_seed"] = effective_split_seed
    metadata["augmentation_seed"] = effective_augmentation_seed
    emit(f"split_counts={metadata['split'].value_counts().to_dict()}")
    image_size = image_size_from_config(config)
    materialization_id = str(
        augmentation_cfg.get(
            "materialization_id",
            (
                f"cattlessfr_{protocol_version}_aug{effective_augmentation_seed}_"
                f"{image_size[0]}x{image_size[1]}"
            ),
        )
    )
    processed_root = Path(output_cfg.get("processed_dir", "artifacts/processed"))
    processed_dir = (
        processed_root / materialization_id
        if protocol_version == HARDENING_PROTOCOL_VERSION
        else processed_root / f"{profile}_{image_size[0]}x{image_size[1]}"
    )
    emit(f"materializing images into {processed_dir}")
    metadata = materialize_augmented_images(
        metadata,
        processed_dir,
        image_size=image_size,
        seed=effective_augmentation_seed,
        log_path=log_path,
        protocol_version=protocol_version,
        materialization_id=materialization_id,
    )
    metadata["dataset_commit_sha"] = commit_sha
    metadata_path = Path(output_cfg.get("metadata_dir", "artifacts/metadata"))
    metadata_path.mkdir(parents=True, exist_ok=True)
    if protocol_version == HARDENING_PROTOCOL_VERSION:
        output_name = metadata_filename(
            requested_protocol,
            split_seed=effective_split_seed,
            augmentation_seed=effective_augmentation_seed,
        )
    else:
        output_name = metadata_filename(
            requested_protocol,
            seed=effective_training_seed if seed is not None else None,
        )
    output_path = metadata_path / output_name
    metadata.to_csv(output_path, index=False)
    if protocol_version == HARDENING_PROTOCOL_VERSION:
        byte_manifest = output_path.with_name(f"{output_path.stem}_byte_manifest.csv")
        metadata[
            [
                "sample_id",
                "source_file",
                "class_id",
                "augmentation_id",
                "split",
                "source_sha256",
                "image_sha256",
                "materialization_id",
            ]
        ].to_csv(byte_manifest, index=False)
        emit(f"wrote byte manifest {byte_manifest}")
    emit(f"wrote metadata {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare CattleSSFR metadata and images.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--protocol", default=None, help="Split protocol override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for matrix runs.")
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    args = parser.parse_args(argv)
    prepare_kwargs: dict[str, object] = {"protocol": args.protocol, "seed": args.seed}
    if args.training_seed is not None:
        prepare_kwargs["training_seed"] = args.training_seed
    if args.split_seed is not None:
        prepare_kwargs["split_seed"] = args.split_seed
    if args.augmentation_seed is not None:
        prepare_kwargs["augmentation_seed"] = args.augmentation_seed
    output_path = prepare_from_config(args.config, **prepare_kwargs)
    print(output_path)


if __name__ == "__main__":
    main()
