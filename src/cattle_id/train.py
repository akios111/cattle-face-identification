from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import strftime

import pandas as pd

from .config import (
    HARDENING_PROTOCOL_VERSION,
    config_for_run,
    image_size_from_config,
    load_config,
    metadata_filename,
    run_seeds_from_config,
)
from .hashing import sha256_file
from .logging_utils import append_event, log_line, runtime_snapshot, write_json
from .models import build_model, normalize_model_name
from .runtime import configure_tensorflow_acceleration
from .tfdata import dataframe_to_dataset
from .training import build_training_stages, fit_stages


def _safe_run_component(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def deterministic_run_id(
    model_name: str,
    protocol: str,
    seed: int | None = None,
    *,
    training_seed: int | None = None,
    split_seed: int | None = None,
    augmentation_seed: int | None = None,
) -> str:
    if training_seed is None:
        if seed is None:
            raise ValueError("deterministic run id requires seed or training_seed")
        training_seed = int(seed)
    if split_seed is None and augmentation_seed is None:
        return (
            f"matrix_{_safe_run_component(model_name)}_"
            f"{_safe_run_component(protocol)}_seed{int(training_seed)}"
        )
    split_seed = int(training_seed if split_seed is None else split_seed)
    augmentation_seed = int(training_seed if augmentation_seed is None else augmentation_seed)
    return (
        f"matrix_{_safe_run_component(model_name)}_{_safe_run_component(protocol)}_"
        f"train{int(training_seed)}_split{split_seed}_aug{augmentation_seed}"
    )


def is_run_complete(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    required = ("run_complete.json", "model.keras", "manifest.json", "history.csv")
    return all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in required)


def train_from_config(
    config_path: str | Path,
    model_name: str,
    protocol: str,
    epochs: int | None = None,
    weights: str | None = "imagenet",
    seed: int | None = None,
    training_seed: int | None = None,
    split_seed: int | None = None,
    augmentation_seed: int | None = None,
    metadata_path: str | Path | None = None,
    run_id: str | None = None,
    skip_completed: bool = False,
) -> Path:
    print(
        f"[train] loading config={config_path} model={model_name} protocol={protocol} "
        f"epochs_override={epochs} weights={weights} seed_override={seed} "
        f"metadata_override={metadata_path} run_id_override={run_id} skip_completed={skip_completed}",
        flush=True,
    )
    config = config_for_run(
        load_config(config_path),
        protocol=protocol,
        seed=seed,
        training_seed=training_seed,
        split_seed=split_seed,
        augmentation_seed=augmentation_seed,
    )
    model_name = normalize_model_name(model_name)
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    seeds = run_seeds_from_config(config)
    effective_seed = seeds["training_seed"]
    effective_split_seed = seeds["split_seed"]
    effective_augmentation_seed = seeds["augmentation_seed"]
    protocol_version = str(config.get("protocol_version", "legacy"))
    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else (
        Path(output_cfg.get("metadata_dir", "artifacts/metadata"))
        / (
            metadata_filename(
                protocol,
                split_seed=effective_split_seed,
                augmentation_seed=effective_augmentation_seed,
            )
            if protocol_version == HARDENING_PROTOCOL_VERSION
            else metadata_filename(protocol, seed=effective_seed if seed is not None else None)
        )
    )
    seed_part = f"_seed{effective_seed}" if seed is not None else ""
    resolved_run_id = _safe_run_component(run_id) if run_id else (
        f"{model_name}_{protocol}{seed_part}_{strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(output_cfg.get("run_dir", "artifacts/runs")) / resolved_run_id
    if skip_completed and is_run_complete(run_dir):
        print(f"[train] completed run exists; skipping {run_dir}", flush=True)
        return run_dir

    import tensorflow as tf

    tf.keras.utils.set_random_seed(effective_seed)
    runtime_info = configure_tensorflow_acceleration(training_cfg.get("acceleration", {}))

    print(f"[train] reading metadata={resolved_metadata_path}", flush=True)
    metadata = pd.read_csv(resolved_metadata_path)
    if protocol_version == HARDENING_PROTOCOL_VERSION:
        required_columns = {
            "sample_id",
            "source_sha256",
            "image_sha256",
            "protocol_version",
            "split_seed",
            "augmentation_seed",
            "materialization_id",
        }
        missing = sorted(required_columns.difference(metadata.columns))
        if missing:
            raise ValueError(f"hardening_v2 metadata missing columns: {', '.join(missing)}")
        if metadata["sample_id"].astype(str).duplicated().any():
            raise ValueError("hardening_v2 metadata contains duplicate sample_id values")
        if set(metadata["protocol_version"].astype(str)) != {HARDENING_PROTOCOL_VERSION}:
            raise ValueError("hardening_v2 metadata has an incompatible protocol_version")
    development = metadata[metadata["split"].isin(["train", "validation"])].copy()
    class_ids = sorted(development["class_id"].astype(int).unique().tolist())
    if class_ids != list(range(len(class_ids))):
        raise ValueError("Training and validation class_id values must be contiguous from zero")
    num_classes = len(class_ids)
    image_size = image_size_from_config(config)
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs_override = epochs
    cache_cfg = training_cfg.get("dataset_cache", False)
    cache_value: bool | str = False
    if cache_cfg == "memory":
        cache_value = True
    elif isinstance(cache_cfg, str) and cache_cfg not in {"", "false", "False", "none"}:
        cache_value = cache_cfg

    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"
    log_line(run_log, "train", f"run_id={resolved_run_id}")
    log_line(run_log, "train", f"config_path={config_path}")
    log_line(run_log, "train", f"metadata_path={resolved_metadata_path}")
    log_line(run_log, "train", f"model={model_name} protocol={protocol} weights={weights}")
    log_line(
        run_log,
        "train",
        f"image_size={image_size} batch_size={batch_size} training_seed={effective_seed} "
        f"split_seed={effective_split_seed} augmentation_seed={effective_augmentation_seed} "
        f"protocol_version={protocol_version}",
    )
    split_counts = metadata["split"].value_counts().to_dict()
    class_counts = metadata.groupby("split")["class_id"].nunique().to_dict()
    log_line(run_log, "train", f"split_counts={split_counts}")
    log_line(run_log, "train", f"class_counts={class_counts}")
    append_event(
        run_dir,
        "run_started",
        run_id=resolved_run_id,
        config_path=str(config_path),
        metadata_path=str(resolved_metadata_path),
        model=model_name,
        protocol=protocol,
        weights=weights,
        image_size=image_size,
        batch_size=batch_size,
        seed=effective_seed,
        training_seed=effective_seed,
        split_seed=effective_split_seed,
        augmentation_seed=effective_augmentation_seed,
        protocol_version=protocol_version,
        split_counts=split_counts,
        class_counts=class_counts,
    )
    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "runtime_snapshot.json", runtime_snapshot())

    log_line(run_log, "train", "building train dataset")
    train_metadata = metadata[metadata["split"] == "train"].copy()
    excluded_families = {
        str(value) for value in training_cfg.get("exclude_augmentation_families", [])
    }
    if excluded_families:
        if "augmentation_family" not in train_metadata.columns:
            raise ValueError("training exclusion requires augmentation_family metadata")
        train_metadata = train_metadata[
            ~train_metadata["augmentation_family"].astype(str).isin(excluded_families)
        ].copy()
        if train_metadata.empty:
            raise ValueError("training exclusions removed every training sample")
        log_line(
            run_log,
            "train",
            f"excluded_training_families={sorted(excluded_families)} "
            f"remaining_training_samples={len(train_metadata)}",
        )
    train_ds = dataframe_to_dataset(
        train_metadata,
        model_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=effective_seed,
        cache=cache_value,
    )
    log_line(run_log, "train", "building validation dataset")
    validation_ds = dataframe_to_dataset(
        metadata[metadata["split"] == "validation"],
        model_name,
        image_size=image_size,
        batch_size=batch_size,
        cache=cache_value,
    )

    log_line(run_log, "train", "building model")
    model = build_model(
        model_name,
        num_classes=num_classes,
        input_shape=(image_size[0], image_size[1], 3),
        weights=weights,
        dropout=float(training_cfg.get("dropout", 0.2)),
    )
    stages = build_training_stages(training_cfg, epochs_override=epochs_override)
    log_line(run_log, "train", f"stages={[stage.__dict__ for stage in stages]}")
    append_event(run_dir, "stages_built", stages=[stage.__dict__ for stage in stages])
    log_line(run_log, "train", "fit_stages begin")
    history_frame = fit_stages(model, train_ds, validation_ds, stages, run_dir, training_cfg)
    history_frame.to_csv(run_dir / "history.csv", index=False)
    log_line(run_log, "train", f"history_written={run_dir / 'history.csv'}")
    manifest = {
        "model": model_name,
        "protocol": protocol,
        "metadata_path": str(resolved_metadata_path),
        "num_classes": num_classes,
        "image_size": image_size,
        "seed": effective_seed,
        "training_seed": effective_seed,
        "split_seed": effective_split_seed,
        "augmentation_seed": effective_augmentation_seed,
        "protocol_version": protocol_version,
        "materialization_id": (
            str(metadata["materialization_id"].iloc[0])
            if "materialization_id" in metadata.columns
            else "legacy"
        ),
        "metadata_sha256": sha256_file(resolved_metadata_path),
        "excluded_training_families": sorted(excluded_families),
        "epochs": int(epochs_override or training_cfg.get("epochs", 100)),
        "batch_size": batch_size,
        "runtime": runtime_info,
        "stages": [stage.__dict__ for stage in stages],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_line(run_log, "train", f"manifest_written={run_dir / 'manifest.json'}")
    append_event(run_dir, "run_completed", run_dir=str(run_dir))
    (run_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "model": model_name,
                "protocol": protocol,
                "seed": effective_seed,
                "training_seed": effective_seed,
                "split_seed": effective_split_seed,
                "augmentation_seed": effective_augmentation_seed,
                "protocol_version": protocol_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-backbone CattleSSFR model.")
    parser.add_argument("--config", default="configs/cattlessfr.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", default="paper_random")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for matrix runs.")
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    parser.add_argument("--metadata", default=None, help="Optional training metadata CSV override.")
    parser.add_argument("--run-id", default=None, help="Deterministic run directory name for resumable jobs.")
    parser.add_argument("--skip-completed", action="store_true", help="Return an already completed deterministic run.")
    args = parser.parse_args(argv)
    weights = None if args.weights.lower() == "none" else args.weights
    call_kwargs = {
        "weights": weights,
        "seed": args.seed,
        "metadata_path": args.metadata,
        "run_id": args.run_id,
        "skip_completed": args.skip_completed,
    }
    if args.training_seed is not None:
        call_kwargs["training_seed"] = args.training_seed
    if args.split_seed is not None:
        call_kwargs["split_seed"] = args.split_seed
    if args.augmentation_seed is not None:
        call_kwargs["augmentation_seed"] = args.augmentation_seed
    run_dir = train_from_config(
        args.config,
        args.model,
        args.protocol,
        args.epochs,
        **call_kwargs,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
