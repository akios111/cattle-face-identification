from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from .metrics import bootstrap_group_mean_ci, compute_retrieval_metrics
from .models import build_embedding_model
from .hashing import sha256_file
from .tfdata import dataframe_to_dataset


@dataclass(frozen=True)
class OpenSetOutputPaths:
    metrics: Path
    predictions: Path
    gallery_embeddings: Path
    probe_embeddings: Path


def open_set_output_paths(run_dir: str | Path, output_suffix: str) -> OpenSetOutputPaths:
    run_dir = Path(run_dir)
    suffix = output_suffix.strip().strip("_")
    if not suffix:
        raise ValueError("open-set output suffix must not be empty")
    return OpenSetOutputPaths(
        metrics=run_dir / f"metrics_{suffix}.json",
        predictions=run_dir / f"predictions_{suffix}.csv",
        gallery_embeddings=run_dir / f"gallery_embeddings_{suffix}.npy",
        probe_embeddings=run_dir / f"probe_embeddings_{suffix}.npy",
    )


def is_open_set_evaluation_complete(paths: OpenSetOutputPaths) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            paths.metrics,
            paths.predictions,
            paths.gallery_embeddings,
            paths.probe_embeddings,
        )
    )


def _embedding_dataset(
    metadata: pd.DataFrame,
    *,
    model_name: str,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    return dataframe_to_dataset(
        metadata,
        model_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        deterministic=True,
    )


def _retrieval_set_sha256(metadata: pd.DataFrame) -> str:
    preferred = ["sha256", "relative_path", "image_path", "animal_id", "split"]
    columns = [column for column in preferred if column in metadata.columns]
    if "animal_id" not in columns or not ({"sha256", "relative_path", "image_path"} & set(columns)):
        raise ValueError("retrieval metadata requires animal_id and a stable image identifier")
    rows = [
        "\x1f".join(str(row[column]) for column in columns)
        for row in metadata[columns].fillna("").to_dict(orient="records")
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def evaluate_open_set_run(
    run_dir: str | Path,
    metadata_path: str | Path,
    *,
    gallery_split: str = "gallery",
    probe_split: str = "probe",
    output_suffix: str = "holstein2025_unseen_identity_reid",
    bootstrap_resamples: int = 2000,
    skip_existing: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    outputs = open_set_output_paths(run_dir, output_suffix)
    if skip_existing and is_open_set_evaluation_complete(outputs):
        return outputs.metrics
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(metadata_path)
    gallery = metadata[metadata["split"] == gallery_split].reset_index(drop=True)
    probe = metadata[metadata["split"] == probe_split].reset_index(drop=True)
    if gallery.empty or probe.empty:
        raise ValueError("open-set evaluation requires non-empty gallery and probe splits")

    classifier = tf.keras.models.load_model(run_dir / "model.keras")
    embedding_model = build_embedding_model(classifier)
    image_size = tuple(int(value) for value in manifest["image_size"])
    batch_size = int(manifest["batch_size"])
    seed = int(manifest["seed"])

    started = time.perf_counter()
    gallery_embeddings = embedding_model.predict(
        _embedding_dataset(
            gallery,
            model_name=str(manifest["model"]),
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        ),
        verbose=0,
    )
    probe_embeddings = embedding_model.predict(
        _embedding_dataset(
            probe,
            model_name=str(manifest["model"]),
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        ),
        verbose=0,
    )
    elapsed = time.perf_counter() - started

    metrics, retrieval = compute_retrieval_metrics(
        gallery_embeddings,
        gallery["animal_id"].astype(str).to_numpy(),
        probe_embeddings,
        probe["animal_id"].astype(str).to_numpy(),
    )
    predictions = probe.copy()
    if "sha256" in predictions.columns:
        predictions["probe_sample_id"] = predictions["sha256"].astype(str)
    elif "relative_path" in predictions.columns:
        predictions["probe_sample_id"] = predictions["relative_path"].astype(str)
    else:
        predictions["probe_sample_id"] = predictions["image_path"].astype(str)
    for column in retrieval.columns:
        if column == "animal_id":
            continue
        predictions[column] = retrieval[column].to_numpy()

    rank1_ci = bootstrap_group_mean_ci(
        predictions["correct_rank_1"].astype(float),
        predictions["animal_id"],
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    map_ci = bootstrap_group_mean_ci(
        predictions["average_precision"],
        predictions["animal_id"],
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    metrics.update(
        {
            "model": manifest["model"],
            "protocol": output_suffix,
            "source_training_protocol": manifest["protocol"],
            "evaluation_protocol": output_suffix,
            "seed": seed,
            "training_seed": int(manifest.get("training_seed", seed)),
            "split_seed": int(manifest.get("split_seed", seed)),
            "augmentation_seed": int(manifest.get("augmentation_seed", seed)),
            "protocol_version": str(manifest.get("protocol_version", "legacy")),
            "metadata_path": str(metadata_path),
            "gallery_set_sha256": _retrieval_set_sha256(gallery),
            "probe_set_sha256": _retrieval_set_sha256(probe),
            "checkpoint_sha256": sha256_file(run_dir / "model.keras"),
            "checkpoint_size_bytes": int((run_dir / "model.keras").stat().st_size),
            "embedding_layer": "global_average_pooling",
            "inference_seconds": float(elapsed),
            "rank_1_ci_low": rank1_ci["ci_low"],
            "rank_1_ci_high": rank1_ci["ci_high"],
            "map_ci_low": map_ci["ci_low"],
            "map_ci_high": map_ci["ci_high"],
            "bootstrap_unit": "animal_id",
            "bootstrap_resamples": bootstrap_resamples,
        }
    )
    outputs.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(outputs.predictions, index=False)
    np.save(outputs.gallery_embeddings, gallery_embeddings)
    np.save(outputs.probe_embeddings, probe_embeddings)
    return outputs.metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained run on unseen gallery/probe identities.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--gallery-split", default="gallery")
    parser.add_argument("--probe-split", default="probe")
    parser.add_argument("--output-suffix", default="holstein2025_unseen_identity_reid")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    output = evaluate_open_set_run(
        args.run,
        args.metadata,
        gallery_split=args.gallery_split,
        probe_split=args.probe_split,
        output_suffix=args.output_suffix,
        bootstrap_resamples=args.bootstrap_resamples,
        skip_existing=args.skip_existing,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
