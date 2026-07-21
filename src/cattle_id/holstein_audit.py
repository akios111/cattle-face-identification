from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hashing import sha256_file
from .metrics import (
    bootstrap_group_mean_ci,
    compute_retrieval_metrics,
    paired_group_mean_delta_ci,
)
from .models import build_imagenet_embedding_model
from .tfdata import dataframe_to_dataset


DEFAULT_SUFFIX = "holstein2025_zero_shot_reid_hardening_v2"


def select_diverse_rank1_errors(predictions: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Select difficult rank-1 errors while prioritizing distinct true identities."""
    required = {"animal_id", "predicted_animal_id", "correct_rank_1"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Holstein predictions missing error-gallery columns: {missing}")
    values = predictions["correct_rank_1"]
    if pd.api.types.is_bool_dtype(values):
        correct = values.astype(bool)
    else:
        normalized = values.astype(str).str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        if not normalized.isin(mapping).all():
            raise ValueError("correct_rank_1 contains non-boolean values")
        correct = normalized.map(mapping).astype(bool)
    errors = predictions.loc[~correct].copy()
    if errors.empty or limit <= 0:
        return errors.head(0)

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "first_correct_rank" in errors.columns:
        sort_columns.append("first_correct_rank")
        ascending.append(False)
    if "average_precision" in errors.columns:
        sort_columns.append("average_precision")
        ascending.append(True)
    if "top_1_similarity" in errors.columns:
        sort_columns.append("top_1_similarity")
        ascending.append(False)
    if sort_columns:
        errors = errors.sort_values(sort_columns, ascending=ascending, kind="mergesort")

    diverse = errors.drop_duplicates(subset=["animal_id"], keep="first").head(limit)
    if len(diverse) < limit:
        stable_ids = set(_stable_probe_ids(diverse))
        remaining = errors.loc[~_stable_probe_ids(errors).isin(stable_ids)]
        diverse = pd.concat([diverse, remaining.head(limit - len(diverse))], ignore_index=True)
    return diverse.head(limit).copy()


def _output_paths(directory: Path, suffix: str) -> dict[str, Path]:
    return {
        "metrics": directory / f"metrics_{suffix}.json",
        "predictions": directory / f"predictions_{suffix}.csv",
        "gallery_embeddings": directory / f"gallery_embeddings_{suffix}.npy",
        "probe_embeddings": directory / f"probe_embeddings_{suffix}.npy",
    }


def _stable_probe_ids(frame: pd.DataFrame) -> pd.Series:
    for column in ("probe_sample_id", "sha256", "relative_path", "image_path"):
        if column in frame.columns:
            return frame[column].astype(str)
    raise ValueError("Holstein predictions require a stable probe identifier")


def evaluate_imagenet_only_control(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    output_suffix: str = DEFAULT_SUFFIX,
    image_size: tuple[int, int] = (384, 384),
    batch_size: int = 64,
    bootstrap_resamples: int = 2000,
) -> Path:
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    gallery = metadata[metadata["split"] == "gallery"].reset_index(drop=True)
    probe = metadata[metadata["split"] == "probe"].reset_index(drop=True)
    if gallery.empty or probe.empty:
        raise ValueError("ImageNet-only control requires non-empty gallery and probe splits")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir, output_suffix)
    model = build_imagenet_embedding_model(
        "efficientnetv2b3",
        input_shape=(image_size[0], image_size[1], 3),
    )

    def embed(frame: pd.DataFrame) -> np.ndarray:
        dataset = dataframe_to_dataset(
            frame,
            "efficientnetv2b3",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            seed=1,
            deterministic=True,
        )
        return np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)

    gallery_embeddings = embed(gallery)
    probe_embeddings = embed(probe)
    metrics, retrieval = compute_retrieval_metrics(
        gallery_embeddings,
        gallery["animal_id"].astype(str),
        probe_embeddings,
        probe["animal_id"].astype(str),
    )
    predictions = probe.copy()
    predictions["probe_sample_id"] = _stable_probe_ids(predictions)
    for column in retrieval.columns:
        if column != "animal_id":
            predictions[column] = retrieval[column].to_numpy()
    rank1_ci = bootstrap_group_mean_ci(
        predictions["correct_rank_1"].astype(float),
        predictions["animal_id"],
        n_resamples=bootstrap_resamples,
        seed=1,
    )
    map_ci = bootstrap_group_mean_ci(
        predictions["average_precision"],
        predictions["animal_id"],
        n_resamples=bootstrap_resamples,
        seed=1,
    )
    control_model = output_dir / "imagenet_only_efficientnetv2b3.keras"
    model.save(control_model)
    metrics.update(
        {
            "model": "efficientnetv2b3",
            "control_type": "imagenet_only",
            "source_training_protocol": "imagenet_only",
            "evaluation_protocol": output_suffix,
            "training_seed": 0,
            "split_seed": 1,
            "augmentation_seed": 1,
            "protocol_version": "hardening_v2",
            "metadata_path": str(metadata_path),
            "checkpoint_sha256": sha256_file(control_model),
            "checkpoint_size_bytes": int(control_model.stat().st_size),
            "rank_1_ci_low": rank1_ci["ci_low"],
            "rank_1_ci_high": rank1_ci["ci_high"],
            "map_ci_low": map_ci["ci_low"],
            "map_ci_high": map_ci["ci_high"],
            "bootstrap_unit": "animal_id",
            "bootstrap_resamples": int(bootstrap_resamples),
        }
    )
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(paths["predictions"], index=False)
    np.save(paths["gallery_embeddings"], gallery_embeddings)
    np.save(paths["probe_embeddings"], probe_embeddings)
    return paths["metrics"]


def _linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or len(first) != len(second):
        raise ValueError("linear CKA requires aligned two-dimensional embeddings")
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(first.T @ second, ord="fro") ** 2
    denominator = np.linalg.norm(first.T @ first, ord="fro") * np.linalg.norm(
        second.T @ second, ord="fro"
    )
    if not denominator:
        return 0.0
    return float(np.clip(cross / denominator, 0.0, 1.0))


def _ranking_positions(gallery: np.ndarray, probe: np.ndarray) -> np.ndarray:
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    probe = probe / np.maximum(np.linalg.norm(probe, axis=1, keepdims=True), 1e-12)
    order = np.argsort(-(probe @ gallery.T), axis=1, kind="mergesort")
    positions = np.empty_like(order)
    positions[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])
    return positions


def _mean_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("ranking correlation requires equal ranking matrices")
    correlations = []
    for left, right in zip(first, second, strict=True):
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        correlations.append(
            float(np.dot(left_centered, right_centered) / denominator) if denominator else 1.0
        )
    return float(np.mean(correlations))


def _load_artifact(run_dir: str | Path, suffix: str) -> dict[str, object]:
    run_dir = Path(run_dir)
    paths = _output_paths(run_dir, suffix)
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"incomplete Holstein audit artifact: {missing}")
    predictions = pd.read_csv(paths["predictions"])
    predictions = predictions.assign(_probe_id=_stable_probe_ids(predictions)).sort_values("_probe_id")
    order = predictions.index.to_numpy()
    predictions = predictions.reset_index(drop=True)
    probe_embeddings = np.load(paths["probe_embeddings"])[order]
    gallery_embeddings = np.load(paths["gallery_embeddings"])
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        model_path = run_dir / "imagenet_only_efficientnetv2b3.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found for Holstein artifact: {run_dir}")
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "predictions": predictions,
        "gallery_embeddings": gallery_embeddings,
        "probe_embeddings": probe_embeddings,
        "checkpoint": model_path,
        "rankings": _ranking_positions(gallery_embeddings, probe_embeddings),
    }


def build_checkpoint_audit(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    output_suffix: str = DEFAULT_SUFFIX,
) -> tuple[Path, Path]:
    if len(run_dirs) < 2:
        raise ValueError("checkpoint audit requires at least two runs")
    artifacts = [_load_artifact(run_dir, output_suffix) for run_dir in run_dirs]
    reference_ids = artifacts[0]["predictions"]["_probe_id"].tolist()
    checkpoint_rows = []
    for artifact in artifacts:
        predictions = artifact["predictions"]
        if predictions["_probe_id"].tolist() != reference_ids:
            raise ValueError("checkpoint audit requires identical ordered probe IDs")
        checkpoint = artifact["checkpoint"]
        metrics = artifact["metrics"]
        checkpoint_rows.append(
            {
                "run_id": artifact["run_dir"].name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size_bytes": int(checkpoint.stat().st_size),
                "cmc_rank_1": float(metrics["cmc_rank_1"]),
                "cmc_rank_5": float(metrics["cmc_rank_5"]),
                "mean_average_precision": float(metrics["mean_average_precision"]),
                "correct_rank_1": int(predictions["correct_rank_1"].astype(bool).sum()),
                "probe_images": int(len(predictions)),
            }
        )
    if len({row["checkpoint_sha256"] for row in checkpoint_rows}) != len(checkpoint_rows):
        raise ValueError("checkpoint audit found duplicate model SHA-256 values")

    pairwise_rows = []
    for left_index, left in enumerate(artifacts):
        for right in artifacts[left_index + 1 :]:
            left_predictions = left["predictions"]
            right_predictions = right["predictions"]
            left_correct = set(
                left_predictions.loc[left_predictions["correct_rank_1"].astype(bool), "_probe_id"]
            )
            right_correct = set(
                right_predictions.loc[right_predictions["correct_rank_1"].astype(bool), "_probe_id"]
            )
            union = left_correct | right_correct
            pairwise_rows.append(
                {
                    "run_a": left["run_dir"].name,
                    "run_b": right["run_dir"].name,
                    "correct_set_jaccard": (
                        float(len(left_correct & right_correct) / len(union)) if union else 1.0
                    ),
                    "top1_agreement": float(
                        np.mean(
                            left_predictions["predicted_animal_id"].astype(str).to_numpy()
                            == right_predictions["predicted_animal_id"].astype(str).to_numpy()
                        )
                    ),
                    "mean_ranking_spearman": _mean_spearman(
                        left["rankings"], right["rankings"]
                    ),
                    "probe_embedding_linear_cka": _linear_cka(
                        left["probe_embeddings"], right["probe_embeddings"]
                    ),
                }
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "holstein_checkpoint_audit.csv"
    pairwise_path = output_dir / "holstein_checkpoint_pairwise.csv"
    pd.DataFrame(checkpoint_rows).to_csv(checkpoint_path, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    return checkpoint_path, pairwise_path


def build_control_deltas(
    reference_dir: str | Path,
    candidate_dirs: list[str | Path],
    output_path: str | Path,
    *,
    output_suffix: str = DEFAULT_SUFFIX,
    bootstrap_resamples: int = 2000,
) -> Path:
    reference = _load_artifact(reference_dir, output_suffix)
    reference_predictions = reference["predictions"]
    rows = []
    for candidate_dir in candidate_dirs:
        candidate = _load_artifact(candidate_dir, output_suffix)
        candidate_predictions = candidate["predictions"]
        if candidate_predictions["_probe_id"].tolist() != reference_predictions["_probe_id"].tolist():
            raise ValueError("Holstein control deltas require identical ordered probe IDs")
        for metric, reference_values, candidate_values in (
            (
                "cmc_rank_1",
                reference_predictions["correct_rank_1"].astype(float),
                candidate_predictions["correct_rank_1"].astype(float),
            ),
            (
                "cmc_rank_5",
                (reference_predictions["first_correct_rank"] <= 5).astype(float),
                (candidate_predictions["first_correct_rank"] <= 5).astype(float),
            ),
            (
                "mean_average_precision",
                reference_predictions["average_precision"].astype(float),
                candidate_predictions["average_precision"].astype(float),
            ),
        ):
            interval = paired_group_mean_delta_ci(
                reference_values,
                candidate_values,
                reference_predictions["animal_id"],
                n_resamples=bootstrap_resamples,
                seed=1,
            )
            rows.append(
                {
                    "reference_run": reference["run_dir"].name,
                    "candidate_run": candidate["run_dir"].name,
                    "metric": metric,
                    "delta": interval["estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "bootstrap_unit": "animal_id",
                    "bootstrap_resamples": int(bootstrap_resamples),
                }
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Holstein2025 controls and checkpoint audits.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    imagenet = subparsers.add_parser("imagenet-control")
    imagenet.add_argument("--metadata", required=True)
    imagenet.add_argument("--out", required=True)
    audit = subparsers.add_parser("checkpoint-audit")
    audit.add_argument("--runs", nargs="+", required=True)
    audit.add_argument("--out", required=True)
    deltas = subparsers.add_parser("control-deltas")
    deltas.add_argument("--reference", required=True)
    deltas.add_argument("--candidates", nargs="+", required=True)
    deltas.add_argument("--out", required=True)
    for subparser in (imagenet, audit, deltas):
        subparser.add_argument("--output-suffix", default=DEFAULT_SUFFIX)
    args = parser.parse_args(argv)
    if args.command == "imagenet-control":
        output = evaluate_imagenet_only_control(
            args.metadata,
            args.out,
            output_suffix=args.output_suffix,
        )
        print(output)
    elif args.command == "checkpoint-audit":
        outputs = build_checkpoint_audit(
            args.runs,
            args.out,
            output_suffix=args.output_suffix,
        )
        print("\n".join(str(path) for path in outputs))
    else:
        output = build_control_deltas(
            args.reference,
            args.candidates,
            args.out,
            output_suffix=args.output_suffix,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
