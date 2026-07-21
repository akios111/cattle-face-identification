from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_id.augmentation import get_augmentation_specs
from cattle_id.config import load_config
from cattle_id.hardening_matrix import classification_run_complete, run_dir_for_job
from cattle_id.holstein_audit import DEFAULT_SUFFIX as HOLSTEIN_SUFFIX
from cattle_id.metrics import (
    bootstrap_seed_cluster_accuracy_ci,
    exact_mcnemar_test,
    hierarchical_paired_metric_delta_ci,
    holm_adjust,
    paired_run_group_mean_delta_ci,
)
from cattle_id.run_matrix import expand_experiment_matrix, load_experiment_matrix


def _sample_id(frame: pd.DataFrame) -> pd.Series:
    if "sample_id" not in frame.columns:
        raise ValueError("hardening predictions require sample_id")
    return frame["sample_id"].astype(str)


def _ordered_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.assign(_sample_id=_sample_id(frame)).sort_values("_sample_id").reset_index(drop=True)
    if output["_sample_id"].duplicated().any():
        raise ValueError("hardening predictions contain duplicate sample IDs")
    return output


def _align_predictions(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _ordered_predictions(reference)
    candidate = _ordered_predictions(candidate)
    if reference["_sample_id"].tolist() != candidate["_sample_id"].tolist():
        raise ValueError("paired hardening comparison requires identical ordered sample IDs")
    if reference["class_id"].astype(int).tolist() != candidate["class_id"].astype(int).tolist():
        raise ValueError("paired hardening comparison has inconsistent true labels")
    if "image_sha256" in reference.columns and "image_sha256" in candidate.columns:
        if reference["image_sha256"].astype(str).tolist() != candidate["image_sha256"].astype(str).tolist():
            raise ValueError("paired hardening comparison requires byte-identical images")
    return reference, candidate


def load_verified_model_hashes(path: str | Path) -> dict[str, str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("verified hardening model hash manifest requires 30 records")
    mapping: dict[str, str] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        sha256 = str(row.get("sha256", ""))
        if not run_id or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("verified hardening model hash manifest contains an invalid record")
        if run_id in mapping:
            raise ValueError(f"verified hardening model hash manifest duplicates {run_id}")
        mapping[run_id] = sha256
    if len(set(mapping.values())) != 30:
        raise ValueError("verified hardening model hashes must be unique")
    return mapping


def validate_classification_evidence(
    run_dir: str | Path,
    job: dict[str, object],
    model_hashes: dict[str, str],
) -> tuple[dict[str, object], pd.DataFrame]:
    run_dir = Path(run_dir)
    required = (
        "run_complete.json",
        "manifest.json",
        "history.csv",
        "metrics.json",
        "predictions.csv",
    )
    if not all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in required):
        raise ValueError(f"verified evidence run is missing required files: {run_dir}")
    try:
        complete = json.loads((run_dir / "run_complete.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"verified evidence run contains invalid JSON: {run_dir}") from exc

    expected = {
        "model": str(job["model"]),
        "protocol": str(job["protocol"]),
        "training_seed": int(job["training_seed"]),
        "split_seed": int(job["split_seed"]),
        "augmentation_seed": int(job["augmentation_seed"]),
        "protocol_version": "hardening_v2",
    }
    for field, value in expected.items():
        for label, payload in (("completion", complete), ("metrics", metrics)):
            if payload.get(field) != value:
                raise ValueError(f"verified evidence {label} {field} mismatch: {run_dir}")
        if manifest.get(field) != value:
            raise ValueError(f"verified evidence manifest {field} mismatch: {run_dir}")

    model_sha256 = str(metrics.get("model_sha256", ""))
    if model_hashes.get(run_dir.name) != model_sha256:
        raise ValueError(f"verified evidence model hash mismatch: {run_dir}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(metrics.get("test_set_sha256", ""))):
        raise ValueError(f"verified evidence test-set hash is invalid: {run_dir}")

    predictions = pd.read_csv(run_dir / "predictions.csv")
    required_prediction_columns = {"sample_id", "class_id", "predicted_class_id"}
    if predictions.empty or required_prediction_columns.difference(predictions.columns):
        raise ValueError(f"verified evidence predictions are incomplete: {run_dir}")
    if len(predictions) != int(metrics.get("samples", -1)):
        raise ValueError(f"verified evidence prediction count mismatch: {run_dir}")
    return metrics, predictions


def collect_training_artifacts(
    runs_dir: str | Path,
    matrix_path: str | Path,
    *,
    verified_model_hashes: str | Path | None = None,
) -> list[dict[str, object]]:
    jobs = expand_experiment_matrix(load_experiment_matrix(matrix_path))
    if len(jobs) != 29:
        raise ValueError("hardening evidence requires exactly 29 training jobs")
    model_hashes = (
        load_verified_model_hashes(verified_model_hashes)
        if verified_model_hashes is not None
        else None
    )
    artifacts = []
    for job in jobs:
        configured_run_dir = run_dir_for_job(job)
        run_dir = Path(runs_dir) / configured_run_dir.name
        if model_hashes is None:
            if not classification_run_complete(run_dir):
                raise ValueError(f"incomplete hardening run: {run_dir}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            predictions = pd.read_csv(run_dir / "predictions.csv")
        else:
            metrics, predictions = validate_classification_evidence(
                run_dir,
                job,
                model_hashes,
            )
        artifacts.append(
            {
                "job": job,
                "run_id": run_dir.name,
                "run_dir": run_dir,
                "metrics": metrics,
                "predictions": predictions,
            }
        )
    return artifacts


def summarize_training_stability(
    artifacts: list[dict[str, object]],
    *,
    bootstrap_resamples: int = 2000,
) -> pd.DataFrame:
    rows = []
    for protocol in ("paper_random_hardening_v2", "transform_holdout_hardening_v2"):
        selected = sorted(
            [
                artifact
                for artifact in artifacts
                if artifact["job"]["protocol"] == protocol
                and int(artifact["job"]["split_seed"]) == 1
            ],
            key=lambda artifact: int(artifact["job"]["training_seed"]),
        )
        if len(selected) != 5:
            raise ValueError(f"training stability requires five fixed-split runs for {protocol}")
        ordered = [_ordered_predictions(artifact["predictions"]) for artifact in selected]
        reference_ids = ordered[0]["_sample_id"].tolist()
        if any(frame["_sample_id"].tolist() != reference_ids for frame in ordered[1:]):
            raise ValueError(f"training stability requires identical test sample IDs for {protocol}")
        if "image_sha256" in ordered[0].columns and any(
            frame["image_sha256"].astype(str).tolist()
            != ordered[0]["image_sha256"].astype(str).tolist()
            for frame in ordered[1:]
        ):
            raise ValueError(f"training stability requires byte-identical test images for {protocol}")
        interval = bootstrap_seed_cluster_accuracy_ci(
            [
                (
                    frame["class_id"].astype(int),
                    frame["predicted_class_id"].astype(int),
                    frame["class_id"].astype(str),
                )
                for frame in ordered
            ],
            n_resamples=bootstrap_resamples,
            seed=1,
        )
        accuracy = np.asarray([float(artifact["metrics"]["accuracy"]) for artifact in selected])
        macro_f1 = np.asarray([float(artifact["metrics"]["macro_f1"]) for artifact in selected])
        rows.append(
            {
                "protocol": protocol,
                "training_seeds": 5,
                "split_seed": 1,
                "augmentation_seed": 1,
                "test_samples": len(reference_ids),
                "test_sample_alignment": "identical",
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_sample_std": float(accuracy.std(ddof=1)),
                "accuracy_ci_low": interval["ci_low"],
                "accuracy_ci_high": interval["ci_high"],
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_sample_std": float(macro_f1.std(ddof=1)),
                "bootstrap_unit": "training_seed_then_class_id",
                "bootstrap_resamples": bootstrap_resamples,
                "test_set_sha256": str(selected[0]["metrics"]["test_set_sha256"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_split_sensitivity(artifacts: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = sorted(
        [
            artifact
            for artifact in artifacts
            if artifact["job"]["protocol"] == "paper_random_hardening_v2"
            and int(artifact["job"]["training_seed"]) == 1
        ],
        key=lambda artifact: int(artifact["job"]["split_seed"]),
    )
    if len(selected) != 5 or [int(item["job"]["split_seed"]) for item in selected] != [1, 2, 3, 4, 5]:
        raise ValueError("split sensitivity requires split seeds 1..5 at training seed 1")
    rows = [
        {
            "run_id": item["run_id"],
            "split_seed": int(item["job"]["split_seed"]),
            "training_seed": 1,
            "augmentation_seed": 1,
            "samples": int(item["metrics"]["samples"]),
            "accuracy": float(item["metrics"]["accuracy"]),
            "macro_f1": float(item["metrics"]["macro_f1"]),
            "raw_errors": int(
                np.sum(
                    item["predictions"]["class_id"].astype(int)
                    != item["predictions"]["predicted_class_id"].astype(int)
                )
            ),
            "test_set_sha256": str(item["metrics"]["test_set_sha256"]),
        }
        for item in selected
    ]
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "splits": 5,
                "training_seed": 1,
                "augmentation_seed": 1,
                "accuracy_mean": float(frame["accuracy"].mean()),
                "accuracy_sample_std": float(frame["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(frame["macro_f1"].mean()),
                "macro_f1_sample_std": float(frame["macro_f1"].std(ddof=1)),
                "test_sample_alignment": "split_specific",
            }
        ]
    )
    return frame, summary


def summarize_controlled_ablations(
    artifacts: list[dict[str, object]],
    *,
    bootstrap_resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_by_seed = {
        int(artifact["job"]["training_seed"]): artifact
        for artifact in artifacts
        if artifact["job"]["protocol"] == "paper_random_hardening_v2"
        and int(artifact["job"]["split_seed"]) == 1
    }
    run_rows = []
    summary_rows = []
    for protocol in (
        "ablation_no_cutout_hardening_v2",
        "ablation_224_hardening_v2",
        "ablation_frozen_hardening_v2",
    ):
        candidates = sorted(
            [artifact for artifact in artifacts if artifact["job"]["protocol"] == protocol],
            key=lambda artifact: int(artifact["job"]["training_seed"]),
        )
        if len(candidates) != 5 or set(base_by_seed) != {1, 2, 3, 4, 5}:
            raise ValueError(f"controlled ablation requires five paired seeds for {protocol}")
        paired_runs = []
        for candidate in candidates:
            seed = int(candidate["job"]["training_seed"])
            reference, comparison = _align_predictions(
                base_by_seed[seed]["predictions"], candidate["predictions"]
            )
            y_true = reference["class_id"].astype(int).to_numpy()
            reference_pred = reference["predicted_class_id"].astype(int).to_numpy()
            candidate_pred = comparison["predicted_class_id"].astype(int).to_numpy()
            paired_runs.append((y_true, reference_pred, candidate_pred, reference["class_id"].astype(str)))
            run_rows.append(
                {
                    "protocol": protocol,
                    "training_seed": seed,
                    "samples": len(reference),
                    "base_accuracy": float(np.mean(reference_pred == y_true)),
                    "ablation_accuracy": float(np.mean(candidate_pred == y_true)),
                    "delta_accuracy": float(
                        np.mean(candidate_pred == y_true) - np.mean(reference_pred == y_true)
                    ),
                    "base_errors": int(np.sum(reference_pred != y_true)),
                    "ablation_errors": int(np.sum(candidate_pred != y_true)),
                    "discordant_errors": int(np.sum((reference_pred == y_true) != (candidate_pred == y_true))),
                    "test_set_sha256": str(candidate["metrics"]["test_set_sha256"]),
                }
            )
        accuracy_interval = hierarchical_paired_metric_delta_ci(
            paired_runs, metric="accuracy", n_resamples=bootstrap_resamples, seed=1
        )
        f1_interval = hierarchical_paired_metric_delta_ci(
            paired_runs, metric="macro_f1", n_resamples=bootstrap_resamples, seed=1
        )
        protocol_rows = [row for row in run_rows if row["protocol"] == protocol]
        summary_rows.append(
            {
                "protocol": protocol,
                "paired_seeds": 5,
                "samples_per_seed": protocol_rows[0]["samples"],
                "delta_accuracy": accuracy_interval["estimate"],
                "delta_accuracy_ci_low": accuracy_interval["ci_low"],
                "delta_accuracy_ci_high": accuracy_interval["ci_high"],
                "delta_macro_f1": f1_interval["estimate"],
                "delta_macro_f1_ci_low": f1_interval["ci_low"],
                "delta_macro_f1_ci_high": f1_interval["ci_high"],
                "base_errors_total": sum(int(row["base_errors"]) for row in protocol_rows),
                "ablation_errors_total": sum(int(row["ablation_errors"]) for row in protocol_rows),
                "discordant_errors_total": sum(int(row["discordant_errors"]) for row in protocol_rows),
                "bootstrap_unit": "training_seed_then_class_id",
                "bootstrap_resamples": bootstrap_resamples,
            }
        )
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows)


def shortcut_mcnemar(
    shortcut_predictions: dict[str, pd.DataFrame],
    cnn_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, predictions in sorted(shortcut_predictions.items()):
        reference, shortcut = _align_predictions(cnn_predictions, predictions)
        test = exact_mcnemar_test(
            reference["class_id"].astype(int),
            shortcut["predicted_class_id"].astype(int),
            reference["predicted_class_id"].astype(int),
        )
        rows.append({"method": method, **test})
    adjusted = holm_adjust([float(row["p_value_exact"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = value
    return pd.DataFrame(rows)


def experiment_settings_table(config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    rows = []
    for spec in get_augmentation_specs("all"):
        rows.append(
            {
                "section": "augmentation",
                "name": spec.identifier,
                "value": json.dumps(spec.params, sort_keys=True),
            }
        )
    for section in ("dataset", "preprocessing", "augmentation", "split", "training"):
        for key, value in config.get(section, {}).items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    rows.append(
                        {
                            "section": section,
                            "name": f"{key}.{nested_key}",
                            "value": json.dumps(nested_value, sort_keys=True),
                        }
                    )
            else:
                rows.append(
                    {
                        "section": section,
                        "name": str(key),
                        "value": json.dumps(value, sort_keys=True),
                    }
                )
    return pd.DataFrame(rows)


def _restore_robustness_columns(predictions: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Restore audit labels omitted by older evaluation prediction exports."""
    output = predictions.copy()
    if suffix == "image_region_audit" and "region_variant" not in output.columns:
        if "augmentation_id" not in output.columns:
            return output
        output["region_variant"] = output["augmentation_id"].astype(str)
    if suffix == "severity_sweep" and {
        "severity_family",
        "severity_value",
    }.difference(output.columns):
        if "augmentation_id" not in output.columns:
            return output
        parsed = output["augmentation_id"].astype(str).str.rsplit("_", n=2, expand=True)
        if parsed.shape[1] != 3:
            return output
        values = pd.to_numeric(parsed[2], errors="coerce")
        if values.isna().any() or not parsed[1].isin({"neg", "pos"}).all():
            return output
        output["severity_family"] = parsed[0]
        output["severity_value"] = values
        output["severity_direction"] = parsed[1].map({"neg": -1, "pos": 1}).astype(int)
    return output


def summarize_robustness_runs(
    artifacts: list[dict[str, object]],
    *,
    suffix: str,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = [
        artifact
        for artifact in artifacts
        if artifact["job"]["protocol"] == "paper_random_hardening_v2"
        and int(artifact["job"]["split_seed"]) == 1
    ]
    if len(selected) != 5:
        raise ValueError(f"robustness summary requires five fixed-split runs for {suffix}")
    rows = []
    for artifact in selected:
        path = artifact["run_dir"] / f"predictions_{suffix}.csv"
        if not path.is_file():
            raise ValueError(f"missing robustness predictions: {path}")
        predictions = _restore_robustness_columns(pd.read_csv(path), suffix)
        missing = sorted(set(group_columns).difference(predictions.columns))
        if missing:
            raise ValueError(f"robustness predictions missing columns: {missing}")
        for keys, group in predictions.groupby(group_columns, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            y_true = group["class_id"].astype(int)
            y_pred = group["predicted_class_id"].astype(int)
            row = {
                "run_id": artifact["run_id"],
                "training_seed": int(artifact["job"]["training_seed"]),
                "samples": int(len(group)),
                "accuracy": float(np.mean(y_true == y_pred)),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "raw_errors": int(np.sum(y_true != y_pred)),
            }
            row.update({column: value for column, value in zip(group_columns, keys, strict=True)})
            rows.append(row)
    run_frame = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in run_frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = {
            "training_seeds": int(group["training_seed"].nunique()),
            "samples_per_seed": int(group["samples"].iloc[0]),
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_sample_std": float(group["accuracy"].std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_sample_std": float(group["macro_f1"].std(ddof=1)),
            "raw_errors_total": int(group["raw_errors"].sum()),
        }
        summary.update({column: value for column, value in zip(group_columns, keys, strict=True)})
        summary_rows.append(summary)
    return run_frame, pd.DataFrame(summary_rows)


def collect_shortcut_tables(
    artifacts: list[dict[str, object]],
    *,
    audit_root: str | Path = "artifacts/audits/shortcut",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_root = Path(audit_root)
    summary_rows = []
    mcnemar_rows = []
    for protocol in ("paper_random_hardening_v2", "transform_holdout_hardening_v2"):
        protocol_root = audit_root / protocol
        summary_path = protocol_root / "shortcut_summary.csv"
        if not summary_path.is_file():
            raise ValueError(f"missing shortcut summary: {summary_path}")
        summary = pd.read_csv(summary_path)
        if len(summary) != 5:
            raise ValueError(f"shortcut summary must contain five methods: {summary_path}")
        summary.insert(0, "protocol", protocol)
        summary_rows.append(summary)
        cnn = next(
            artifact
            for artifact in artifacts
            if artifact["job"]["protocol"] == protocol
            and int(artifact["job"]["training_seed"]) == 1
            and int(artifact["job"]["split_seed"]) == 1
        )
        predictions = {
            method: pd.read_csv(protocol_root / f"predictions_{method}.csv")
            for method in summary["method"].astype(str)
        }
        tests = shortcut_mcnemar(predictions, cnn["predictions"])
        tests.insert(0, "protocol", protocol)
        tests["cnn_run_id"] = cnn["run_id"]
        mcnemar_rows.append(tests)
    return pd.concat(summary_rows, ignore_index=True), pd.concat(mcnemar_rows, ignore_index=True)


def collect_holstein_tables(
    artifacts: list[dict[str, object]],
    *,
    audit_root: str | Path = "artifacts/audits/holstein",
    runs_root: str | Path = "artifacts/runs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_artifacts = [
        artifact
        for artifact in artifacts
        if artifact["job"]["protocol"]
        in {
            "paper_random_hardening_v2",
            "transform_holdout_hardening_v2",
            "ablation_frozen_hardening_v2",
        }
        and int(artifact["job"]["split_seed"]) == 1
    ]
    rows = []
    for artifact in candidate_artifacts:
        metrics_path = artifact["run_dir"] / f"metrics_{HOLSTEIN_SUFFIX}.json"
        predictions_path = artifact["run_dir"] / f"predictions_{HOLSTEIN_SUFFIX}.csv"
        if not metrics_path.is_file():
            raise ValueError(f"missing Holstein metrics: {metrics_path}")
        if not predictions_path.is_file():
            raise ValueError(f"missing Holstein predictions: {predictions_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
        if "correct_rank_1" not in predictions.columns:
            raise ValueError(f"Holstein predictions lack correct_rank_1: {predictions_path}")
        identity_balanced = summarize_identity_balanced_holstein(predictions)
        rows.append(
            {
                "run_id": artifact["run_id"],
                "control_type": (
                    "frozen"
                    if artifact["job"]["protocol"] == "ablation_frozen_hardening_v2"
                    else "fine_tuned"
                ),
                "source_protocol": artifact["job"]["protocol"],
                "training_seed": artifact["job"]["training_seed"],
                "probe_images": int(len(predictions)),
                "correct_rank_1": int(predictions["correct_rank_1"].astype(bool).sum()),
                "cmc_rank_1": metrics["cmc_rank_1"],
                "cmc_rank_5": metrics["cmc_rank_5"],
                "mean_average_precision": metrics["mean_average_precision"],
                **identity_balanced,
                "rank_1_ci_low": metrics["rank_1_ci_low"],
                "rank_1_ci_high": metrics["rank_1_ci_high"],
                "map_ci_low": metrics["map_ci_low"],
                "map_ci_high": metrics["map_ci_high"],
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "checkpoint_size_bytes": metrics["checkpoint_size_bytes"],
            }
        )
    imagenet_dir = Path(runs_root) / "imagenet_only_efficientnetv2b3_hardening_v2"
    imagenet_metrics_path = imagenet_dir / f"metrics_{HOLSTEIN_SUFFIX}.json"
    if not imagenet_metrics_path.is_file():
        raise ValueError(f"missing ImageNet-only Holstein control: {imagenet_metrics_path}")
    imagenet = json.loads(imagenet_metrics_path.read_text(encoding="utf-8"))
    imagenet_predictions_path = imagenet_dir / f"predictions_{HOLSTEIN_SUFFIX}.csv"
    if not imagenet_predictions_path.is_file():
        raise ValueError(f"missing ImageNet-only Holstein predictions: {imagenet_predictions_path}")
    imagenet_predictions = pd.read_csv(imagenet_predictions_path)
    if "correct_rank_1" not in imagenet_predictions.columns:
        raise ValueError(
            f"ImageNet-only Holstein predictions lack correct_rank_1: {imagenet_predictions_path}"
        )
    identity_balanced = summarize_identity_balanced_holstein(imagenet_predictions)
    rows.append(
        {
            "run_id": imagenet_dir.name,
            "control_type": "imagenet_only",
            "source_protocol": "imagenet_only",
            "training_seed": 0,
            "probe_images": int(len(imagenet_predictions)),
            "correct_rank_1": int(imagenet_predictions["correct_rank_1"].astype(bool).sum()),
            "cmc_rank_1": imagenet["cmc_rank_1"],
            "cmc_rank_5": imagenet["cmc_rank_5"],
            "mean_average_precision": imagenet["mean_average_precision"],
            **identity_balanced,
            "rank_1_ci_low": imagenet["rank_1_ci_low"],
            "rank_1_ci_high": imagenet["rank_1_ci_high"],
            "map_ci_low": imagenet["map_ci_low"],
            "map_ci_high": imagenet["map_ci_high"],
            "checkpoint_sha256": imagenet["checkpoint_sha256"],
            "checkpoint_size_bytes": imagenet["checkpoint_size_bytes"],
        }
    )
    frame = pd.DataFrame(rows).sort_values(
        ["control_type", "source_protocol", "training_seed"]
    )
    if len(frame) != 16:
        raise ValueError(f"Holstein control table requires 16 rows, found {len(frame)}")
    audit_root = Path(audit_root)
    deltas = pd.read_csv(audit_root / "holstein_control_deltas.csv")
    pairwise = pd.read_csv(audit_root / "holstein_checkpoint_pairwise.csv")
    if len(deltas) != 45:
        raise ValueError(f"Holstein control delta table requires 45 rows, found {len(deltas)}")
    if len(pairwise) != 45:
        raise ValueError(f"ten-checkpoint pairwise audit requires 45 rows, found {len(pairwise)}")
    grouped_deltas = summarize_holstein_group_control_deltas(
        candidate_artifacts,
        runs_root=runs_root,
    )
    return frame.reset_index(drop=True), deltas, grouped_deltas, pairwise


def _ordered_holstein_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    id_column = next(
        (
            column
            for column in ("probe_sample_id", "sha256", "relative_path", "image_path")
            if column in frame.columns
        ),
        None,
    )
    if id_column is None:
        raise ValueError("Holstein predictions require a stable probe identifier")
    required = {
        "animal_id",
        "correct_rank_1",
        "first_correct_rank",
        "average_precision",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Holstein predictions are missing columns: {missing}")
    ordered = (
        frame.assign(_probe_id=frame[id_column].astype(str))
        .sort_values("_probe_id")
        .reset_index(drop=True)
    )
    if ordered["_probe_id"].duplicated().any():
        raise ValueError("Holstein predictions contain duplicate probe IDs")
    return ordered


def _holstein_boolean(series: pd.Series) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError("Holstein correctness column contains non-boolean values")
    return normalized.isin({"true", "1"}).to_numpy(dtype=float)


def summarize_identity_balanced_holstein(predictions: pd.DataFrame) -> dict[str, float | int]:
    ordered = _ordered_holstein_predictions(predictions)
    ranks = pd.to_numeric(ordered["first_correct_rank"], errors="raise").to_numpy(dtype=float)
    average_precision = pd.to_numeric(
        ordered["average_precision"], errors="raise"
    ).to_numpy(dtype=float)
    scored = pd.DataFrame(
        {
            "animal_id": ordered["animal_id"].astype(str),
            "rank_1": _holstein_boolean(ordered["correct_rank_1"]),
            "rank_5": (ranks <= 5).astype(float),
            "average_precision": average_precision,
        }
    )
    per_identity = scored.groupby("animal_id", sort=True, observed=True).mean(numeric_only=True)
    if len(per_identity) != 20:
        raise ValueError(
            f"identity-balanced Holstein metrics require 20 identities, found {len(per_identity)}"
        )
    return {
        "identity_count": int(len(per_identity)),
        "identity_balanced_cmc_rank_1": float(per_identity["rank_1"].mean()),
        "identity_balanced_cmc_rank_5": float(per_identity["rank_5"].mean()),
        "identity_balanced_mean_average_precision": float(
            per_identity["average_precision"].mean()
        ),
    }


def _align_holstein_predictions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _ordered_holstein_predictions(reference)
    candidate = _ordered_holstein_predictions(candidate)
    if reference["_probe_id"].tolist() != candidate["_probe_id"].tolist():
        raise ValueError("paired Holstein comparison requires identical ordered probe IDs")
    if reference["animal_id"].astype(str).tolist() != candidate["animal_id"].astype(str).tolist():
        raise ValueError("paired Holstein comparison has inconsistent animal IDs")
    return reference, candidate


def summarize_holstein_group_control_deltas(
    artifacts: list[dict[str, object]],
    *,
    runs_root: str | Path = "artifacts/runs",
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 2026,
) -> pd.DataFrame:
    imagenet_path = (
        Path(runs_root)
        / "imagenet_only_efficientnetv2b3_hardening_v2"
        / f"predictions_{HOLSTEIN_SUFFIX}.csv"
    )
    if not imagenet_path.is_file():
        raise ValueError(f"missing ImageNet-only Holstein predictions: {imagenet_path}")
    reference_predictions = pd.read_csv(imagenet_path)

    group_specs = (
        ("frozen", "ablation_frozen_hardening_v2"),
        ("fine_tuned_paper_random", "paper_random_hardening_v2"),
        ("fine_tuned_transform_holdout", "transform_holdout_hardening_v2"),
    )
    metric_extractors = (
        ("cmc_rank_1", lambda frame: _holstein_boolean(frame["correct_rank_1"])),
        (
            "cmc_rank_5",
            lambda frame: (
                pd.to_numeric(frame["first_correct_rank"], errors="raise").to_numpy()
                <= 5
            ).astype(float),
        ),
        (
            "mean_average_precision",
            lambda frame: pd.to_numeric(
                frame["average_precision"], errors="raise"
            ).to_numpy(dtype=float),
        ),
    )
    rows: list[dict[str, object]] = []
    for group_name, protocol in group_specs:
        candidates = sorted(
            (artifact for artifact in artifacts if artifact["job"]["protocol"] == protocol),
            key=lambda artifact: int(artifact["job"]["training_seed"]),
        )
        if len(candidates) != 5 or {
            int(artifact["job"]["training_seed"]) for artifact in candidates
        } != set(range(1, 6)):
            raise ValueError(f"Holstein grouped comparison requires five seeds for {protocol}")

        aligned_runs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for artifact in candidates:
            candidate_path = artifact["run_dir"] / f"predictions_{HOLSTEIN_SUFFIX}.csv"
            if not candidate_path.is_file():
                raise ValueError(f"missing Holstein predictions: {candidate_path}")
            aligned_runs.append(
                _align_holstein_predictions(
                    reference_predictions,
                    pd.read_csv(candidate_path),
                )
            )

        for metric_name, extractor in metric_extractors:
            runs = [
                (
                    extractor(reference),
                    extractor(candidate),
                    reference["animal_id"].astype(str).to_numpy(),
                )
                for reference, candidate in aligned_runs
            ]
            interval = paired_run_group_mean_delta_ci(
                runs,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "candidate_group": group_name,
                    "source_protocol": protocol,
                    "checkpoints": int(interval["runs"]),
                    "metric": metric_name,
                    "delta": interval["estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "bootstrap_unit": "animal_id",
                    "bootstrap_resamples": bootstrap_resamples,
                    "run_aggregation": interval["run_aggregation"],
                    "animals": int(interval["groups"]),
                    "probes": len(aligned_runs[0][0]),
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != 9:
        raise ValueError(f"Holstein grouped control table requires 9 rows, found {len(output)}")
    return output


def _write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> list[Path]:
    if frame.empty:
        raise ValueError(f"refusing to write header-only hardening table: {name}")
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")
    return [csv_path, tex_path]


def build_hardening_summary(
    *,
    runs_dir: str | Path = "artifacts/runs",
    matrix_path: str | Path = "configs/experiment_matrix_hardening_v2.yaml",
    output_dir: str | Path = "thesis/tables/hardening_v2",
    verified_model_hashes: str | Path | None = None,
) -> list[Path]:
    artifacts = collect_training_artifacts(
        runs_dir,
        matrix_path,
        verified_model_hashes=verified_model_hashes,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    stability = summarize_training_stability(artifacts)
    split_runs, split_summary = summarize_split_sensitivity(artifacts)
    ablation_runs, ablation_summary = summarize_controlled_ablations(artifacts)
    shortcut_summary, shortcut_tests = collect_shortcut_tables(artifacts)
    region_runs, region_summary = summarize_robustness_runs(
        artifacts,
        suffix="image_region_audit",
        group_columns=["region_variant"],
    )
    severity_runs, severity_summary = summarize_robustness_runs(
        artifacts,
        suffix="severity_sweep",
        group_columns=["severity_family", "severity_value"],
    )
    holstein_runs, holstein_deltas, holstein_group_deltas, holstein_pairwise = (
        collect_holstein_tables(artifacts)
    )
    primary_runs = pd.DataFrame(
        [
            {
                "run_id": artifact["run_id"],
                "protocol": artifact["job"]["protocol"],
                "training_seed": artifact["job"]["training_seed"],
                "split_seed": artifact["job"]["split_seed"],
                "augmentation_seed": artifact["job"]["augmentation_seed"],
                "samples": artifact["metrics"]["samples"],
                "accuracy": artifact["metrics"]["accuracy"],
                "macro_f1": artifact["metrics"]["macro_f1"],
                "test_set_sha256": artifact["metrics"]["test_set_sha256"],
                "model_sha256": artifact["metrics"]["model_sha256"],
            }
            for artifact in artifacts
            if artifact["job"]["group"] == "training_seed_variability"
        ]
    )
    for name, frame in (
        ("hardening_primary_runs", primary_runs),
        ("hardening_training_seed_stability", stability),
        ("hardening_split_sensitivity_runs", split_runs),
        ("hardening_split_sensitivity_summary", split_summary),
        ("hardening_ablation_runs", ablation_runs),
        ("hardening_ablation_summary", ablation_summary),
        ("hardening_shortcut_summary", shortcut_summary),
        ("hardening_shortcut_mcnemar", shortcut_tests),
        ("hardening_region_audit_runs", region_runs),
        ("hardening_region_audit_summary", region_summary),
        ("hardening_severity_runs", severity_runs),
        ("hardening_severity_summary", severity_summary),
        ("hardening_holstein_runs", holstein_runs),
        ("hardening_holstein_control_deltas", holstein_deltas),
        ("hardening_holstein_group_control_deltas", holstein_group_deltas),
        ("hardening_holstein_checkpoint_pairwise", holstein_pairwise),
        (
            "hardening_experiment_settings",
            experiment_settings_table("configs/cattlessfr_hardening_v2_colab_proplus.yaml"),
        ),
    ):
        generated.extend(_write_table(frame, output_dir, name))
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate hardening_v2 thesis evidence tables.")
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--matrix", default="configs/experiment_matrix_hardening_v2.yaml")
    parser.add_argument("--out", default="thesis/tables/hardening_v2")
    parser.add_argument(
        "--verified-model-hashes",
        help="Verified MODEL_HASHES.json permitting evidence-only aggregation without local checkpoints.",
    )
    args = parser.parse_args(argv)
    generated = build_hardening_summary(
        runs_dir=args.runs,
        matrix_path=args.matrix,
        output_dir=args.out,
        verified_model_hashes=args.verified_model_hashes,
    )
    print("\n".join(str(path) for path in generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
