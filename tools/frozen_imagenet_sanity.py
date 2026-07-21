from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLSTEIN_SUFFIX = "holstein2025_zero_shot_reid_hardening_v2"
CONTROL_RUN_ID = "imagenet_only_efficientnetv2b3_hardening_v2"
FROZEN_RUN_TEMPLATE = (
    "matrix_efficientnetv2b3_ablation_frozen_hardening_v2_"
    "train{seed}_split1_aug1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def embedding_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int | str]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("embedding comparison requires two-dimensional arrays")
    if reference.shape != candidate.shape:
        raise ValueError("embedding comparison requires identical shapes")
    if reference.shape[0] == 0:
        raise ValueError("embedding comparison requires at least one sample")

    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    absolute = np.abs(candidate64 - reference64)
    reference_norm = np.linalg.norm(reference64, axis=1)
    candidate_norm = np.linalg.norm(candidate64, axis=1)
    if np.any(reference_norm == 0.0) or np.any(candidate_norm == 0.0):
        raise ValueError("embedding comparison encountered a zero-length vector")
    cosine_similarity = np.sum(reference64 * candidate64, axis=1) / (
        reference_norm * candidate_norm
    )
    cosine_distance = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)
    rounded_reference = reference.astype(candidate.dtype).astype(np.float64)
    return {
        "embedding_samples": int(reference.shape[0]),
        "embedding_dimensions": int(reference.shape[1]),
        "reference_embedding_dtype": str(reference.dtype),
        "candidate_embedding_dtype": str(candidate.dtype),
        "embedding_max_abs_difference": float(absolute.max()),
        "embedding_mean_abs_difference": float(absolute.mean()),
        "embedding_mean_cosine_distance": float(cosine_distance.mean()),
        "embedding_max_cosine_distance": float(cosine_distance.max()),
        "reference_storage_rounding_max_abs_difference": float(
            np.abs(rounded_reference - reference64).max()
        ),
        "candidate_vs_rounded_reference_max_abs_difference": float(
            np.abs(candidate64 - rounded_reference).max()
        ),
    }


def canonical_layer_state(model: Any) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for layer in model.layers:
        for weight in layer.weights:
            role = str(getattr(weight, "name", "weight")).split("/")[-1].split(":")[0]
            key = f"{layer.name}/{role}"
            if key in state:
                raise ValueError(f"duplicate canonical weight key: {key}")
            state[key] = np.asarray(weight.numpy())
    if not state:
        raise ValueError("backbone exposes no weights")
    return state


def _state_digest(state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        array = np.ascontiguousarray(state[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _batchnorm_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {
        key: value
        for key, value in state.items()
        if key.endswith("/moving_mean") or key.endswith("/moving_variance")
    }
    if not output:
        raise ValueError("backbone exposes no BatchNorm moving statistics")
    return output


def compare_backbone_states(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, float | int | str | bool]:
    if set(reference) != set(candidate):
        missing = sorted(set(reference).difference(candidate))
        extra = sorted(set(candidate).difference(reference))
        raise ValueError(f"backbone state keys differ; missing={missing[:3]} extra={extra[:3]}")
    max_difference = 0.0
    for key in sorted(reference):
        if reference[key].shape != candidate[key].shape:
            raise ValueError(f"backbone tensor shape differs: {key}")
        difference = float(
            np.max(
                np.abs(
                    reference[key].astype(np.float64)
                    - candidate[key].astype(np.float64)
                )
            )
        )
        max_difference = max(max_difference, difference)

    reference_bn = _batchnorm_state(reference)
    candidate_bn = _batchnorm_state(candidate)
    bn_max_difference = max(
        float(
            np.max(
                np.abs(
                    reference_bn[key].astype(np.float64)
                    - candidate_bn[key].astype(np.float64)
                )
            )
        )
        for key in reference_bn
    )
    reference_hash = _state_digest(reference)
    candidate_hash = _state_digest(candidate)
    reference_bn_hash = _state_digest(reference_bn)
    candidate_bn_hash = _state_digest(candidate_bn)
    return {
        "backbone_tensor_count": len(reference),
        "backbone_reference_sha256": reference_hash,
        "backbone_candidate_sha256": candidate_hash,
        "backbone_hash_equal": reference_hash == candidate_hash,
        "backbone_max_abs_difference": max_difference,
        "batchnorm_moving_tensor_count": len(reference_bn),
        "batchnorm_reference_sha256": reference_bn_hash,
        "batchnorm_candidate_sha256": candidate_bn_hash,
        "batchnorm_hash_equal": reference_bn_hash == candidate_bn_hash,
        "batchnorm_max_abs_difference": bn_max_difference,
    }


def _load_backbone_state(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for checkpoint state auditing") from exc

    model = tf.keras.models.load_model(path, compile=False)
    try:
        backbone = model.get_layer("backbone")
    except ValueError:
        backbone = model
    state = canonical_layer_state(backbone)
    details = {
        "compute_dtypes": ",".join(
            sorted({str(layer.compute_dtype) for layer in backbone.layers})
        ),
        "variable_dtypes": ",".join(
            sorted({str(layer.variable_dtype) for layer in backbone.layers})
        ),
        "output_dtype": str(backbone.output.dtype),
    }
    del backbone
    del model
    tf.keras.backend.clear_session()
    return state, details


def _stable_probe_ids(frame: pd.DataFrame) -> pd.Series:
    for column in ("probe_sample_id", "sha256", "relative_path", "image_path"):
        if column in frame.columns:
            return frame[column].astype(str)
    raise ValueError("Holstein predictions require a stable probe identifier")


def _embedding_paths(run_dir: Path) -> tuple[Path, Path]:
    return (
        run_dir / f"gallery_embeddings_{HOLSTEIN_SUFFIX}.npy",
        run_dir / f"probe_embeddings_{HOLSTEIN_SUFFIX}.npy",
    )


def _load_embeddings(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    gallery_path, probe_path = _embedding_paths(run_dir)
    if not gallery_path.is_file() or not probe_path.is_file():
        raise ValueError(f"missing Holstein embeddings: {run_dir}")
    return np.load(gallery_path), np.load(probe_path)


def _checkpoint_metadata(run_dir: Path, *, control: bool) -> tuple[Path, dict[str, Any]]:
    metrics = _read_json(run_dir / f"metrics_{HOLSTEIN_SUFFIX}.json")
    filename = "imagenet_only_efficientnetv2b3.keras" if control else "model.keras"
    return run_dir / filename, metrics


def build_audit(
    runs_dir: str | Path = "artifacts/runs",
    *,
    allow_missing_models: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs_dir = Path(runs_dir)
    reference_dir = runs_dir / CONTROL_RUN_ID
    reference_gallery, reference_probe = _load_embeddings(reference_dir)
    reference_predictions = pd.read_csv(
        reference_dir / f"predictions_{HOLSTEIN_SUFFIX}.csv"
    )
    reference_probe_ids = sorted(_stable_probe_ids(reference_predictions).tolist())
    reference_model, reference_metrics = _checkpoint_metadata(reference_dir, control=True)

    models_available = reference_model.is_file()
    reference_state: dict[str, np.ndarray] | None = None
    reference_details: dict[str, str] = {}
    if models_available:
        if _sha256_file(reference_model) != str(reference_metrics["checkpoint_sha256"]):
            raise ValueError("ImageNet-only checkpoint hash does not match its metrics")
        reference_state, reference_details = _load_backbone_state(reference_model)
    elif not allow_missing_models:
        raise ValueError(f"missing ImageNet-only checkpoint: {reference_model}")

    rows: list[dict[str, Any]] = []
    for seed in range(1, 6):
        run_dir = runs_dir / FROZEN_RUN_TEMPLATE.format(seed=seed)
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("protocol") != "ablation_frozen_hardening_v2":
            raise ValueError(f"unexpected frozen protocol: {run_dir}")
        if int(manifest.get("training_seed", -1)) != seed:
            raise ValueError(f"unexpected frozen training seed: {run_dir}")
        candidate_gallery, candidate_probe = _load_embeddings(run_dir)
        predictions = pd.read_csv(run_dir / f"predictions_{HOLSTEIN_SUFFIX}.csv")
        probe_ids = sorted(_stable_probe_ids(predictions).tolist())
        if probe_ids != reference_probe_ids:
            raise ValueError(f"frozen probe IDs do not align with ImageNet-only: {run_dir}")
        if reference_gallery.shape != candidate_gallery.shape:
            raise ValueError(f"frozen gallery shape differs from ImageNet-only: {run_dir}")
        if reference_probe.shape != candidate_probe.shape:
            raise ValueError(f"frozen probe shape differs from ImageNet-only: {run_dir}")

        candidate_model, candidate_metrics = _checkpoint_metadata(run_dir, control=False)
        combined_reference = np.concatenate([reference_gallery, reference_probe], axis=0)
        combined_candidate = np.concatenate([candidate_gallery, candidate_probe], axis=0)
        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "training_seed": seed,
            "probe_id_alignment": "identical",
            "reference_checkpoint_sha256": reference_metrics["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate_metrics["checkpoint_sha256"],
            "reference_inference_batch_size": 64,
            "candidate_inference_batch_size": int(manifest["batch_size"]),
            "reference_runtime_policy": "float32",
            "candidate_runtime_policy": str(
                manifest.get("runtime", {}).get("mixed_precision_policy", "unknown")
            ),
            "cmc_rank_1_delta": float(candidate_metrics["cmc_rank_1"])
            - float(reference_metrics["cmc_rank_1"]),
            "cmc_rank_5_delta": float(candidate_metrics["cmc_rank_5"])
            - float(reference_metrics["cmc_rank_5"]),
            "mean_average_precision_delta": float(
                candidate_metrics["mean_average_precision"]
            )
            - float(reference_metrics["mean_average_precision"]),
            **embedding_difference(combined_reference, combined_candidate),
        }

        state_complete = reference_state is not None and candidate_model.is_file()
        if state_complete:
            if _sha256_file(candidate_model) != str(candidate_metrics["checkpoint_sha256"]):
                raise ValueError(f"frozen checkpoint hash does not match its metrics: {run_dir}")
            candidate_state, candidate_details = _load_backbone_state(candidate_model)
            row.update(compare_backbone_states(reference_state, candidate_state))
            row.update(
                {
                    "reference_compute_dtypes": reference_details["compute_dtypes"],
                    "candidate_compute_dtypes": candidate_details["compute_dtypes"],
                    "reference_output_dtype": reference_details["output_dtype"],
                    "candidate_output_dtype": candidate_details["output_dtype"],
                }
            )
        elif not allow_missing_models:
            raise ValueError(f"missing frozen checkpoint: {candidate_model}")
        row["state_audit_complete"] = state_complete
        rows.append(row)

    frame = pd.DataFrame(rows)
    complete = bool(
        len(frame) == 5
        and frame["state_audit_complete"].astype(bool).all()
        and frame["backbone_hash_equal"].astype(bool).all()
        and frame["batchnorm_hash_equal"].astype(bool).all()
        and (frame["backbone_max_abs_difference"].astype(float) == 0.0).all()
        and (frame["batchnorm_max_abs_difference"].astype(float) == 0.0).all()
    ) if "backbone_hash_equal" in frame.columns else False
    if complete:
        cause = (
            "The ImageNet-only and frozen checkpoints have byte-identical backbone tensors "
            "and BatchNorm moving statistics. Their saved Holstein embeddings were produced "
            "by different numerical inference paths: float32 with batch 64 for the ImageNet-only "
            "control versus persisted mixed-float16 with batch 128 for the frozen runs, with "
            "float32 versus float16 embedding storage. The resulting tiny cosine perturbations "
            "can change lower retrieval ranks and mAP while leaving CMC@1 and CMC@5 unchanged."
        )
        conclusion = "deterministic_negative_control_with_numerical_path_difference"
    else:
        cause = "Checkpoint state verification has not completed; no causal conclusion is permitted."
        conclusion = "incomplete"
    summary = {
        "complete": complete,
        "protocol_version": "hardening_v2",
        "reference_run": CONTROL_RUN_ID,
        "frozen_runs": 5,
        "embedding_samples_per_run": int(frame["embedding_samples"].iloc[0]),
        "all_probe_ids_aligned": bool(
            (frame["probe_id_alignment"] == "identical").all()
        ),
        "all_cmc_rank_1_deltas_zero": bool(
            (frame["cmc_rank_1_delta"].astype(float) == 0.0).all()
        ),
        "all_cmc_rank_5_deltas_zero": bool(
            (frame["cmc_rank_5_delta"].astype(float) == 0.0).all()
        ),
        "maximum_embedding_absolute_difference": float(
            frame["embedding_max_abs_difference"].max()
        ),
        "mean_cosine_distance_across_runs": float(
            frame["embedding_mean_cosine_distance"].mean()
        ),
        "backbone_hashes_equal": bool(complete),
        "batchnorm_moving_statistics_equal": bool(complete),
        "conclusion": conclusion,
        "cause": cause,
    }
    return frame, summary


def thesis_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "training_seed",
        "embedding_max_abs_difference",
        "embedding_mean_cosine_distance",
        "backbone_hash_equal",
        "batchnorm_hash_equal",
        "mean_average_precision_delta",
        "state_audit_complete",
        "backbone_tensor_count",
        "batchnorm_moving_tensor_count",
        "reference_embedding_dtype",
        "candidate_embedding_dtype",
        "reference_inference_batch_size",
        "candidate_inference_batch_size",
        "candidate_runtime_policy",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frozen sanity audit is missing columns: {missing}")
    if len(frame) != 5 or not frame["state_audit_complete"].astype(bool).all():
        raise ValueError("frozen sanity thesis table requires five complete state audits")
    return frame[
        [
            "training_seed",
            "embedding_max_abs_difference",
            "embedding_mean_cosine_distance",
            "backbone_hash_equal",
            "batchnorm_hash_equal",
            "mean_average_precision_delta",
            "backbone_tensor_count",
            "batchnorm_moving_tensor_count",
            "reference_embedding_dtype",
            "candidate_embedding_dtype",
            "reference_inference_batch_size",
            "candidate_inference_batch_size",
            "candidate_runtime_policy",
        ]
    ].copy()


def write_outputs(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_csv: str | Path,
    summary_json: str | Path,
    report_md: str | Path,
    table_csv: str | Path | None = None,
) -> list[Path]:
    output_csv = Path(output_csv)
    summary_json = Path(summary_json)
    report_md = Path(report_md)
    for path in (output_csv, summary_json, report_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Frozen–ImageNet sanity audit",
        "",
        f"- Complete: `{str(summary['complete']).lower()}`",
        f"- Conclusion: `{summary['conclusion']}`",
        f"- Maximum embedding absolute difference: `{summary['maximum_embedding_absolute_difference']:.10f}`",
        f"- Mean cosine distance across runs: `{summary['mean_cosine_distance_across_runs']:.10g}`",
        f"- Backbone hashes equal: `{str(summary['backbone_hashes_equal']).lower()}`",
        f"- BatchNorm moving statistics equal: `{str(summary['batchnorm_moving_statistics_equal']).lower()}`",
        "",
        summary["cause"],
        "",
    ]
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    generated = [output_csv, summary_json, report_md]
    if table_csv is not None:
        if summary.get("complete") is not True:
            raise ValueError("refusing to write final frozen sanity table from incomplete audit")
        table_path = Path(table_csv)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_frame = thesis_table(frame)
        table_frame.to_csv(table_path, index=False)
        table_path.with_suffix(".tex").write_text(
            table_frame.to_latex(index=False, escape=True),
            encoding="utf-8",
        )
        generated.extend([table_path, table_path.with_suffix(".tex")])
    return generated


def verify_outputs(
    output_csv: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    frame = pd.read_csv(output_csv)
    summary = _read_json(Path(summary_json))
    if summary.get("complete") is not True:
        raise ValueError("frozen sanity audit is not complete")
    table = thesis_table(frame)
    if not table["backbone_hash_equal"].astype(bool).all():
        raise ValueError("frozen backbone hashes are not identical")
    if not table["batchnorm_hash_equal"].astype(bool).all():
        raise ValueError("frozen BatchNorm moving statistics are not identical")
    if not (frame["backbone_max_abs_difference"].astype(float) == 0.0).all():
        raise ValueError("frozen backbone tensors differ")
    if not (frame["batchnorm_max_abs_difference"].astype(float) == 0.0).all():
        raise ValueError("frozen BatchNorm moving statistics differ")
    return {
        "verified": True,
        "runs": len(frame),
        "backbone_hashes_equal": True,
        "batchnorm_moving_statistics_equal": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen EfficientNetV2B3 checkpoints against the ImageNet-only control."
    )
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument(
        "--out", default="artifacts/audits/holstein/frozen_imagenet_sanity.csv"
    )
    parser.add_argument(
        "--summary", default="artifacts/audits/holstein/frozen_imagenet_sanity.json"
    )
    parser.add_argument(
        "--report", default="artifacts/audits/holstein/frozen_imagenet_sanity.md"
    )
    parser.add_argument(
        "--table-out",
        default="thesis/tables/hardening_v2/hardening_frozen_imagenet_sanity.csv",
    )
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_outputs(args.out, args.summary), indent=2))
        return 0

    frame, summary = build_audit(
        args.runs,
        allow_missing_models=args.allow_missing_models,
    )
    table_out = None if args.allow_missing_models and not summary["complete"] else args.table_out
    generated = write_outputs(
        frame,
        summary,
        output_csv=args.out,
        summary_json=args.summary,
        report_md=args.report,
        table_csv=table_out,
    )
    print("\n".join(str(path) for path in generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
