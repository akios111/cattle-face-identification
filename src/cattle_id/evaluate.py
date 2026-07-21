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

from .logging_utils import append_event, log_line, runtime_snapshot, write_json
from .metrics import compute_classification_metrics
from .tfdata import dataframe_to_dataset
from .hashing import sha256_file


@dataclass(frozen=True)
class EvaluationOutputPaths:
    metrics: Path
    confusion_matrix: Path
    predictions: Path


def evaluation_output_paths(run_dir: str | Path, output_suffix: str = "") -> EvaluationOutputPaths:
    run_dir = Path(run_dir)
    suffix = output_suffix.strip().strip("_")
    infix = f"_{suffix}" if suffix else ""
    return EvaluationOutputPaths(
        metrics=run_dir / f"metrics{infix}.json",
        confusion_matrix=run_dir / f"confusion_matrix{infix}.csv",
        predictions=run_dir / f"predictions{infix}.csv",
    )


def load_evaluation_metadata(
    manifest: dict,
    *,
    split: str = "test",
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    source = Path(metadata_path) if metadata_path is not None else Path(manifest["metadata_path"])
    metadata = pd.read_csv(source)
    if "split" not in metadata.columns:
        raise ValueError(f"Evaluation metadata is missing required column: split ({source})")
    return metadata[metadata["split"] == split].copy()


def _prediction_export_frame(split_metadata: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "sample_id",
        "original_sample_id",
        "source_file",
        "image_path",
        "source_sha256",
        "image_sha256",
        "animal_id",
        "acquisition_id",
        "class_id",
        "augmentation_id",
        "augmentation_family",
        "region_variant",
        "severity_family",
        "severity_value",
        "severity_direction",
        "protocol",
        "split",
    ]
    columns = [column for column in preferred if column in split_metadata.columns]
    if "class_id" not in columns:
        columns.append("class_id")
    if "image_path" not in columns and "image_path" in split_metadata.columns:
        columns.append("image_path")
    return split_metadata[columns].copy()


def evaluation_sample_set_sha256(split_metadata: pd.DataFrame) -> tuple[str, list[str]]:
    preferred = [
        "sample_id",
        "source_file",
        "image_sha256",
        "image_path",
        "class_id",
        "augmentation_id",
        "split",
    ]
    columns = [column for column in preferred if column in split_metadata.columns]
    if "class_id" not in columns or not ({"source_file", "image_path"} & set(columns)):
        raise ValueError("evaluation metadata requires class_id and source_file or image_path")
    records = []
    for row in split_metadata[columns].fillna("").astype(str).to_dict(orient="records"):
        records.append("\x1f".join(row[column] for column in columns))
    payload = "\n".join(sorted(records)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), columns


def evaluate_run(
    run_dir: str | Path,
    split: str = "test",
    metadata_path: str | Path | None = None,
    output_suffix: str = "",
) -> Path:
    run_dir = Path(run_dir)
    outputs = evaluation_output_paths(run_dir, output_suffix)
    log_path = run_dir / "evaluate.log"
    log_line(
        log_path,
        "evaluate",
        f"run_dir={run_dir} split={split} metadata_override={metadata_path or ''} output_suffix={output_suffix}",
    )
    append_event(run_dir, "evaluation_started", split=split, metadata_path=str(metadata_path or ""))
    write_json(run_dir / "runtime_snapshot_evaluate.json", runtime_snapshot())
    log_line(log_path, "evaluate", "loading manifest")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    effective_metadata_path = metadata_path or manifest["metadata_path"]
    log_line(log_path, "evaluate", f"reading metadata={effective_metadata_path}")
    split_metadata = load_evaluation_metadata(manifest, split=split, metadata_path=metadata_path)
    if str(manifest.get("protocol_version", "legacy")) == "hardening_v2":
        required = {"sample_id", "image_sha256", "source_sha256"}
        missing = sorted(required.difference(split_metadata.columns))
        if missing:
            raise ValueError(f"hardening_v2 evaluation metadata missing columns: {', '.join(missing)}")
    log_line(log_path, "evaluate", f"samples={len(split_metadata)} model={manifest['model']}")
    log_line(log_path, "evaluate", "loading model")
    model = tf.keras.models.load_model(run_dir / "model.keras")
    log_line(log_path, "evaluate", "building dataset")
    dataset = dataframe_to_dataset(
        split_metadata,
        manifest["model"],
        image_size=tuple(manifest["image_size"]),
        batch_size=int(manifest["batch_size"]),
        shuffle=False,
        seed=int(manifest["seed"]),
    )

    log_line(log_path, "evaluate", "predict begin")
    started = time.perf_counter()
    y_prob = model.predict(dataset)
    elapsed = time.perf_counter() - started
    log_line(log_path, "evaluate", f"predict end elapsed_seconds={elapsed:.3f}")
    y_true = split_metadata["class_id"].to_numpy()
    labels = list(range(int(manifest["num_classes"])))
    log_line(log_path, "evaluate", "computing metrics")
    result = compute_classification_metrics(y_true, y_prob, labels=labels)
    confusion = result.pop("confusion_matrix")
    sample_set_sha256, sample_id_columns = evaluation_sample_set_sha256(split_metadata)
    result.update(
        {
            "model": manifest["model"],
            "protocol": manifest["protocol"],
            "seed": int(manifest["seed"]),
            "training_seed": int(manifest.get("training_seed", manifest["seed"])),
            "split_seed": int(manifest.get("split_seed", manifest["seed"])),
            "augmentation_seed": int(manifest.get("augmentation_seed", manifest["seed"])),
            "protocol_version": str(manifest.get("protocol_version", "legacy")),
            "split": split,
            "samples": int(len(split_metadata)),
            "metadata_path": str(effective_metadata_path),
            "test_set_sha256": sample_set_sha256,
            "test_sample_id_columns": sample_id_columns,
            "output_suffix": output_suffix,
            "inference_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(1, len(split_metadata))),
            "parameter_count": int(model.count_params()),
            "model_size_bytes": int((run_dir / "model.keras").stat().st_size),
            "model_sha256": sha256_file(run_dir / "model.keras"),
        }
    )
    outputs.metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(confusion).to_csv(outputs.confusion_matrix, index=False)
    predictions = _prediction_export_frame(split_metadata)
    predictions["predicted_class_id"] = np.argmax(y_prob, axis=1)
    predictions["confidence"] = np.max(y_prob, axis=1)
    predictions.to_csv(outputs.predictions, index=False)
    log_line(log_path, "evaluate", f"metrics_written={outputs.metrics}")
    log_line(log_path, "evaluate", f"predictions_written={outputs.predictions}")
    append_event(
        run_dir,
        "evaluation_completed",
        split=split,
        samples=int(len(split_metadata)),
        inference_seconds=float(elapsed),
        metrics_path=str(outputs.metrics),
        predictions_path=str(outputs.predictions),
    )
    return outputs.metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CattleSSFR run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--metadata", help="Optional metadata CSV override for external or masked evaluation.")
    parser.add_argument("--output-suffix", default="", help="Suffix for metrics/confusion/prediction output files.")
    args = parser.parse_args(argv)
    print(evaluate_run(args.run, split=args.split, metadata_path=args.metadata, output_suffix=args.output_suffix))


if __name__ == "__main__":
    main()
