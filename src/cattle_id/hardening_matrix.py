from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import pandas as pd

from .config import load_config, metadata_filename
from .hashing import sha256_file
from .run_matrix import expand_experiment_matrix, load_experiment_matrix, matrix_job_id
from .train import deterministic_run_id


CommandRunner = Callable[[list[str]], None]


def metadata_protocol_for_job(job: dict[str, object]) -> str:
    protocol = str(job["protocol"])
    if protocol.startswith("ablation_"):
        return "paper_random_hardening_v2"
    return protocol


def metadata_path_for_job(job: dict[str, object]) -> Path:
    config = load_config(str(job["base_config"]))
    metadata_dir = Path(config.get("output", {}).get("metadata_dir", "artifacts/metadata"))
    protocol = metadata_protocol_for_job(job)
    return metadata_dir / metadata_filename(
        protocol,
        split_seed=int(job["split_seed"]),
        augmentation_seed=int(job["augmentation_seed"]),
    )


def run_dir_for_job(job: dict[str, object]) -> Path:
    config = load_config(str(job["base_config"]))
    run_root = Path(config.get("output", {}).get("run_dir", "artifacts/runs"))
    run_id = deterministic_run_id(
        str(job["model"]),
        str(job["protocol"]),
        training_seed=int(job["training_seed"]),
        split_seed=int(job["split_seed"]),
        augmentation_seed=int(job["augmentation_seed"]),
    )
    return run_root / run_id


def validate_hardening_metadata(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    metadata = pd.read_csv(path)
    required = {
        "sample_id",
        "image_path",
        "source_sha256",
        "image_sha256",
        "protocol_version",
        "split_seed",
        "augmentation_seed",
        "materialization_id",
        "split",
    }
    if required.difference(metadata.columns) or metadata.empty:
        return False
    if set(metadata["protocol_version"].astype(str)) != {"hardening_v2"}:
        return False
    if metadata["sample_id"].astype(str).duplicated().any():
        return False
    return metadata["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all() and metadata[
        "image_sha256"
    ].str.fullmatch(r"[0-9a-f]{64}").all()


def classification_run_complete(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    required = (
        "run_complete.json",
        "model.keras",
        "manifest.json",
        "history.csv",
        "metrics.json",
        "predictions.csv",
    )
    if not all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in required):
        return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("protocol_version") != "hardening_v2":
        return False
    if metrics.get("protocol_version") != "hardening_v2":
        return False
    if metrics.get("model_sha256") != sha256_file(run_dir / "model.keras"):
        return False
    return True


def _default_runner(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def commands_for_job(job: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    metadata_protocol = metadata_protocol_for_job(job)
    metadata_path = metadata_path_for_job(job)
    run_dir = run_dir_for_job(job)
    run_id = run_dir.name
    common_seeds = [
        "--training-seed",
        str(job["training_seed"]),
        "--split-seed",
        str(job["split_seed"]),
        "--augmentation-seed",
        str(job["augmentation_seed"]),
    ]
    prepare = [
        sys.executable,
        "-u",
        "-m",
        "cattle_id.prepare",
        "--config",
        str(job["base_config"]),
        "--protocol",
        metadata_protocol,
        *common_seeds,
    ]
    train = [
        sys.executable,
        "-u",
        "-m",
        "cattle_id.train",
        "--config",
        str(job["base_config"]),
        "--model",
        str(job["model"]),
        "--protocol",
        str(job["protocol"]),
        "--metadata",
        str(metadata_path),
        "--run-id",
        run_id,
        "--skip-completed",
        *common_seeds,
    ]
    evaluate = [
        sys.executable,
        "-u",
        "-m",
        "cattle_id.evaluate",
        "--run",
        str(run_dir),
        "--split",
        "test",
    ]
    return prepare, train, evaluate


def execute_hardening_matrix(
    matrix_path: str | Path,
    *,
    progress_path: str | Path | None = None,
    runner: CommandRunner = _default_runner,
    dry_run: bool = False,
) -> dict[str, object]:
    matrix = load_experiment_matrix(matrix_path)
    jobs = expand_experiment_matrix(matrix)
    if len(jobs) != 29 or any(job.get("kind") != "train" for job in jobs):
        raise ValueError("hardening training contract requires exactly 29 train jobs")
    progress_path = Path(
        progress_path
        or matrix.get("output", {}).get(
            "progress_json", "artifacts/matrix/hardening_v2_progress.json"
        )
    )
    completed: list[dict[str, object]] = []
    prepared_metadata: set[Path] = set()
    planned_commands: list[list[str]] = []
    for job in jobs:
        job_id = matrix_job_id(job)
        metadata_path = metadata_path_for_job(job)
        run_dir = run_dir_for_job(job)
        prepare, train, evaluate = commands_for_job(job)
        if metadata_path not in prepared_metadata:
            if dry_run or not validate_hardening_metadata(metadata_path):
                planned_commands.append(prepare)
                if not dry_run:
                    runner(prepare)
                    if not validate_hardening_metadata(metadata_path):
                        raise RuntimeError(f"prepared metadata failed validation: {metadata_path}")
            prepared_metadata.add(metadata_path)
        if dry_run or not classification_run_complete(run_dir):
            planned_commands.extend([train, evaluate])
            if not dry_run:
                runner(train)
                runner(evaluate)
                if not classification_run_complete(run_dir):
                    raise RuntimeError(f"training job failed completion contract: {run_dir}")
        completed.append(
            {
                "job_id": job_id,
                "run_dir": str(run_dir),
                "metadata_path": str(metadata_path),
                "completed": not dry_run,
            }
        )
        if not dry_run:
            payload = {
                "version": 2,
                "expected_entries": len(jobs),
                "completed_entries": len(completed),
                "complete": len(completed) == len(jobs),
                "entries": completed,
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "jobs": len(jobs),
        "metadata_sets": len(prepared_metadata),
        "planned_commands": planned_commands,
        "completed_entries": 0 if dry_run else len(completed),
        "complete": False if dry_run else len(completed) == len(jobs),
        "progress_path": str(progress_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the 29-job hardening_v2 matrix sequentially.")
    parser.add_argument("--matrix", default="configs/experiment_matrix_hardening_v2.yaml")
    parser.add_argument("--progress")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = execute_hardening_matrix(
        args.matrix,
        progress_path=args.progress,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
