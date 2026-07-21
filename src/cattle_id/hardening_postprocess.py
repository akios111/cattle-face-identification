from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import pandas as pd

from .hardening_matrix import (
    classification_run_complete,
    metadata_path_for_job,
    run_dir_for_job,
)
from .open_set_evaluate import is_open_set_evaluation_complete, open_set_output_paths
from .run_matrix import expand_experiment_matrix, load_experiment_matrix, matrix_job_id


CommandRunner = Callable[[list[str]], None]
HOLSTEIN_SUFFIX = "holstein2025_zero_shot_reid_hardening_v2"


def _default_runner(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _classification_eval_complete(run_dir: Path, suffix: str) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            run_dir / f"metrics_{suffix}.json",
            run_dir / f"predictions_{suffix}.csv",
            run_dir / f"confusion_matrix_{suffix}.csv",
        )
    )


def _jobs_by_protocol(jobs: list[dict[str, object]], protocol: str) -> list[dict[str, object]]:
    return sorted(
        [job for job in jobs if job["protocol"] == protocol and int(job["split_seed"]) == 1],
        key=lambda job: int(job["training_seed"]),
    )


def execute_hardening_postprocess(
    *,
    training_matrix: str | Path = "configs/experiment_matrix_hardening_v2.yaml",
    holstein_metadata: str | Path = "artifacts/metadata/holstein2025_open_set.csv",
    progress_path: str | Path = "artifacts/matrix/holstein_hardening_v2_progress.json",
    runner: CommandRunner = _default_runner,
    dry_run: bool = False,
) -> dict[str, object]:
    jobs = expand_experiment_matrix(load_experiment_matrix(training_matrix))
    if len(jobs) != 29:
        raise ValueError("hardening postprocess requires the 29-job training matrix")
    if not dry_run:
        incomplete = [str(run_dir_for_job(job)) for job in jobs if not classification_run_complete(run_dir_for_job(job))]
        if incomplete:
            raise RuntimeError(f"hardening training matrix is incomplete: {incomplete}")

    paper_jobs = _jobs_by_protocol(jobs, "paper_random_hardening_v2")
    transform_jobs = _jobs_by_protocol(jobs, "transform_holdout_hardening_v2")
    frozen_jobs = _jobs_by_protocol(jobs, "ablation_frozen_hardening_v2")
    if not (len(paper_jobs) == len(transform_jobs) == len(frozen_jobs) == 5):
        raise ValueError("hardening postprocess could not resolve 5/5/5 primary and frozen jobs")
    paper_metadata = metadata_path_for_job(paper_jobs[0])
    transform_metadata = metadata_path_for_job(transform_jobs[0])
    commands: list[list[str]] = []

    for protocol_name, metadata_path in (
        ("paper_random_hardening_v2", paper_metadata),
        ("transform_holdout_hardening_v2", transform_metadata),
    ):
        output_dir = Path("artifacts/audits/shortcut") / protocol_name
        summary = output_dir / "shortcut_summary.csv"
        if dry_run or not summary.is_file() or len(pd.read_csv(summary)) != 5:
            command = [
                sys.executable,
                "-u",
                "-m",
                "cattle_id.shortcut_audit",
                "--metadata",
                str(metadata_path),
                "--out",
                str(output_dir),
            ]
            commands.append(command)
            if not dry_run:
                runner(command)

    robustness_root = Path("artifacts/audits/robustness")
    region_metadata = robustness_root / "metadata_region_audit.csv"
    severity_metadata = robustness_root / "metadata_severity_sweep.csv"
    if dry_run or not region_metadata.is_file() or len(pd.read_csv(region_metadata)) != 9330:
        command = [
            sys.executable,
            "-u",
            "-m",
            "cattle_id.robustness",
            "--metadata",
            str(paper_metadata),
            "--out",
            str(robustness_root / "region_images"),
            "--mode",
            "region",
            "--metadata-out",
            str(region_metadata),
        ]
        commands.append(command)
        if not dry_run:
            runner(command)
    if dry_run or not severity_metadata.is_file() or len(pd.read_csv(severity_metadata)) != 7464:
        command = [
            sys.executable,
            "-u",
            "-m",
            "cattle_id.robustness",
            "--metadata",
            str(paper_metadata),
            "--out",
            str(robustness_root / "severity_images"),
            "--mode",
            "severity",
            "--metadata-out",
            str(severity_metadata),
        ]
        commands.append(command)
        if not dry_run:
            runner(command)
    for job in paper_jobs:
        run_dir = run_dir_for_job(job)
        for suffix, metadata_path in (
            ("image_region_audit", region_metadata),
            ("severity_sweep", severity_metadata),
        ):
            if dry_run or not _classification_eval_complete(run_dir, suffix):
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "cattle_id.evaluate",
                    "--run",
                    str(run_dir),
                    "--metadata",
                    str(metadata_path),
                    "--split",
                    "test",
                    "--output-suffix",
                    suffix,
                ]
                commands.append(command)
                if not dry_run:
                    runner(command)

    for job in (paper_jobs[0], transform_jobs[0]):
        run_dir = run_dir_for_job(job)
        faithfulness = run_dir / "gradcam" / "gradcam_faithfulness_curves.csv"
        if dry_run or not faithfulness.is_file() or faithfulness.stat().st_size == 0:
            command = [
                sys.executable,
                "-u",
                "-m",
                "cattle_id.gradcam",
                "--run",
                str(run_dir),
                "--samples",
                "curated",
                "--limit",
                "6",
                "--faithfulness-steps",
                "20",
            ]
            commands.append(command)
            if not dry_run:
                runner(command)

    holstein_metadata = Path(holstein_metadata)
    imagenet_dir = Path("artifacts/runs/imagenet_only_efficientnetv2b3_hardening_v2")
    imagenet_outputs = open_set_output_paths(imagenet_dir, HOLSTEIN_SUFFIX)
    if dry_run or not is_open_set_evaluation_complete(imagenet_outputs):
        command = [
            sys.executable,
            "-u",
            "-m",
            "cattle_id.holstein_audit",
            "imagenet-control",
            "--metadata",
            str(holstein_metadata),
            "--out",
            str(imagenet_dir),
            "--output-suffix",
            HOLSTEIN_SUFFIX,
        ]
        commands.append(command)
        if not dry_run:
            runner(command)

    candidate_jobs = [*frozen_jobs, *paper_jobs, *transform_jobs]
    holstein_entries = [
        {"job_id": "evaluate_imagenet_only", "run_dir": str(imagenet_dir), "control_type": "imagenet_only"}
    ]
    for job in candidate_jobs:
        run_dir = run_dir_for_job(job)
        outputs = open_set_output_paths(run_dir, HOLSTEIN_SUFFIX)
        if dry_run or not is_open_set_evaluation_complete(outputs):
            command = [
                sys.executable,
                "-u",
                "-m",
                "cattle_id.open_set_evaluate",
                "--run",
                str(run_dir),
                "--metadata",
                str(holstein_metadata),
                "--output-suffix",
                HOLSTEIN_SUFFIX,
                "--skip-existing",
            ]
            commands.append(command)
            if not dry_run:
                runner(command)
        holstein_entries.append(
            {
                "job_id": f"evaluate_{matrix_job_id(job)}",
                "run_dir": str(run_dir),
                "control_type": "frozen" if job in frozen_jobs else "fine_tuned",
            }
        )

    main_run_dirs = [str(run_dir_for_job(job)) for job in [*paper_jobs, *transform_jobs]]
    audit_dir = Path("artifacts/audits/holstein")
    audit_command = [
        sys.executable,
        "-u",
        "-m",
        "cattle_id.holstein_audit",
        "checkpoint-audit",
        "--runs",
        *main_run_dirs,
        "--out",
        str(audit_dir),
        "--output-suffix",
        HOLSTEIN_SUFFIX,
    ]
    delta_command = [
        sys.executable,
        "-u",
        "-m",
        "cattle_id.holstein_audit",
        "control-deltas",
        "--reference",
        str(imagenet_dir),
        "--candidates",
        *[str(run_dir_for_job(job)) for job in candidate_jobs],
        "--out",
        str(audit_dir / "holstein_control_deltas.csv"),
        "--output-suffix",
        HOLSTEIN_SUFFIX,
    ]
    commands.extend([audit_command, delta_command])
    if not dry_run:
        runner(audit_command)
        runner(delta_command)
        progress = {
            "version": 2,
            "expected_entries": 16,
            "completed_entries": len(holstein_entries),
            "complete": len(holstein_entries) == 16,
            "entries": holstein_entries,
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        progress_path = Path(progress_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    return {
        "training_jobs": len(jobs),
        "holstein_evaluations": 16,
        "planned_commands": commands,
        "complete": not dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hardening shortcut, robustness and Holstein audits.")
    parser.add_argument("--training-matrix", default="configs/experiment_matrix_hardening_v2.yaml")
    parser.add_argument("--holstein-metadata", default="artifacts/metadata/holstein2025_open_set.csv")
    parser.add_argument("--progress", default="artifacts/matrix/holstein_hardening_v2_progress.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = execute_hardening_postprocess(
        training_matrix=args.training_matrix,
        holstein_metadata=args.holstein_metadata,
        progress_path=args.progress,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
