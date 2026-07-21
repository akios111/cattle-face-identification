from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_id.augmentation import apply_augmentation, get_augmentation_specs, resize_image
from cattle_id.hardening_matrix import metadata_path_for_job
from cattle_id.hashing import sha256_file
from cattle_id.holstein_audit import (
    DEFAULT_SUFFIX as HOLSTEIN_SUFFIX,
    select_diverse_rank1_errors,
)
from cattle_id.run_matrix import expand_experiment_matrix, load_experiment_matrix


DEFAULT_HOLSTEIN_ROOT = Path("/content/cattle_runtime/raw/Holstein2025")
DEFAULT_MANIFEST = Path("artifacts/matrix/hardening_v2_figure_inputs_ready.json")
RUNTIME_ROOT = Path("/content/cattle_runtime")


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=check)


def _remove_ephemeral_repository(path: Path) -> None:
    resolved = path.resolve()
    runtime_root = RUNTIME_ROOT.resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(f"refusing to remove non-runtime dataset path: {resolved}") from exc
    if resolved == runtime_root:
        raise ValueError(f"refusing to remove runtime root: {resolved}")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def ensure_pinned_repository(*, url: str, commit: str, destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists() and not (destination / ".git").is_dir():
        _remove_ephemeral_repository(destination)

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        command = [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            url,
            str(destination),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)

    _run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit])
    hooks_dir = Path("/tmp/cattle_id_empty_hooks")
    hooks_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    checkout = [
        "git",
        "-c",
        f"core.hooksPath={hooks_dir}",
        "-C",
        str(destination),
        "checkout",
        "--detach",
        commit,
    ]
    print("+", " ".join(checkout), flush=True)
    subprocess.run(checkout, check=True, env=env)
    _run(["git", "lfs", "install", "--skip-repo"], check=False)
    _run(["git", "-C", str(destination), "lfs", "pull"])

    actual_commit = _git_commit(destination)
    if actual_commit != commit:
        raise ValueError(f"dataset commit mismatch: {actual_commit} != {commit}")
    return destination


def _canonical_montage_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_file",
        "source_sha256",
        "image_path",
        "image_sha256",
        "augmentation_id",
        "dataset_commit_sha",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"hardening metadata missing montage columns: {missing}")
    source_file = metadata["source_file"].astype(str).sort_values().iloc[0]
    selected = metadata[metadata["source_file"].astype(str) == source_file].copy()
    specs = get_augmentation_specs("all")
    expected_ids = [spec.identifier for spec in specs]
    if len(selected) != len(expected_ids) or set(selected["augmentation_id"]) != set(expected_ids):
        raise ValueError("augmentation montage requires all twenty canonical variants")
    selected["augmentation_id"] = pd.Categorical(
        selected["augmentation_id"], categories=expected_ids, ordered=True
    )
    return selected.sort_values("augmentation_id").reset_index(drop=True)


def validate_cattlessfr_montage(metadata: pd.DataFrame) -> list[dict[str, str]]:
    selected = _canonical_montage_rows(metadata)
    rows = []
    for row in selected.to_dict(orient="records"):
        path = Path(str(row["image_path"]))
        expected = str(row["image_sha256"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"montage image SHA-256 mismatch: {path}: {actual} != {expected}")
        rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "augmentation_id": str(row["augmentation_id"]),
                "sha256": actual,
            }
        )
    return rows


def restore_cattlessfr_montage(
    metadata: pd.DataFrame,
    *,
    source_dir: Path,
    image_size: tuple[int, int],
    augmentation_seed: int,
    expected_commit: str,
) -> list[dict[str, str]]:
    selected = _canonical_montage_rows(metadata)
    commits = set(selected["dataset_commit_sha"].astype(str))
    if commits != {expected_commit}:
        raise ValueError(f"CattleSSFR metadata commit mismatch: {sorted(commits)}")

    source_file = str(selected.iloc[0]["source_file"])
    source_path = source_dir / source_file
    if not source_path.is_file():
        raise FileNotFoundError(f"CattleSSFR source image not found: {source_path}")
    expected_source_sha = str(selected.iloc[0]["source_sha256"])
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            f"CattleSSFR source SHA-256 mismatch: {actual_source_sha} != {expected_source_sha}"
        )

    specs = {spec.identifier: spec for spec in get_augmentation_specs("all")}
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    for row in selected.to_dict(orient="records"):
        target = Path(str(row["image_path"]))
        expected_image_sha = str(row["image_sha256"])
        if not target.is_file() or sha256_file(target) != expected_image_sha:
            target.parent.mkdir(parents=True, exist_ok=True)
            transformed = apply_augmentation(
                source,
                specs[str(row["augmentation_id"])],
                seed=augmentation_seed,
                source_id=source_file,
                protocol_version="hardening_v2",
            )
            resize_image(transformed, image_size).save(target)
        actual_image_sha = sha256_file(target)
        if actual_image_sha != expected_image_sha:
            raise ValueError(
                f"restored montage image SHA-256 mismatch: {target}: "
                f"{actual_image_sha} != {expected_image_sha}"
            )
    return validate_cattlessfr_montage(metadata)


def select_holstein_error_rows(
    *, tables_dir: Path, runs_dir: Path, limit: int = 8
) -> tuple[str, pd.DataFrame]:
    runs = pd.read_csv(tables_dir / "hardening_holstein_runs.csv")
    fine_tuned = runs[runs["control_type"].astype(str) == "fine_tuned"].sort_values(
        ["source_protocol", "training_seed", "run_id"]
    )
    if fine_tuned.empty:
        raise ValueError("Holstein error gallery requires at least one fine-tuned run")
    run_id = str(fine_tuned.iloc[0]["run_id"])
    predictions_path = runs_dir / run_id / f"predictions_{HOLSTEIN_SUFFIX}.csv"
    predictions = pd.read_csv(predictions_path)
    required = {
        "image_path",
        "relative_path",
        "sha256",
        "animal_id",
        "predicted_animal_id",
        "correct_rank_1",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Holstein predictions missing figure columns: {missing}")
    errors = select_diverse_rank1_errors(predictions, limit=limit)
    if errors.empty:
        raise ValueError("Holstein error gallery requires at least one rank-1 error")
    return run_id, errors


def validate_holstein_error_images(
    errors: pd.DataFrame, *, dataset_root: Path
) -> list[dict[str, str]]:
    rows = []
    for row in errors.to_dict(orient="records"):
        configured = Path(str(row["image_path"]))
        path = configured if configured.is_file() else dataset_root / str(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Holstein2025 figure image not found: {path}")
        expected = str(row["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Holstein2025 image SHA-256 mismatch: {path}: {actual} != {expected}")
        rows.append(
            {
                "relative_path": str(row["relative_path"]),
                "sha256": actual,
                "animal_id": str(row["animal_id"]),
                "predicted_animal_id": str(row["predicted_animal_id"]),
            }
        )
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def ensure_figure_inputs(
    *,
    matrix_path: Path,
    tables_dir: Path,
    runs_dir: Path,
    holstein_config_path: Path,
    holstein_root: Path,
    output_manifest: Path,
) -> Path:
    matrix = load_experiment_matrix(matrix_path)
    jobs = expand_experiment_matrix(matrix)
    paper_job = next(
        job
        for job in jobs
        if job["protocol"] == "paper_random_hardening_v2"
        and int(job["training_seed"]) == 1
        and int(job["split_seed"]) == 1
    )
    cattle_config_path = PROJECT_ROOT / str(matrix["base_config"])
    cattle_config = _load_yaml(cattle_config_path)
    cattle_dataset = cattle_config["dataset"]
    expected_cattle_commit = str(cattle_dataset["commit_sha"])
    paper_metadata_path = metadata_path_for_job(paper_job)
    metadata = pd.read_csv(paper_metadata_path)

    try:
        cattle_rows = validate_cattlessfr_montage(metadata)
        print("[figure-inputs] CattleSSFR montage inputs already verified", flush=True)
    except (FileNotFoundError, ValueError):
        cattle_root = ensure_pinned_repository(
            url=str(cattle_dataset["repo_url"]),
            commit=expected_cattle_commit,
            destination=Path(str(cattle_dataset["raw_dir"])),
        )
        cattle_rows = restore_cattlessfr_montage(
            metadata,
            source_dir=cattle_root / str(cattle_dataset.get("image_subdir", "cattle_images")),
            image_size=tuple(int(value) for value in cattle_config["preprocessing"]["image_size"]),
            augmentation_seed=int(paper_job["augmentation_seed"]),
            expected_commit=expected_cattle_commit,
        )
        print("[figure-inputs] restored and verified 20/20 CattleSSFR variants", flush=True)

    holstein_config = _load_yaml(holstein_config_path)
    holstein_dataset = holstein_config["dataset"]
    expected_holstein_commit = str(holstein_dataset["source_commit"])
    error_run, errors = select_holstein_error_rows(
        tables_dir=tables_dir, runs_dir=runs_dir, limit=8
    )
    try:
        holstein_rows = validate_holstein_error_images(errors, dataset_root=holstein_root)
        print("[figure-inputs] Holstein2025 error-gallery inputs already verified", flush=True)
    except (FileNotFoundError, ValueError):
        ensure_pinned_repository(
            url=str(holstein_dataset["source_url"]),
            commit=expected_holstein_commit,
            destination=holstein_root,
        )
        holstein_rows = validate_holstein_error_images(errors, dataset_root=holstein_root)
        print(
            f"[figure-inputs] restored and verified {len(holstein_rows)}/8 Holstein2025 errors",
            flush=True,
        )

    payload = {
        "protocol_version": "hardening_v2",
        "ephemeral_runtime_inputs": True,
        "raw_dataset_images_bundled": False,
        "cattlessfr": {
            "dataset_commit": expected_cattle_commit,
            "source_file": str(metadata["source_file"].astype(str).sort_values().iloc[0]),
            "verified_variants": cattle_rows,
        },
        "holstein2025": {
            "dataset_commit": expected_holstein_commit,
            "error_run_id": error_run,
            "verified_errors": holstein_rows,
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[figure-inputs] readiness manifest: {output_manifest}", flush=True)
    return output_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore hash-verified, runtime-only inputs required by hardening_v2 figures."
    )
    parser.add_argument("--matrix", type=Path, default="configs/experiment_matrix_hardening_v2.yaml")
    parser.add_argument("--tables", type=Path, default="thesis/tables/hardening_v2")
    parser.add_argument("--runs", type=Path, default="artifacts/runs")
    parser.add_argument("--holstein-config", type=Path, default="configs/holstein2025_open_set.yaml")
    parser.add_argument("--holstein-root", type=Path, default=DEFAULT_HOLSTEIN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    ensure_figure_inputs(
        matrix_path=args.matrix,
        tables_dir=args.tables,
        runs_dir=args.runs,
        holstein_config_path=args.holstein_config,
        holstein_root=args.holstein_root,
        output_manifest=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
