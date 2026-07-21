from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from import_hardening_evidence import import_evidence
from verify_hardening_evidence import verify_bundle


SUMMARY_TABLES = (
    "hardening_primary_runs.csv",
    "hardening_training_seed_stability.csv",
    "hardening_split_sensitivity_runs.csv",
    "hardening_split_sensitivity_summary.csv",
    "hardening_ablation_runs.csv",
    "hardening_ablation_summary.csv",
    "hardening_shortcut_summary.csv",
    "hardening_shortcut_mcnemar.csv",
    "hardening_region_audit_runs.csv",
    "hardening_region_audit_summary.csv",
    "hardening_severity_runs.csv",
    "hardening_severity_summary.csv",
    "hardening_holstein_runs.csv",
    "hardening_holstein_control_deltas.csv",
    "hardening_holstein_group_control_deltas.csv",
    "hardening_holstein_checkpoint_pairwise.csv",
    "hardening_experiment_settings.csv",
)
AUXILIARY_VERIFIED_TABLES = (
    "hardening_frozen_imagenet_sanity.csv",
    "hardening_frozen_imagenet_sanity.tex",
    "hardening_gradcam_faithfulness.csv",
    "hardening_gradcam_faithfulness.tex",
)
CommandRunner = Callable[[Sequence[str], Path], None]


def canonical_text_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compare_text_outputs(
    generated_dir: str | Path,
    expected_dir: str | Path,
    filenames: Sequence[str],
) -> list[dict[str, object]]:
    generated_dir = Path(generated_dir)
    expected_dir = Path(expected_dir)
    rows: list[dict[str, object]] = []
    for filename in filenames:
        generated = generated_dir / filename
        expected = expected_dir / filename
        if not generated.is_file():
            raise ValueError(f"release reproduction did not generate {filename}")
        if not expected.is_file():
            raise ValueError(f"verified evidence does not contain {filename}")
        generated_sha = canonical_text_sha256(generated)
        expected_sha = canonical_text_sha256(expected)
        if generated_sha != expected_sha:
            raise ValueError(f"release reproduction differs from verified evidence: {filename}")
        rows.append(
            {
                "path": filename,
                "canonical_sha256": generated_sha,
                "match": True,
            }
        )
    return rows


def compare_csv_outputs(
    generated_dir: str | Path,
    expected_dir: str | Path,
    filenames: Sequence[str],
    *,
    absolute_tolerance: float = 1e-12,
) -> list[dict[str, object]]:
    generated_dir = Path(generated_dir)
    expected_dir = Path(expected_dir)
    rows: list[dict[str, object]] = []
    for filename in filenames:
        generated = generated_dir / filename
        expected = expected_dir / filename
        if not generated.is_file():
            raise ValueError(f"release reproduction did not generate {filename}")
        if not expected.is_file():
            raise ValueError(f"verified evidence does not contain {filename}")
        generated_frame = pd.read_csv(generated)
        expected_frame = pd.read_csv(expected)
        if generated_frame.columns.tolist() != expected_frame.columns.tolist():
            raise ValueError(f"release reproduction schema differs from verified evidence: {filename}")
        if generated_frame.shape != expected_frame.shape:
            raise ValueError(f"release reproduction shape differs from verified evidence: {filename}")

        max_abs_difference = 0.0
        for column in expected_frame.columns:
            expected_values = expected_frame[column]
            generated_values = generated_frame[column]
            if pd.api.types.is_numeric_dtype(expected_values) and pd.api.types.is_numeric_dtype(
                generated_values
            ):
                expected_array = expected_values.to_numpy(dtype=float)
                generated_array = generated_values.to_numpy(dtype=float)
                finite = np.isfinite(expected_array) & np.isfinite(generated_array)
                if np.any(finite):
                    max_abs_difference = max(
                        max_abs_difference,
                        float(np.max(np.abs(expected_array[finite] - generated_array[finite]))),
                    )
                if not np.allclose(
                    expected_array,
                    generated_array,
                    rtol=0.0,
                    atol=absolute_tolerance,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"release reproduction numeric values differ from verified evidence: {filename}"
                    )
            elif not expected_values.fillna("<NA>").astype(str).equals(
                generated_values.fillna("<NA>").astype(str)
            ):
                raise ValueError(
                    f"release reproduction categorical values differ from verified evidence: {filename}"
                )

        generated_sha = canonical_text_sha256(generated)
        expected_sha = canonical_text_sha256(expected)
        rows.append(
            {
                "path": filename,
                "generated_canonical_sha256": generated_sha,
                "expected_canonical_sha256": expected_sha,
                "byte_match_after_newline_normalization": generated_sha == expected_sha,
                "max_abs_numeric_difference": max_abs_difference,
                "absolute_tolerance": absolute_tolerance,
                "match": True,
            }
        )
    return rows


def run_command(command: Sequence[str], root: Path) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(list(command), cwd=root, check=True)


def summary_commands(
    *,
    root: Path,
    tables_dir: Path,
    python: str = sys.executable,
) -> list[list[str]]:
    return [
        [
            python,
            "-u",
            "tools/hardening_evidence_summary.py",
            "--runs",
            "artifacts/runs",
            "--matrix",
            "configs/experiment_matrix_hardening_v2.yaml",
            "--out",
            str(tables_dir.relative_to(root)),
            "--verified-model-hashes",
            "artifacts/evidence/hardening_v2_MODEL_HASHES.json",
        ]
    ]


def reproduce_from_evidence(
    evidence_zip: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
    report_path: str | Path = "artifacts/reproduction/clean_clone_reproduction.json",
    runner: CommandRunner = run_command,
) -> dict[str, object]:
    root = Path(root).resolve()
    evidence_zip = Path(evidence_zip).resolve()
    try:
        evidence_source = evidence_zip.relative_to(root).as_posix()
    except ValueError:
        evidence_source = evidence_zip.name
    verification = verify_bundle(str(evidence_zip))
    imported = import_evidence(evidence_zip, root=root)

    reproduction_root = root / "artifacts" / "reproduction"
    reproduction_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hardening_v2_", dir=reproduction_root) as temporary:
        temporary_root = Path(temporary)
        tables_dir = temporary_root / "tables"
        tables_dir.mkdir(parents=True)

        commands = summary_commands(
            root=root,
            tables_dir=tables_dir,
        )
        runner(commands[0], root)
        canonical_tables = root / "thesis" / "tables" / "hardening_v2"
        table_comparisons = compare_csv_outputs(
            tables_dir,
            canonical_tables,
            SUMMARY_TABLES,
        )

        auxiliary_artifacts = []
        for filename in AUXILIARY_VERIFIED_TABLES:
            source = canonical_tables / filename
            if not source.is_file():
                raise ValueError(f"verified evidence does not contain {filename}")
            auxiliary_artifacts.append(
                {
                    "path": filename,
                    "canonical_sha256": canonical_text_sha256(source),
                    "size_bytes": source.stat().st_size,
                    "match": True,
                }
            )

    runner([sys.executable, "-u", "tools/frozen_imagenet_sanity.py", "--verify-only"], root)
    runner([sys.executable, "-u", "tools/hardening_v2_contract.py", "--strict"], root)

    payload: dict[str, object] = {
        "version": 2,
        "complete": True,
        "mode": "verified_evidence_rebuild",
        "evidence_source": evidence_source,
        "evidence_sha256": imported["source_sha256"],
        "training_jobs": verification["training_jobs"],
        "holstein_evaluations": verification["holstein_evaluations"],
        "model_hashes": verification["models"],
        "regenerated_tables": table_comparisons,
        "verified_auxiliary_artifacts": auxiliary_artifacts,
    }
    destination = root / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**payload, "report": destination.relative_to(root).as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the central hardening_v2 tables from a verified public evidence ZIP."
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reproduction/clean_clone_reproduction.json"),
    )
    args = parser.parse_args(argv)
    result = reproduce_from_evidence(
        args.evidence,
        root=args.root,
        report_path=args.report,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
