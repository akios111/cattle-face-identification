from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))

import hardening_v2_contract


def _write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id"])
        writer.writeheader()
        writer.writerows({"row_id": index} for index in range(count))
    path.with_suffix(".tex").write_text("table\n", encoding="utf-8")


def _write_progress(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "expected_entries": count,
                "completed_entries": count,
                "complete": True,
                "entries": [{"job_id": f"job-{index}"} for index in range(count)],
            }
        ),
        encoding="utf-8",
    )


def _complete_fixture(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "cattlessfr_hardening_v2_colab_proplus.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_version": "hardening_v2",
                "dataset": {"commit_sha": hardening_v2_contract.PINNED_DATASET_COMMIT},
            }
        ),
        encoding="utf-8",
    )
    train_jobs = [{"kind": "train", "id": index} for index in range(29)]
    eval_jobs = [{"kind": "evaluate", "id": index} for index in range(16)]
    (configs / "experiment_matrix_hardening_v2.yaml").write_text(
        yaml.safe_dump({"jobs": train_jobs}), encoding="utf-8"
    )
    (configs / "experiment_matrix_holstein_hardening_v2.yaml").write_text(
        yaml.safe_dump({"jobs": eval_jobs}), encoding="utf-8"
    )
    matrix_dir = root / "artifacts" / "matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "experiment_matrix_hardening_v2_jobs.json").write_text(
        json.dumps(train_jobs), encoding="utf-8"
    )
    (matrix_dir / "experiment_matrix_holstein_hardening_v2_jobs.json").write_text(
        json.dumps(eval_jobs), encoding="utf-8"
    )
    _write_progress(matrix_dir / "hardening_v2_progress.json", 29)
    _write_progress(matrix_dir / "holstein_hardening_v2_progress.json", 16)

    for filename, count in hardening_v2_contract.TABLE_CONTRACTS.items():
        _write_rows(root / "thesis" / "tables" / "hardening_v2" / filename, count)
    for name in hardening_v2_contract.FIGURES:
        path = root / "thesis" / "figures" / "hardening_v2" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    for name in hardening_v2_contract.FRAGMENTS:
        path = root / "thesis" / "chapters" / "generated" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fragment\n", encoding="utf-8")
    sanity_root = root / "artifacts" / "audits" / "holstein"
    sanity_root.mkdir(parents=True)
    (sanity_root / "frozen_imagenet_sanity.json").write_text(
        json.dumps(
            {
                "complete": True,
                "backbone_hashes_equal": True,
                "batchnorm_moving_statistics_equal": True,
            }
        ),
        encoding="utf-8",
    )
    bundle = root / "artifacts" / "evidence" / "hardening_v2_evidence.zip"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"verified fixture")


def _verified(_path: str) -> dict[str, object]:
    return {
        "verified": True,
        "models": 30,
        "training_jobs": 29,
        "holstein_evaluations": 16,
    }


def test_hardening_contract_accepts_exact_complete_fixture(tmp_path):
    _complete_fixture(tmp_path)

    contract = hardening_v2_contract.collect_contract(tmp_path, bundle_verifier=_verified)

    assert contract["complete"] is True
    assert contract["training_progress"]["completed"] == 29
    assert contract["holstein_progress"]["completed"] == 16
    assert all(row["ready"] for row in contract["tables"])


def test_hardening_contract_rejects_header_only_table(tmp_path):
    _complete_fixture(tmp_path)
    table = tmp_path / "thesis" / "tables" / "hardening_v2" / "hardening_primary_runs.csv"
    table.write_text("row_id\n", encoding="utf-8")

    contract = hardening_v2_contract.collect_contract(tmp_path, bundle_verifier=_verified)

    assert contract["complete"] is False
    assert contract["checks"]["tables"] is False
