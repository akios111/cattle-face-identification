from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from verify_hardening_evidence import verify_bundle


DEFAULT_OUT = PROJECT_ROOT / "thesis" / "gates" / "hardening-v2-contract.md"
PINNED_DATASET_COMMIT = "099d749e9a766ff0c9b9fbc49112c6b77b29542e"
TABLE_CONTRACTS = {
    "hardening_primary_runs.csv": 10,
    "hardening_training_seed_stability.csv": 2,
    "hardening_split_sensitivity_runs.csv": 5,
    "hardening_split_sensitivity_summary.csv": 1,
    "hardening_ablation_summary.csv": 3,
    "hardening_shortcut_summary.csv": 10,
    "hardening_shortcut_mcnemar.csv": 10,
    "hardening_region_audit_summary.csv": 5,
    "hardening_severity_summary.csv": 20,
    "hardening_holstein_runs.csv": 16,
    "hardening_holstein_control_deltas.csv": 45,
    "hardening_holstein_group_control_deltas.csv": 9,
    "hardening_holstein_checkpoint_pairwise.csv": 45,
    "hardening_frozen_imagenet_sanity.csv": 5,
}
FIGURES = (
    "pipeline.png",
    "augmentation_montage.png",
    "split_protocols.png",
    "shortcut_comparison.png",
    "learning_curves.png",
    "robustness_controls.png",
    "holstein_curves.png",
    "checkpoint_overlap.png",
    "holstein_error_gallery.png",
    "gradcam_triptych.png",
)
FRAGMENTS = (
    "hardening_v2_results.tex",
    "hardening_v2_abstract_el.tex",
    "hardening_v2_abstract_en.tex",
    "hardening_v2_discussion.tex",
    "hardening_v2_conclusion.tex",
)


def _read_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return -1


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _progress_state(path: Path, expected: int) -> dict[str, object]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"ready": False, "completed": 0, "expected": expected}
    entries = payload.get("entries", [])
    completed = int(payload.get("completed_entries", 0) or 0)
    ready = bool(
        payload.get("complete") is True
        and int(payload.get("expected_entries", 0) or 0) == expected
        and completed == expected
        and isinstance(entries, list)
        and len(entries) == expected
        and len({str(row.get("job_id", "")) for row in entries if isinstance(row, dict)}) == expected
    )
    return {"ready": ready, "completed": completed, "expected": expected}


def _matrix_state(config_path: Path, jobs_path: Path, expected: int, kind: str) -> dict[str, object]:
    config = _read_yaml(config_path)
    configured = config.get("jobs", []) if isinstance(config, dict) else []
    generated = _read_json(jobs_path)
    configured_count = len(configured) if isinstance(configured, list) else 0
    generated_count = len(generated) if isinstance(generated, list) else 0
    configured_kinds = {
        str(row.get("kind", config.get("job_defaults", {}).get("kind", "")))
        for row in configured
        if isinstance(row, dict)
    }
    generated_kinds = {str(row.get("kind", "")) for row in generated or [] if isinstance(row, dict)}
    ready = bool(
        configured_count == expected
        and generated_count == expected
        and configured_kinds == {kind}
        and generated_kinds == {kind}
    )
    return {
        "ready": ready,
        "configured": configured_count,
        "generated": generated_count,
        "expected": expected,
    }


def collect_contract(
    root: str | Path = PROJECT_ROOT,
    *,
    bundle_verifier: Callable[[str], dict[str, object]] = verify_bundle,
) -> dict[str, object]:
    root = Path(root).resolve()
    training_matrix = _matrix_state(
        root / "configs" / "experiment_matrix_hardening_v2.yaml",
        root / "artifacts" / "matrix" / "experiment_matrix_hardening_v2_jobs.json",
        29,
        "train",
    )
    holstein_matrix = _matrix_state(
        root / "configs" / "experiment_matrix_holstein_hardening_v2.yaml",
        root / "artifacts" / "matrix" / "experiment_matrix_holstein_hardening_v2_jobs.json",
        16,
        "evaluate",
    )
    training_progress = _progress_state(
        root / "artifacts" / "matrix" / "hardening_v2_progress.json", 29
    )
    holstein_progress = _progress_state(
        root / "artifacts" / "matrix" / "holstein_hardening_v2_progress.json", 16
    )

    dataset_config = _read_yaml(root / "configs" / "cattlessfr_hardening_v2_colab_proplus.yaml")
    dataset_commit = str(dataset_config.get("dataset", {}).get("commit_sha", ""))
    protocol_version = str(dataset_config.get("protocol_version", ""))
    dataset_ready = dataset_commit == PINNED_DATASET_COMMIT and protocol_version == "hardening_v2"

    table_rows: list[dict[str, object]] = []
    tables_ready = True
    tables_dir = root / "thesis" / "tables" / "hardening_v2"
    for filename, expected_rows in TABLE_CONTRACTS.items():
        csv_path = tables_dir / filename
        tex_path = csv_path.with_suffix(".tex")
        actual_rows = _csv_rows(csv_path)
        ready = actual_rows == expected_rows and _nonempty(tex_path)
        tables_ready = tables_ready and ready
        table_rows.append(
            {
                "table": filename,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "tex_present": _nonempty(tex_path),
                "ready": ready,
            }
        )

    figure_rows = [
        {"figure": name, "ready": _nonempty(root / "thesis" / "figures" / "hardening_v2" / name)}
        for name in FIGURES
    ]
    figures_ready = all(bool(row["ready"]) for row in figure_rows)
    fragment_rows = [
        {"fragment": name, "ready": _nonempty(root / "thesis" / "chapters" / "generated" / name)}
        for name in FRAGMENTS
    ]
    fragments_ready = all(bool(row["ready"]) for row in fragment_rows)

    sanity = _read_json(
        root / "artifacts" / "audits" / "holstein" / "frozen_imagenet_sanity.json"
    )
    sanity_ready = bool(
        isinstance(sanity, dict)
        and sanity.get("complete") is True
        and sanity.get("backbone_hashes_equal") is True
        and sanity.get("batchnorm_moving_statistics_equal") is True
    )

    bundle_path = root / "artifacts" / "evidence" / "hardening_v2_evidence.zip"
    bundle_ready = False
    bundle_error = "missing"
    if _nonempty(bundle_path):
        try:
            verification = bundle_verifier(str(bundle_path))
            bundle_ready = bool(
                verification.get("verified") is True
                and verification.get("training_jobs") == 29
                and verification.get("holstein_evaluations") == 16
                and verification.get("models") == 30
            )
            bundle_error = "" if bundle_ready else "verification contract mismatch"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            bundle_error = f"{type(exc).__name__}: {exc}"

    checks = {
        "dataset_pin": dataset_ready,
        "training_matrix": bool(training_matrix["ready"]),
        "holstein_matrix": bool(holstein_matrix["ready"]),
        "training_progress": bool(training_progress["ready"]),
        "holstein_progress": bool(holstein_progress["ready"]),
        "tables": tables_ready,
        "figures": figures_ready,
        "fragments": fragments_ready,
        "frozen_imagenet_sanity": sanity_ready,
        "bundle": bundle_ready,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "dataset_commit": dataset_commit,
        "training_matrix": training_matrix,
        "holstein_matrix": holstein_matrix,
        "training_progress": training_progress,
        "holstein_progress": holstein_progress,
        "tables": table_rows,
        "figures": figure_rows,
        "fragments": fragment_rows,
        "bundle_path": bundle_path.relative_to(root).as_posix(),
        "bundle_ready": bundle_ready,
        "bundle_error": bundle_error,
    }


def render_markdown(contract: dict[str, object]) -> str:
    lines = [
        "# Hardening v2 Evidence Contract",
        "",
        f"Hardening v2 complete {contract['complete']}.",
        f"Dataset commit `{contract['dataset_commit']}`.",
        f"Training progress {contract['training_progress']['completed']} / {contract['training_progress']['expected']}.",
        f"Holstein evaluations {contract['holstein_progress']['completed']} / {contract['holstein_progress']['expected']}.",
        f"Evidence bundle verified {contract['bundle_ready']}.",
        "",
        "## Checks",
        "",
        "| Check | Ready |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {ready} |" for name, ready in contract["checks"].items())
    lines.extend(["", "## Tables", "", "| Table | Rows | Expected | TEX | Ready |", "|---|---:|---:|---|---|"])
    for row in contract["tables"]:
        lines.append(
            f"| `{row['table']}` | {row['actual_rows']} | {row['expected_rows']} | {row['tex_present']} | {row['ready']} |"
        )
    if contract["bundle_error"]:
        lines.extend(["", "## Bundle Status", "", f"`{contract['bundle_error']}`"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the final hardening_v2 evidence contract.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", "-s", action="store_true")
    args = parser.parse_args(argv)
    contract = collect_contract(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(contract), encoding="utf-8")
    print(
        "hardening_v2_complete={complete} trainings={train}/{train_expected} "
        "holstein={holstein}/{holstein_expected} bundle={bundle} out={out}".format(
            complete=contract["complete"],
            train=contract["training_progress"]["completed"],
            train_expected=contract["training_progress"]["expected"],
            holstein=contract["holstein_progress"]["completed"],
            holstein_expected=contract["holstein_progress"]["expected"],
            bundle=contract["bundle_ready"],
            out=args.out,
        )
    )
    return 2 if args.strict and not contract["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
