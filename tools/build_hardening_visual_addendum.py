from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from cattle_id.hashing import sha256_file


SOURCE_EVIDENCE_SHA256 = "03cab43292e8c6f5200361035421ba375eb5a8907537e7802b45713773bae3c1"
PRIMARY_VISUAL_RUNS = (
    "matrix_efficientnetv2b3_paper_random_hardening_v2_train1_split1_aug1",
    "matrix_efficientnetv2b3_transform_holdout_hardening_v2_train1_split1_aug1",
)
REQUIRED_CURVES = {
    "deletion_gradcam",
    "deletion_random",
    "insertion_gradcam",
    "insertion_random",
}


def _project_file(path_value: object, *, root: Path, run_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
    candidates.append(run_dir / "gradcam" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"visual artifact is outside project root: {resolved}") from exc
            return resolved
    raise FileNotFoundError(path)


def validate_gradcam_run(
    run_dir: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, object], list[Path]]:
    root = Path(project_root).resolve()
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    gradcam_dir = run_dir / "gradcam"
    samples_path = gradcam_dir / "gradcam_samples.csv"
    curves_path = gradcam_dir / "gradcam_faithfulness_curves.csv"
    samples = pd.read_csv(samples_path)
    curves = pd.read_csv(curves_path)
    if samples.empty:
        raise ValueError(f"empty Grad-CAM sample table: {samples_path}")

    required_sample_columns = {
        "sample_id",
        "original_path",
        "heatmap_path",
        "overlay_path",
        "gradcam_score_space",
        "heatmap_nonzero_fraction",
    }
    missing = sorted(required_sample_columns.difference(samples.columns))
    if missing:
        raise ValueError(f"Grad-CAM samples missing columns: {missing}")
    if set(samples["gradcam_score_space"].astype(str)) != {"pre_softmax_logit"}:
        raise ValueError(f"Grad-CAM score space is not pre_softmax_logit: {run_dir.name}")
    nonzero = pd.to_numeric(samples["heatmap_nonzero_fraction"], errors="coerce")
    if nonzero.isna().any() or (nonzero <= 0).any():
        raise ValueError(f"Grad-CAM contains empty heatmaps: {run_dir.name}")

    required_curve_columns = {"sample_id", "curve", "fraction", "target_probability"}
    missing_curves = sorted(required_curve_columns.difference(curves.columns))
    if missing_curves:
        raise ValueError(f"Grad-CAM curves missing columns: {missing_curves}")
    sample_ids = set(samples["sample_id"].astype(str))
    if set(curves["sample_id"].astype(str)) != sample_ids:
        raise ValueError(f"Grad-CAM curve/sample IDs differ: {run_dir.name}")
    for sample_id, group in curves.groupby(curves["sample_id"].astype(str)):
        if set(group["curve"].astype(str)) != REQUIRED_CURVES:
            raise ValueError(f"incomplete faithfulness curves for {sample_id}")
        fractions = group.groupby("curve")["fraction"].nunique()
        if fractions.nunique() != 1 or int(fractions.iloc[0]) < 3:
            raise ValueError(f"inconsistent faithfulness steps for {sample_id}")
    probabilities = pd.to_numeric(curves["target_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite faithfulness probabilities: {run_dir.name}")

    files = [samples_path.resolve(), curves_path.resolve()]
    heatmap_nonzero_pixels: list[int] = []
    for row in samples.to_dict(orient="records"):
        original = _project_file(row["original_path"], root=root, run_dir=run_dir)
        heatmap = _project_file(row["heatmap_path"], root=root, run_dir=run_dir)
        overlay = _project_file(row["overlay_path"], root=root, run_dir=run_dir)
        with Image.open(heatmap) as opened:
            array = np.asarray(opened.convert("L"))
        count = int(np.count_nonzero(array))
        if int(array.max()) <= 0 or count == 0:
            raise ValueError(f"blank Grad-CAM heatmap: {heatmap}")
        heatmap_nonzero_pixels.append(count)
        files.extend([original, heatmap, overlay])

    auc_columns = [column for column in samples.columns if column.endswith("_auc")]
    auc_means = {
        column: float(pd.to_numeric(samples[column], errors="raise").mean())
        for column in sorted(auc_columns)
    }
    validation = {
        "run_id": run_dir.name,
        "samples": int(len(samples)),
        "score_space": "pre_softmax_logit",
        "minimum_heatmap_nonzero_fraction": float(nonzero.min()),
        "minimum_heatmap_nonzero_pixels": int(min(heatmap_nonzero_pixels)),
        "faithfulness_rows": int(len(curves)),
        "faithfulness_auc_means": auc_means,
    }
    return validation, sorted(set(files))


def _model_expectations(
    path: Path,
    run_ids: tuple[str, ...],
    *,
    source_evidence: Path | None = None,
) -> list[dict[str, object]]:
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif source_evidence is not None and source_evidence.is_file():
        with zipfile.ZipFile(source_evidence) as archive:
            rows = json.loads(archive.read("MODEL_HASHES.json").decode("utf-8"))
    else:
        raise FileNotFoundError(path)
    by_run = {str(row["run_id"]): row for row in rows}
    missing = [run_id for run_id in run_ids if run_id not in by_run]
    if missing:
        raise ValueError(f"model hash manifest missing visual runs: {missing}")
    return [by_run[run_id] for run_id in run_ids]


def build_visual_addendum(
    *,
    project_root: str | Path = PROJECT_ROOT,
    runs_dir: str | Path = "artifacts/runs",
    figures_dir: str | Path = "thesis/figures/hardening_v2",
    source_evidence: str | Path = "artifacts/evidence/hardening_v2_evidence.zip",
    model_hashes: str | Path = "artifacts/evidence/hardening_v2_MODEL_HASHES.json",
    visual_inputs: str | Path = "artifacts/evidence/hardening_v2_visual_inputs.json",
    output: str | Path = "artifacts/evidence/hardening_v2_visual_addendum.zip",
    run_ids: tuple[str, ...] = PRIMARY_VISUAL_RUNS,
    expected_source_sha256: str = SOURCE_EVIDENCE_SHA256,
) -> dict[str, object]:
    root = Path(project_root).resolve()

    def rooted(path: str | Path) -> Path:
        path = Path(path)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    runs_root = rooted(runs_dir)
    figures_root = rooted(figures_dir)
    source_path = rooted(source_evidence)
    hashes_path = rooted(model_hashes)
    inputs_path = rooted(visual_inputs)
    output_path = rooted(output)
    source_sha = sha256_file(source_path)
    if source_sha != expected_source_sha256:
        raise ValueError(f"source evidence SHA-256 mismatch: {source_sha}")

    model_rows = _model_expectations(
        hashes_path,
        run_ids,
        source_evidence=source_path,
    )
    validations: list[dict[str, object]] = []
    files: list[Path] = []
    for row in model_rows:
        run_id = str(row["run_id"])
        model_path = runs_root / run_id / "model.keras"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        actual_size = int(model_path.stat().st_size)
        actual_sha = sha256_file(model_path)
        if actual_size != int(row["size_bytes"]) or actual_sha != str(row["sha256"]):
            raise ValueError(f"checkpoint mismatch for {run_id}")
        validation, run_files = validate_gradcam_run(
            runs_root / run_id,
            project_root=root,
        )
        validations.append(validation)
        files.extend(run_files)

    for name in ("gradcam_triptych.png", "holstein_error_gallery.png"):
        figure = figures_root / name
        if not figure.is_file() or figure.stat().st_size == 0:
            raise FileNotFoundError(figure)
        files.append(figure.resolve())
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    files.append(inputs_path)

    code_paths = [
        root / "src/cattle_id/gradcam.py",
        root / "src/cattle_id/holstein_audit.py",
        root / "tools/render_hardening_figures.py",
        root / "tools/ensure_hardening_figure_inputs.py",
        root / "tools/build_hardening_visual_addendum.py",
        root / "tools/refresh_hardening_visuals_colab.py",
    ]
    files.extend(path.resolve() for path in code_paths if path.is_file())
    unique_files = sorted(set(files), key=lambda path: path.as_posix())

    entries: list[dict[str, object]] = []
    for path in unique_files:
        try:
            archive_name = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"addendum file is outside project root: {path}") from exc
        entries.append(
            {
                "path": archive_name,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )

    visual_input_payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "artifact": "hardening_v2_visual_addendum",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Corrected Grad-CAM/faithfulness and diverse Holstein error-gallery visuals only.",
        "source_evidence": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": source_sha,
            "immutable": True,
        },
        "supersedes": [
            "Grad-CAM heatmaps and faithfulness curves generated from saturated softmax scores",
            "Holstein2025 error gallery without identity-diversity selection",
        ],
        "does_not_supersede": "Training, evaluation, prediction, embedding, or central numerical-table artifacts.",
        "checkpoints": model_rows,
        "gradcam_validation": validations,
        "visual_inputs": visual_input_payload,
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    checksums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries).encode(
        "utf-8"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, entry in zip(unique_files, entries, strict=True):
            archive.write(path, arcname=str(entry["path"]))
        archive.writestr("ADDENDUM.json", manifest_bytes)
        archive.writestr("SHA256SUMS", checksums)

    zip_sha = sha256_file(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha}  {output_path.name}\n", encoding="ascii")
    return {
        "output": str(output_path),
        "sha256": zip_sha,
        "size_bytes": int(output_path.stat().st_size),
        "files": int(len(entries)),
        "gradcam_runs": int(len(validations)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the hardening_v2 visual correction addendum.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--runs", type=Path, default="artifacts/runs")
    parser.add_argument("--figures", type=Path, default="thesis/figures/hardening_v2")
    parser.add_argument("--source-evidence", type=Path, default="artifacts/evidence/hardening_v2_evidence.zip")
    parser.add_argument("--model-hashes", type=Path, default="artifacts/evidence/hardening_v2_MODEL_HASHES.json")
    parser.add_argument("--visual-inputs", type=Path, default="artifacts/evidence/hardening_v2_visual_inputs.json")
    parser.add_argument("--out", type=Path, default="artifacts/evidence/hardening_v2_visual_addendum.zip")
    args = parser.parse_args(argv)
    result = build_visual_addendum(
        project_root=args.project_root,
        runs_dir=args.runs,
        figures_dir=args.figures,
        source_evidence=args.source_evidence,
        model_hashes=args.model_hashes,
        visual_inputs=args.visual_inputs,
        output=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
