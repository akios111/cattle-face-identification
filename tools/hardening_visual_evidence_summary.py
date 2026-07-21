from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "tools"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import pandas as pd

from import_hardening_visual_addendum import verify_visual_addendum


VISUAL_RUNS = (
    (
        "paper_random_hardening_v2",
        "matrix_efficientnetv2b3_paper_random_hardening_v2_train1_split1_aug1",
    ),
    (
        "transform_holdout_hardening_v2",
        "matrix_efficientnetv2b3_transform_holdout_hardening_v2_train1_split1_aug1",
    ),
)
AUC_COLUMNS = (
    "deletion_gradcam_auc",
    "deletion_random_auc",
    "insertion_gradcam_auc",
    "insertion_random_auc",
)


def summarize_visual_evidence(
    *,
    project_root: str | Path = PROJECT_ROOT,
    addendum_zip: str | Path = "artifacts/evidence/hardening_v2_visual_addendum.zip",
    source_evidence: str | Path = "artifacts/evidence/hardening_v2_evidence.zip",
    output_csv: str | Path = "thesis/tables/hardening_v2/hardening_gradcam_faithfulness.csv",
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(project_root).resolve()

    def rooted(path: str | Path) -> Path:
        path = Path(path)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    verification, _ = verify_visual_addendum(
        rooted(addendum_zip),
        source_evidence=rooted(source_evidence),
    )
    rows: list[dict[str, object]] = []
    for protocol, run_id in VISUAL_RUNS:
        path = root / "artifacts/runs" / run_id / "gradcam/gradcam_samples.csv"
        samples = pd.read_csv(path)
        missing = sorted(set(AUC_COLUMNS).difference(samples.columns))
        if missing:
            raise ValueError(f"Grad-CAM sample table missing AUC columns: {missing}")
        if len(samples) != 6:
            raise ValueError(f"Grad-CAM summary requires six samples: {run_id}")
        row: dict[str, object] = {
            "protocol": protocol,
            "run_id": run_id,
            "samples": int(len(samples)),
            "score_space": "pre_softmax_logit",
            "minimum_heatmap_nonzero_fraction": float(
                pd.to_numeric(samples["heatmap_nonzero_fraction"]).min()
            ),
        }
        for column in AUC_COLUMNS:
            values = pd.to_numeric(samples[column], errors="raise")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sample_std"] = float(values.std(ddof=1))
        row["deletion_gradcam_minus_random"] = float(
            (samples["deletion_gradcam_auc"] - samples["deletion_random_auc"]).mean()
        )
        row["insertion_gradcam_minus_random"] = float(
            (samples["insertion_gradcam_auc"] - samples["insertion_random_auc"]).mean()
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    output_path = rooted(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    tex_path = output_path.with_suffix(".tex")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Πρωτόκολλο & Deletion Grad-CAM & Deletion random & Insertion Grad-CAM & Insertion random \\",
        r"\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        label = str(row["protocol"]).replace("_hardening_v2", "").replace("_", " ")
        cells = [label]
        for column in AUC_COLUMNS:
            cells.append(
                f"{float(row[f'{column}_mean']):.3f} $\\pm$ "
                f"{float(row[f'{column}_sample_std']):.3f}"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_addendum_sha256": verification["zip_sha256"],
        "source_evidence_sha256": verification["source_evidence_sha256"],
        "rows": int(len(frame)),
        "output_csv": output_path.relative_to(root).as_posix(),
        "output_tex": tex_path.relative_to(root).as_posix(),
    }
    manifest_path = root / "artifacts/evidence/hardening_v2_visual_summary.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize verified hardening_v2 visual evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--addendum", type=Path, default="artifacts/evidence/hardening_v2_visual_addendum.zip")
    parser.add_argument("--source-evidence", type=Path, default="artifacts/evidence/hardening_v2_evidence.zip")
    parser.add_argument("--out", type=Path, default="thesis/tables/hardening_v2/hardening_gradcam_faithfulness.csv")
    args = parser.parse_args(argv)
    frame, manifest = summarize_visual_evidence(
        project_root=args.project_root,
        addendum_zip=args.addendum,
        source_evidence=args.source_evidence,
        output_csv=args.out,
    )
    print(frame.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
