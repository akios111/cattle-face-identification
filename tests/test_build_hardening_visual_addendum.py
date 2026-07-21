from __future__ import annotations

from pathlib import Path
import sys
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))

from build_hardening_visual_addendum import _model_expectations, validate_gradcam_run


def _write_gradcam_fixture(root: Path, *, blank_heatmap: bool = False) -> Path:
    run_dir = root / "artifacts/runs/example_run"
    gradcam_dir = run_dir / "gradcam"
    gradcam_dir.mkdir(parents=True)
    original = gradcam_dir / "sample_original.png"
    heatmap = gradcam_dir / "sample_heatmap.png"
    overlay = gradcam_dir / "sample_overlay.png"
    Image.fromarray(np.full((8, 8, 3), 96, dtype=np.uint8)).save(original)
    heatmap_array = np.zeros((8, 8), dtype=np.uint8)
    if not blank_heatmap:
        heatmap_array[2:6, 2:6] = 255
    Image.fromarray(heatmap_array).save(heatmap)
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(overlay)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "original_path": original.relative_to(root).as_posix(),
                "heatmap_path": heatmap.relative_to(root).as_posix(),
                "overlay_path": overlay.relative_to(root).as_posix(),
                "gradcam_score_space": "pre_softmax_logit",
                "heatmap_nonzero_fraction": 0.25 if not blank_heatmap else 0.01,
                "deletion_gradcam_auc": 0.3,
                "deletion_random_auc": 0.6,
            }
        ]
    ).to_csv(gradcam_dir / "gradcam_samples.csv", index=False)
    rows = []
    for curve in (
        "deletion_gradcam",
        "deletion_random",
        "insertion_gradcam",
        "insertion_random",
    ):
        for fraction in (0.0, 0.5, 1.0):
            rows.append(
                {
                    "sample_id": "sample-1",
                    "curve": curve,
                    "fraction": fraction,
                    "target_probability": 1.0 - 0.5 * fraction,
                }
            )
    pd.DataFrame(rows).to_csv(
        gradcam_dir / "gradcam_faithfulness_curves.csv",
        index=False,
    )
    return run_dir


def test_validate_gradcam_run_accepts_logit_nonblank_fixture(tmp_path: Path) -> None:
    run_dir = _write_gradcam_fixture(tmp_path)

    validation, files = validate_gradcam_run(run_dir, project_root=tmp_path)

    assert validation["samples"] == 1
    assert validation["score_space"] == "pre_softmax_logit"
    assert validation["minimum_heatmap_nonzero_pixels"] == 16
    assert len(files) == 5


def test_validate_gradcam_run_rejects_blank_heatmap(tmp_path: Path) -> None:
    run_dir = _write_gradcam_fixture(tmp_path, blank_heatmap=True)

    with pytest.raises(ValueError, match="blank Grad-CAM heatmap"):
        validate_gradcam_run(run_dir, project_root=tmp_path)


def test_model_expectations_falls_back_to_evidence_zip(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.zip"
    rows = [{"run_id": "run-1", "sha256": "abc", "size_bytes": 12}]
    with zipfile.ZipFile(evidence, "w") as archive:
        archive.writestr("MODEL_HASHES.json", json.dumps(rows))

    selected = _model_expectations(
        tmp_path / "missing.json",
        ("run-1",),
        source_evidence=evidence,
    )

    assert selected == rows
