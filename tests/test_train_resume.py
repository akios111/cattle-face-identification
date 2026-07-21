from __future__ import annotations

from pathlib import Path

from cattle_id.train import deterministic_run_id, is_run_complete


def test_deterministic_run_id_is_stable_and_path_safe():
    assert deterministic_run_id("ConvNeXtTiny", "holstein2025 closed set", 3) == (
        "matrix_convnexttiny_holstein2025_closed_set_seed3"
    )


def test_is_run_complete_requires_marker_model_manifest_and_history(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("run_complete.json", "model.keras", "manifest.json", "history.csv"):
        (run_dir / name).write_text("{}", encoding="utf-8")

    assert is_run_complete(run_dir) is True
    (run_dir / "history.csv").unlink()
    assert is_run_complete(run_dir) is False
