from __future__ import annotations

from pathlib import Path

import numpy as np

from cattle_id.open_set_evaluate import is_open_set_evaluation_complete, open_set_output_paths


def test_open_set_output_paths_are_suffix_scoped(tmp_path: Path):
    paths = open_set_output_paths(tmp_path, "holstein2025_zero_shot")

    assert paths.metrics == tmp_path / "metrics_holstein2025_zero_shot.json"
    assert paths.predictions == tmp_path / "predictions_holstein2025_zero_shot.csv"
    assert paths.gallery_embeddings == tmp_path / "gallery_embeddings_holstein2025_zero_shot.npy"
    assert paths.probe_embeddings == tmp_path / "probe_embeddings_holstein2025_zero_shot.npy"


def test_open_set_output_paths_reject_empty_suffix(tmp_path: Path):
    try:
        open_set_output_paths(tmp_path, "")
    except ValueError as exc:
        assert "suffix" in str(exc)
    else:
        raise AssertionError("Expected an empty suffix to be rejected")


def test_open_set_evaluation_complete_requires_all_outputs(tmp_path: Path):
    paths = open_set_output_paths(tmp_path, "holstein")
    for path in (
        paths.metrics,
        paths.predictions,
        paths.gallery_embeddings,
        paths.probe_embeddings,
    ):
        if path.suffix == ".npy":
            np.save(path, np.array([[1.0]]))
        else:
            path.write_text("{}", encoding="utf-8")

    assert is_open_set_evaluation_complete(paths) is True
    paths.predictions.unlink()
    assert is_open_set_evaluation_complete(paths) is False
