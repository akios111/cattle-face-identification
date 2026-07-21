from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cattle_id.shortcut_audit import (
    deterministic_one_nearest_neighbor,
    extract_handcrafted_features,
    run_shortcut_method,
)


def test_one_nearest_neighbor_uses_lexicographic_sample_id_for_exact_ties():
    predicted, distances, neighbours = deterministic_one_nearest_neighbor(
        np.asarray([[0.0], [0.0]], dtype=np.float32),
        np.asarray([2, 1]),
        ["z-sample", "a-sample"],
        np.asarray([[0.0]], dtype=np.float32),
        metric="euclidean",
    )

    assert predicted.tolist() == [1]
    assert distances.tolist() == [0.0]
    assert neighbours == ["a-sample"]


def test_handcrafted_feature_extractors_return_one_row_per_image(tmp_path: Path):
    rows = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        rows.append({"image_path": str(path)})
    metadata = pd.DataFrame(rows)

    for method in ("phash", "pixels32", "hsv_histogram", "hog"):
        features = extract_handcrafted_features(metadata, method)
        assert features.shape[0] == 2
        assert features.ndim == 2


def test_shortcut_metrics_and_predictions_are_aligned_and_cluster_bootstrapped():
    metadata = pd.DataFrame(
        [
            {"sample_id": "train-a", "image_path": "a", "class_id": 0, "split": "train"},
            {"sample_id": "train-b", "image_path": "b", "class_id": 1, "split": "train"},
            {"sample_id": "test-a", "image_path": "c", "class_id": 0, "split": "test"},
            {"sample_id": "test-b", "image_path": "d", "class_id": 1, "split": "test"},
        ]
    )
    feature_map = {
        "a": [0.0, 0.0],
        "b": [1.0, 1.0],
        "c": [0.1, 0.0],
        "d": [0.9, 1.0],
    }

    metrics, predictions = run_shortcut_method(
        metadata,
        method="pixels32",
        bootstrap_resamples=50,
        feature_extractor=lambda frame: np.asarray(
            [feature_map[path] for path in frame["image_path"]], dtype=np.float32
        ),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["raw_errors"] == 0
    assert metrics["bootstrap_unit"] == "class_id"
    assert predictions["predicted_class_id"].tolist() == [0, 1]
