from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "frozen_imagenet_sanity.py"
SPEC = importlib.util.spec_from_file_location("frozen_imagenet_sanity", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_embedding_difference_reports_storage_and_cosine_effects():
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.asarray([[1.0, 0.0], [0.01, 1.0]], dtype=np.float16)

    result = MODULE.embedding_difference(reference, candidate)

    assert result["embedding_samples"] == 2
    assert result["embedding_dimensions"] == 2
    assert result["reference_embedding_dtype"] == "float32"
    assert result["candidate_embedding_dtype"] == "float16"
    assert result["embedding_max_abs_difference"] == pytest.approx(0.0100021362)
    assert result["embedding_mean_cosine_distance"] > 0.0


def test_embedding_difference_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        MODULE.embedding_difference(np.ones((2, 3)), np.ones((3, 3)))


def test_backbone_state_hash_and_batchnorm_comparison_is_fail_closed():
    reference = {
        "conv/kernel": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "bn/moving_mean": np.asarray([0.0, 1.0], dtype=np.float32),
        "bn/moving_variance": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    identical = {key: value.copy() for key, value in reference.items()}

    result = MODULE.compare_backbone_states(reference, identical)

    assert result["backbone_hash_equal"] is True
    assert result["batchnorm_hash_equal"] is True
    assert result["backbone_max_abs_difference"] == 0.0
    assert result["batchnorm_max_abs_difference"] == 0.0

    changed = {key: value.copy() for key, value in reference.items()}
    changed["bn/moving_mean"][0] = 0.5
    changed_result = MODULE.compare_backbone_states(reference, changed)
    assert changed_result["backbone_hash_equal"] is False
    assert changed_result["batchnorm_hash_equal"] is False


def test_thesis_table_rejects_incomplete_state_audit():
    frame = pd.DataFrame(
        [
            {
                "training_seed": seed,
                "embedding_max_abs_difference": 0.01,
                "embedding_mean_cosine_distance": 1e-5,
                "backbone_hash_equal": True,
                "batchnorm_hash_equal": True,
                "mean_average_precision_delta": 0.0,
                "state_audit_complete": seed != 5,
                "backbone_tensor_count": 536,
                "batchnorm_moving_tensor_count": 176,
                "reference_embedding_dtype": "float32",
                "candidate_embedding_dtype": "float16",
                "reference_inference_batch_size": 64,
                "candidate_inference_batch_size": 128,
                "candidate_runtime_policy": "mixed_float16",
            }
            for seed in range(1, 6)
        ]
    )

    with pytest.raises(ValueError, match="five complete state audits"):
        MODULE.thesis_table(frame)
