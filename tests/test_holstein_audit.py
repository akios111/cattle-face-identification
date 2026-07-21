from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cattle_id.holstein_audit import build_checkpoint_audit, build_control_deltas


SUFFIX = "fixture"


def _write_artifact(
    root: Path,
    name: str,
    *,
    predictions: list[str],
    correct: list[bool],
    probe_embeddings: np.ndarray,
) -> Path:
    run = root / name
    run.mkdir()
    (run / "model.keras").write_bytes(name.encode("utf-8"))
    frame = pd.DataFrame(
        {
            "probe_sample_id": ["probe-a", "probe-b"],
            "animal_id": ["cow-a", "cow-b"],
            "predicted_animal_id": predictions,
            "correct_rank_1": correct,
            "first_correct_rank": [1 if value else 2 for value in correct],
            "average_precision": [1.0 if value else 0.5 for value in correct],
        }
    )
    frame.to_csv(run / f"predictions_{SUFFIX}.csv", index=False)
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(run / f"gallery_embeddings_{SUFFIX}.npy", gallery)
    np.save(run / f"probe_embeddings_{SUFFIX}.npy", probe_embeddings)
    metrics = {
        "cmc_rank_1": float(np.mean(correct)),
        "cmc_rank_5": 1.0,
        "mean_average_precision": float(frame["average_precision"].mean()),
    }
    (run / f"metrics_{SUFFIX}.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


def test_checkpoint_audit_reports_hashes_overlap_rankings_and_cka(tmp_path: Path):
    first = _write_artifact(
        tmp_path,
        "run-a",
        predictions=["cow-a", "cow-b"],
        correct=[True, True],
        probe_embeddings=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    )
    second = _write_artifact(
        tmp_path,
        "run-b",
        predictions=["cow-a", "cow-a"],
        correct=[True, False],
        probe_embeddings=np.asarray([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
    )

    checkpoint_path, pairwise_path = build_checkpoint_audit(
        [first, second], tmp_path / "out", output_suffix=SUFFIX
    )

    checkpoints = pd.read_csv(checkpoint_path)
    pairwise = pd.read_csv(pairwise_path)
    assert len(checkpoints) == 2
    assert checkpoints["checkpoint_sha256"].nunique() == 2
    assert len(pairwise) == 1
    assert pairwise.loc[0, "correct_set_jaccard"] == 0.5
    assert 0.0 <= pairwise.loc[0, "probe_embedding_linear_cka"] <= 1.0


def test_control_deltas_are_paired_by_probe_and_bootstrapped_by_animal(tmp_path: Path):
    reference = _write_artifact(
        tmp_path,
        "imagenet",
        predictions=["cow-a", "cow-a"],
        correct=[True, False],
        probe_embeddings=np.asarray([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
    )
    candidate = _write_artifact(
        tmp_path,
        "fine-tuned",
        predictions=["cow-a", "cow-b"],
        correct=[True, True],
        probe_embeddings=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    )

    output = build_control_deltas(
        reference,
        [candidate],
        tmp_path / "deltas.csv",
        output_suffix=SUFFIX,
        bootstrap_resamples=50,
    )

    rows = pd.read_csv(output)
    assert set(rows["metric"]) == {"cmc_rank_1", "cmc_rank_5", "mean_average_precision"}
    assert rows.loc[rows["metric"] == "cmc_rank_1", "delta"].iloc[0] == 0.5
    assert set(rows["bootstrap_unit"]) == {"animal_id"}
