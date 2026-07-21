from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "hardening_evidence_summary.py"
SPEC = importlib.util.spec_from_file_location("hardening_evidence_summary", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _predictions(seed: int, *, candidate: bool = False) -> pd.DataFrame:
    predicted = [0, 0, 1, 1] if candidate else [0, seed % 2, 1, 0]
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "image_sha256": ["1" * 64, "2" * 64, "3" * 64, "4" * 64],
            "class_id": [0, 0, 1, 1],
            "predicted_class_id": predicted,
        }
    )


def test_training_stability_requires_and_reports_identical_test_samples():
    artifacts = []
    for protocol in ("paper_random_hardening_v2", "transform_holdout_hardening_v2"):
        for seed in range(1, 6):
            frame = _predictions(seed)
            accuracy = float((frame["class_id"] == frame["predicted_class_id"]).mean())
            artifacts.append(
                {
                    "job": {"protocol": protocol, "split_seed": 1, "training_seed": seed},
                    "metrics": {
                        "accuracy": accuracy,
                        "macro_f1": accuracy,
                        "test_set_sha256": "a" * 64,
                    },
                    "predictions": frame,
                }
            )

    summary = MODULE.summarize_training_stability(artifacts, bootstrap_resamples=25)

    assert len(summary) == 2
    assert set(summary["training_seeds"]) == {5}
    assert set(summary["test_sample_alignment"]) == {"identical"}
    assert set(summary["bootstrap_resamples"]) == {25}


def test_controlled_ablation_summary_is_paired_by_seed_sample_and_bytes():
    artifacts = []
    for seed in range(1, 6):
        base = _predictions(seed)
        base_accuracy = float((base["class_id"] == base["predicted_class_id"]).mean())
        artifacts.append(
            {
                "job": {
                    "protocol": "paper_random_hardening_v2",
                    "split_seed": 1,
                    "training_seed": seed,
                },
                "metrics": {"test_set_sha256": "a" * 64, "accuracy": base_accuracy},
                "predictions": base,
            }
        )
        for protocol in (
            "ablation_no_cutout_hardening_v2",
            "ablation_224_hardening_v2",
            "ablation_frozen_hardening_v2",
        ):
            candidate = _predictions(seed, candidate=True)
            artifacts.append(
                {
                    "job": {"protocol": protocol, "split_seed": 1, "training_seed": seed},
                    "metrics": {"test_set_sha256": "a" * 64, "accuracy": 1.0},
                    "predictions": candidate,
                }
            )

    runs, summary = MODULE.summarize_controlled_ablations(
        artifacts, bootstrap_resamples=25
    )

    assert len(runs) == 15
    assert len(summary) == 3
    assert set(summary["paired_seeds"]) == {5}
    assert (summary["delta_accuracy"] >= 0).all()


def test_shortcut_mcnemar_applies_holm_to_all_methods():
    cnn = _predictions(1, candidate=True)
    shortcuts = {f"method-{index}": _predictions(index) for index in range(1, 6)}

    result = MODULE.shortcut_mcnemar(shortcuts, cnn)

    assert len(result) == 5
    assert result["p_value_holm"].between(0, 1).all()


def test_experiment_settings_table_contains_all_twenty_augmentations():
    table = MODULE.experiment_settings_table(
        "configs/cattlessfr_hardening_v2_colab_proplus.yaml"
    )

    augmentations = table[table["section"] == "augmentation"]
    assert set([spec for spec in augmentations["name"] if spec == "original"]) == {"original"}
    assert len(augmentations[augmentations["name"].isin([
        "original", "flip_horizontal", "flip_vertical", "rotate_neg_15", "rotate_pos_15",
        "brightness_up", "brightness_down", "contrast_up", "contrast_down", "gaussian_noise",
        "blur", "sharpen", "translate_left", "translate_right", "zoom_in", "zoom_out",
        "shear_x", "shear_y", "cutout_center", "cutout_random",
    ])]) == 20


def test_legacy_robustness_prediction_columns_are_restored():
    region = pd.DataFrame(
        {
            "augmentation_id": ["grayscale", "central_region_only"],
            "class_id": [0, 1],
            "predicted_class_id": [0, 1],
        }
    )
    severity = pd.DataFrame(
        {
            "augmentation_id": ["gaussian_noise_pos_6", "cutout_center_neg_0.125"],
            "class_id": [0, 1],
            "predicted_class_id": [0, 1],
        }
    )

    restored_region = MODULE._restore_robustness_columns(region, "image_region_audit")
    restored_severity = MODULE._restore_robustness_columns(severity, "severity_sweep")

    assert restored_region["region_variant"].tolist() == ["grayscale", "central_region_only"]
    assert restored_severity["severity_family"].tolist() == ["gaussian_noise", "cutout_center"]
    assert restored_severity["severity_value"].tolist() == [6.0, 0.125]
    assert restored_severity["severity_direction"].tolist() == [1, -1]


def _write_evidence_run(root: Path, job: dict[str, object], *, model_sha256: str) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    common = {
        "model": job["model"],
        "protocol": job["protocol"],
        "training_seed": job["training_seed"],
        "split_seed": job["split_seed"],
        "augmentation_seed": job["augmentation_seed"],
        "protocol_version": "hardening_v2",
    }
    (run_dir / "run_complete.json").write_text(json.dumps(common), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(common), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                **common,
                "model_sha256": model_sha256,
                "test_set_sha256": "a" * 64,
                "samples": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.csv").write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "class_id": [0, 1],
            "predicted_class_id": [0, 1],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    return run_dir


def test_verified_evidence_run_does_not_require_local_checkpoint(tmp_path: Path):
    model_sha256 = "b" * 64
    job = {
        "model": "efficientnetv2b3",
        "protocol": "paper_random_hardening_v2",
        "training_seed": 1,
        "split_seed": 1,
        "augmentation_seed": 1,
    }
    run_dir = _write_evidence_run(tmp_path, job, model_sha256=model_sha256)

    metrics, predictions = MODULE.validate_classification_evidence(
        run_dir,
        job,
        {run_dir.name: model_sha256},
    )

    assert metrics["model_sha256"] == model_sha256
    assert len(predictions) == 2
    assert not (run_dir / "model.keras").exists()


def test_verified_evidence_run_rejects_model_hash_mismatch(tmp_path: Path):
    job = {
        "model": "efficientnetv2b3",
        "protocol": "paper_random_hardening_v2",
        "training_seed": 1,
        "split_seed": 1,
        "augmentation_seed": 1,
    }
    run_dir = _write_evidence_run(tmp_path, job, model_sha256="b" * 64)

    with pytest.raises(ValueError, match="model hash mismatch"):
        MODULE.validate_classification_evidence(
            run_dir,
            job,
            {run_dir.name: "c" * 64},
        )


def _holstein_predictions(*, candidate: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "probe_sample_id": ["probe-a1", "probe-a2", "probe-b1", "probe-b2"],
            "animal_id": ["cow-a", "cow-a", "cow-b", "cow-b"],
            "correct_rank_1": [candidate, True, False, True],
            "first_correct_rank": [1 if candidate else 6, 1, 6, 1],
            "average_precision": [1.0 if candidate else 0.2, 1.0, 0.1, 1.0],
        }
    )


def test_holstein_group_control_deltas_pair_animals_across_five_checkpoints(tmp_path: Path):
    runs = tmp_path / "runs"
    reference = runs / "imagenet_only_efficientnetv2b3_hardening_v2"
    reference.mkdir(parents=True)
    _holstein_predictions(candidate=False).to_csv(
        reference / f"predictions_{MODULE.HOLSTEIN_SUFFIX}.csv",
        index=False,
    )
    artifacts = []
    for protocol in (
        "ablation_frozen_hardening_v2",
        "paper_random_hardening_v2",
        "transform_holdout_hardening_v2",
    ):
        for seed in range(1, 6):
            run_dir = runs / f"{protocol}-{seed}"
            run_dir.mkdir()
            _holstein_predictions(candidate=True).to_csv(
                run_dir / f"predictions_{MODULE.HOLSTEIN_SUFFIX}.csv",
                index=False,
            )
            artifacts.append(
                {
                    "job": {"protocol": protocol, "training_seed": seed},
                    "run_dir": run_dir,
                }
            )

    result = MODULE.summarize_holstein_group_control_deltas(
        artifacts,
        runs_root=runs,
        bootstrap_resamples=50,
        bootstrap_seed=4,
    )

    assert len(result) == 9
    assert set(result["checkpoints"]) == {5}
    assert set(result["bootstrap_unit"]) == {"animal_id"}
    assert set(result["run_aggregation"]) == {"equal_run_mean"}
    deltas = result.set_index(["candidate_group", "metric"])["delta"]
    assert deltas.loc[("frozen", "cmc_rank_1")] == 0.25
    assert deltas.loc[("frozen", "cmc_rank_5")] == 0.25
    assert deltas.loc[("frozen", "mean_average_precision")] == pytest.approx(0.2)


def test_identity_balanced_holstein_gives_each_animal_equal_weight():
    rows = []
    for probe in range(100):
        rows.append(
            {
                "probe_sample_id": f"heavy-{probe}",
                "animal_id": "cow-00",
                "correct_rank_1": False,
                "first_correct_rank": 6,
                "average_precision": 0.0,
            }
        )
    for animal in range(1, 20):
        rows.append(
            {
                "probe_sample_id": f"cow-{animal:02d}",
                "animal_id": f"cow-{animal:02d}",
                "correct_rank_1": True,
                "first_correct_rank": 1,
                "average_precision": 1.0,
            }
        )

    result = MODULE.summarize_identity_balanced_holstein(pd.DataFrame(rows))

    assert result["identity_count"] == 20
    assert result["identity_balanced_cmc_rank_1"] == pytest.approx(0.95)
    assert result["identity_balanced_cmc_rank_5"] == pytest.approx(0.95)
    assert result["identity_balanced_mean_average_precision"] == pytest.approx(0.95)
