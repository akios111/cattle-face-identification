from pathlib import Path

import yaml

from cattle_id.run_matrix import (
    expand_experiment_matrix,
    load_experiment_matrix,
    matrix_job_id,
    render_gpu_runbook,
    select_seed_shard,
    summarize_experiment_matrix,
)


def test_expand_experiment_matrix_builds_seed_model_protocol_cartesian_product(tmp_path):
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "configs/cattlessfr_colab_proplus.yaml",
                "seeds": [1, 2, 3, 4, 5],
                "models": ["efficientnetv2b3", "convnexttiny", "efficientnetb0"],
                "protocols": ["paper_random", "transform_holdout"],
                "evaluations": ["external_acquisition_holdout", "mask_ear_tag"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    matrix = load_experiment_matrix(config_path)
    jobs = expand_experiment_matrix(matrix)

    train_jobs = [job for job in jobs if job["kind"] == "train"]
    eval_jobs = [job for job in jobs if job["kind"] == "evaluate"]
    assert len(train_jobs) == 30
    assert len(eval_jobs) == 30
    assert train_jobs[0] == {
        "kind": "train",
        "base_config": "configs/cattlessfr_colab_proplus.yaml",
        "seed": 1,
        "model": "efficientnetv2b3",
        "protocol": "paper_random",
    }
    assert eval_jobs[-1]["evaluation"] == "mask_ear_tag"


def test_summarize_experiment_matrix_reports_full_100_point_matrix():
    matrix = load_experiment_matrix(Path("configs/experiment_matrix_100.yaml"))
    jobs = expand_experiment_matrix(matrix)

    summary = summarize_experiment_matrix(matrix, jobs)

    assert summary == {
        "base_config": "configs/cattlessfr_colab_proplus.yaml",
        "seed_count": 5,
        "model_count": 3,
        "protocol_count": 7,
        "evaluation_count": 4,
        "train_jobs": 105,
        "evaluation_jobs": 60,
        "total_jobs": 165,
        "external_metadata": "artifacts/metadata/external_acquisition_holdout.csv",
        "masking_annotations": "data/external_acquisition/masking_annotations.csv",
    }


def test_render_gpu_runbook_keeps_external_claims_gated():
    matrix = load_experiment_matrix(Path("configs/experiment_matrix_100.yaml"))
    jobs = expand_experiment_matrix(matrix)

    runbook = render_gpu_runbook(matrix, jobs)

    assert "A100/H100" in runbook
    assert "Train jobs: 105" in runbook
    assert "Evaluation jobs: 60" in runbook
    assert "Total jobs: 165" in runbook
    assert "external_acquisition_holdout" in runbook
    assert "data/external_acquisition/manifest.csv" in runbook
    assert "No 100/100 field-generalization claim is allowed" in runbook
    assert "python -m cattle_id.run_matrix --matrix configs/experiment_matrix_100.yaml" in runbook
    assert "notebooks/colab_train.ipynb" in runbook


def test_seed_sharding_keeps_complete_seed_groups_and_is_deterministic():
    matrix = load_experiment_matrix(Path("configs/experiment_matrix_100.yaml"))
    jobs = expand_experiment_matrix(matrix)

    shard = select_seed_shard(jobs, shard_index=2, shard_count=5)

    assert {job["seed"] for job in shard} == {3}
    assert len(shard) == 33
    assert shard == select_seed_shard(jobs, shard_index=2, shard_count=5)


def test_matrix_job_id_is_stable_for_train_and_evaluation_jobs():
    assert matrix_job_id(
        {
            "kind": "train",
            "seed": 1,
            "model": "efficientnetv2b3",
            "protocol": "paper_random",
        }
    ) == "train_efficientnetv2b3_paper_random_seed1"
    assert matrix_job_id(
        {
            "kind": "evaluate",
            "seed": 2,
            "model": "convnexttiny",
            "evaluation": "holstein2025_unseen_identity_reid",
        }
    ) == "evaluate_convnexttiny_holstein2025_unseen_identity_reid_seed2"


def test_open_set_matrix_expands_source_and_target_training_with_paired_evaluations():
    matrix = load_experiment_matrix(Path("configs/experiment_matrix_open_set.yaml"))
    jobs = expand_experiment_matrix(matrix)
    summary = summarize_experiment_matrix(matrix, jobs)

    assert summary["train_jobs"] == 120
    assert summary["evaluation_jobs"] == 120
    assert summary["total_jobs"] == 240
    assert summary["source_train_jobs"] == 105
    assert summary["target_train_jobs"] == 15
    source_evaluations = [
        job
        for job in jobs
        if job["kind"] == "evaluate" and job.get("training_scope") == "source"
    ]
    assert len(source_evaluations) == 105
    assert {job["source_protocol"] for job in source_evaluations} == set(matrix["protocols"])
    target_jobs = [job for job in jobs if job.get("training_scope") == "target"]
    assert len(target_jobs) == 30
    assert {job.get("metadata") for job in target_jobs} == {
        "artifacts/metadata/holstein2025_open_set.csv"
    }


def test_open_set_runbook_states_public_unseen_identity_boundary():
    matrix = load_experiment_matrix(Path("configs/experiment_matrix_open_set.yaml"))

    runbook = render_gpu_runbook(
        matrix,
        expand_experiment_matrix(matrix),
        matrix_path="configs/experiment_matrix_open_set.yaml",
    )

    assert "Holstein2025" in runbook
    assert "240" in runbook
    assert "same-identity external acquisition" in runbook
    assert "unseen-identity" in runbook
    assert "holstein2025_readiness.py" in runbook


def test_final_matrices_expand_to_the_approved_official_minimum_scope():
    baseline = expand_experiment_matrix(
        load_experiment_matrix(Path("configs/experiment_matrix_paper_baseline_final.yaml"))
    )
    primary = expand_experiment_matrix(
        load_experiment_matrix(Path("configs/experiment_matrix_primary_final.yaml"))
    )

    assert len(baseline) == 5
    assert all(job["kind"] == "train" for job in baseline)
    assert {job["model"] for job in baseline} == {
        "vgg16",
        "resnet50",
        "mobilenetv2",
        "densenet121",
        "efficientnetb0",
    }
    assert {job["seed"] for job in baseline} == {2026}

    primary_train = [job for job in primary if job["kind"] == "train"]
    primary_eval = [job for job in primary if job["kind"] == "evaluate"]
    assert len(primary_train) == 10
    assert len(primary_eval) == 10
    assert {job["model"] for job in primary} == {"efficientnetv2b3"}
    assert {job["seed"] for job in primary} == {1, 2, 3, 4, 5}
    assert {job.get("source_protocol") for job in primary_eval} == {
        "paper_random",
        "transform_holdout",
    }


def test_hardening_matrices_expand_to_exact_training_and_holstein_contracts():
    training = expand_experiment_matrix(
        load_experiment_matrix(Path("configs/experiment_matrix_hardening_v2.yaml"))
    )
    holstein = expand_experiment_matrix(
        load_experiment_matrix(Path("configs/experiment_matrix_holstein_hardening_v2.yaml"))
    )

    assert len(training) == 29
    assert len({matrix_job_id(job) for job in training}) == 29
    assert all(job["kind"] == "train" for job in training)
    assert len([job for job in training if job["group"] == "training_seed_variability"]) == 10
    assert len([job for job in training if job["group"] == "split_sensitivity"]) == 4
    assert len([job for job in training if job["group"] == "controlled_ablation"]) == 15

    assert len(holstein) == 16
    assert len({matrix_job_id(job) for job in holstein}) == 16
    assert [job["control_type"] for job in holstein].count("imagenet_only") == 1
    assert [job["control_type"] for job in holstein].count("frozen") == 5
    assert [job["control_type"] for job in holstein].count("fine_tuned") == 10


def test_hardening_job_id_encodes_all_independent_seeds():
    assert matrix_job_id(
        {
            "kind": "train",
            "model": "efficientnetv2b3",
            "protocol": "paper_random_hardening_v2",
            "training_seed": 1,
            "split_seed": 5,
            "augmentation_seed": 1,
        }
    ) == "train_efficientnetv2b3_paper_random_hardening_v2_train1_split5_aug1"
