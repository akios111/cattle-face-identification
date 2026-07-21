from pathlib import Path

from cattle_id.hardening_matrix import (
    commands_for_job,
    execute_hardening_matrix,
    metadata_path_for_job,
    metadata_protocol_for_job,
    run_dir_for_job,
)
from cattle_id.run_matrix import expand_experiment_matrix, load_experiment_matrix


def test_hardening_matrix_dry_run_has_29_train_eval_pairs_and_six_metadata_sets():
    result = execute_hardening_matrix(
        "configs/experiment_matrix_hardening_v2.yaml",
        dry_run=True,
    )

    assert result["jobs"] == 29
    assert result["metadata_sets"] == 6
    commands = result["planned_commands"]
    assert len([command for command in commands if "cattle_id.train" in command]) == 29
    assert len([command for command in commands if "cattle_id.evaluate" in command]) == 29
    assert len([command for command in commands if "cattle_id.prepare" in command]) == 6


def test_ablation_jobs_reuse_fixed_paper_random_metadata_but_keep_distinct_run_ids():
    jobs = expand_experiment_matrix(
        load_experiment_matrix(Path("configs/experiment_matrix_hardening_v2.yaml"))
    )
    ablation = next(job for job in jobs if job["protocol"] == "ablation_no_cutout_hardening_v2")
    base = next(
        job
        for job in jobs
        if job["protocol"] == "paper_random_hardening_v2"
        and job["training_seed"] == ablation["training_seed"]
        and job["split_seed"] == 1
    )

    assert metadata_protocol_for_job(ablation) == "paper_random_hardening_v2"
    assert metadata_path_for_job(ablation) == metadata_path_for_job(base)
    assert run_dir_for_job(ablation) != run_dir_for_job(base)


def test_job_commands_pass_all_three_seeds_and_shared_metadata_override():
    job = {
        "kind": "train",
        "base_config": "configs/cattlessfr_hardening_v2_colab_proplus.yaml",
        "model": "efficientnetv2b3",
        "protocol": "ablation_224_hardening_v2",
        "training_seed": 3,
        "split_seed": 1,
        "augmentation_seed": 1,
    }

    prepare, train, evaluate = commands_for_job(job)

    assert prepare[prepare.index("--protocol") + 1] == "paper_random_hardening_v2"
    assert train[train.index("--training-seed") + 1] == "3"
    assert train[train.index("--split-seed") + 1] == "1"
    assert train[train.index("--augmentation-seed") + 1] == "1"
    assert any(
        value.endswith("metadata_paper_random_hardening_v2_split1_aug1.csv")
        for value in train
    )
    assert evaluate[-2:] == ["--split", "test"]
