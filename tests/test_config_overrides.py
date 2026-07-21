from __future__ import annotations

from cattle_id.config import config_for_run, metadata_filename, run_seeds_from_config


def _base_config() -> dict:
    return {
        "preprocessing": {"image_size": [384, 384]},
        "augmentation": {"profile": "all"},
        "split": {"protocol": "paper_random"},
        "training": {"seed": 2026, "fine_tune": {"enabled": True}},
    }


def test_config_for_run_applies_seed_without_mutating_base_config():
    base = _base_config()

    updated = config_for_run(base, protocol="paper_random", seed=3)

    assert updated["training"]["seed"] == 3
    assert base["training"]["seed"] == 2026


def test_config_for_run_maps_ablation_protocols_to_executable_settings():
    assert config_for_run(_base_config(), protocol="augmentation_ablation")["split"]["protocol"] == "augmentation_ablation"
    assert config_for_run(_base_config(), protocol="ablation_no_aug")["augmentation"]["profile"] == "none"
    assert config_for_run(_base_config(), protocol="ablation_geometric_only")["augmentation"]["profile"] == "geometric"
    assert config_for_run(_base_config(), protocol="ablation_no_cutout")["augmentation"]["profile"] == "no_cutout"
    assert config_for_run(_base_config(), protocol="ablation_cutout_only")["augmentation"]["profile"] == "cutout_only"
    assert config_for_run(_base_config(), protocol="ablation_224")["preprocessing"]["image_size"] == [224, 224]

    frozen = config_for_run(_base_config(), protocol="ablation_frozen")
    assert frozen["training"]["fine_tune"]["enabled"] is False
    assert frozen["split"]["protocol"] == "paper_random"

    holstein = config_for_run(_base_config(), protocol="holstein2025_closed_set")
    assert holstein["split"]["protocol"] == "holstein2025_closed_set"


def test_metadata_filename_keeps_legacy_name_without_seed_and_adds_seed_suffix_when_requested():
    assert metadata_filename("paper_random") == "metadata_paper_random.csv"
    assert metadata_filename("paper_random", seed=4) == "metadata_paper_random_seed4.csv"
    assert metadata_filename("ablation_no_aug", seed=1) == "metadata_ablation_no_aug_seed1.csv"


def test_hardening_config_separates_training_split_and_augmentation_seeds():
    updated = config_for_run(
        _base_config(),
        protocol="paper_random_hardening_v2",
        training_seed=5,
        split_seed=2,
        augmentation_seed=1,
    )

    assert updated["protocol_version"] == "hardening_v2"
    assert updated["split"]["protocol"] == "paper_random"
    assert run_seeds_from_config(updated) == {
        "training_seed": 5,
        "split_seed": 2,
        "augmentation_seed": 1,
    }
    assert metadata_filename(
        "paper_random_hardening_v2", split_seed=2, augmentation_seed=1
    ) == "metadata_paper_random_hardening_v2_split2_aug1.csv"


def test_hardening_no_cutout_filters_training_only_instead_of_changing_metadata_profile():
    updated = config_for_run(
        _base_config(),
        protocol="ablation_no_cutout_hardening_v2",
        training_seed=1,
        split_seed=1,
        augmentation_seed=1,
    )

    assert updated["augmentation"]["profile"] == "all"
    assert updated["training"]["exclude_augmentation_families"] == ["cutout"]
