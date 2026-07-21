from pathlib import Path

from cattle_id.evidence_scope import (
    annotate_rows,
    filter_predictions,
    group_for_row,
    load_evidence_scope,
)


SCOPE_PATH = Path("configs/final_evidence_scope.yaml")


def test_final_scope_classifies_confirmatory_and_exploratory_rows():
    scope = load_evidence_scope(SCOPE_PATH)

    primary = group_for_row(
        {
            "model": "efficientnetv2b3",
            "protocol": "paper_random",
            "seed": 3,
            "accuracy": 0.99,
        },
        scope,
    )
    ablation = group_for_row(
        {
            "model": "efficientnetv2b3",
            "protocol": "ablation_no_cutout",
            "seed": 1,
            "accuracy": 0.98,
        },
        scope,
    )

    assert primary["name"] == "primary_confirmatory"
    assert primary["role"] == "confirmatory"
    assert ablation["name"] == "primary_ablation"
    assert ablation["role"] == "exploratory"


def test_final_scope_excludes_full_matrix_overplay_rows():
    scope = load_evidence_scope(SCOPE_PATH)

    excluded = group_for_row(
        {
            "model": "convnexttiny",
            "protocol": "ablation_frozen",
            "seed": 4,
            "accuracy": 0.99,
        },
        scope,
    )
    masked = group_for_row(
        {
            "model": "efficientnetv2b3",
            "protocol": "mask_background",
            "seed": 1,
            "accuracy": 0.50,
        },
        scope,
    )

    assert excluded is None
    assert masked is None


def test_scope_filters_predictions_by_scoped_metric_id():
    scope = load_evidence_scope(SCOPE_PATH)
    metrics = annotate_rows(
        [
            {
                "run_id": "kept",
                "model": "efficientnetv2b3",
                "protocol": "paper_random",
                "seed": 1,
                "accuracy": 1.0,
            },
            {
                "run_id": "dropped",
                "model": "convnexttiny",
                "protocol": "ablation_224",
                "seed": 5,
                "accuracy": 1.0,
            },
        ],
        scope,
    )

    predictions = filter_predictions({"kept": [{"class_id": "0"}], "dropped": []}, metrics)

    assert list(predictions) == ["kept"]


def test_hardening_scope_distinguishes_legacy_fixed_split_and_split_sensitivity():
    scope = load_evidence_scope("configs/final_evidence_scope_hardening_v2.yaml")
    legacy = group_for_row(
        {
            "model": "efficientnetb0",
            "protocol": "paper_random",
            "seed": 2026,
            "protocol_version": "legacy",
        },
        scope,
    )
    fixed = group_for_row(
        {
            "model": "efficientnetv2b3",
            "protocol": "paper_random_hardening_v2",
            "seed": 3,
            "training_seed": 3,
            "split_seed": 1,
            "augmentation_seed": 1,
            "protocol_version": "hardening_v2",
        },
        scope,
    )
    sensitivity = group_for_row(
        {
            "model": "efficientnetv2b3",
            "protocol": "paper_random_hardening_v2",
            "seed": 1,
            "training_seed": 1,
            "split_seed": 4,
            "augmentation_seed": 1,
            "protocol_version": "hardening_v2",
        },
        scope,
    )

    assert legacy["role"] == "legacy_replication"
    assert fixed["name"] == "hardening_primary_fixed_split"
    assert fixed["role"] == "confirmatory_hardening_v2"
    assert sensitivity["name"] == "hardening_split_sensitivity"
