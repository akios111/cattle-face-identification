import numpy as np
import pytest
from sklearn.metrics import f1_score

from cattle_id.metrics import (
    aggregate_seed_metrics,
    bootstrap_accuracy_ci,
    bootstrap_cluster_accuracy_ci,
    bootstrap_seed_cluster_accuracy_ci,
    bootstrap_group_mean_ci,
    compute_classification_metrics,
    compute_retrieval_metrics,
    exact_mcnemar_test,
    holm_adjust,
    hierarchical_paired_metric_delta_ci,
    mcnemar_counts,
    paired_group_mean_delta_ci,
    paired_run_group_mean_delta_ci,
)


def test_compute_classification_metrics_for_multiclass_probabilities():
    y_true = np.array([0, 1, 2, 2])
    y_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    result = compute_classification_metrics(y_true, y_prob, labels=[0, 1, 2])

    assert result["accuracy"] == 0.75
    assert round(result["macro_precision"], 6) == 0.833333
    assert round(result["macro_recall"], 6) == 0.833333
    assert round(result["macro_f1"], 6) == 0.777778
    assert result["top_5_accuracy"] == 1.0
    assert result["confusion_matrix"].tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_bootstrap_accuracy_ci_is_stable_for_fixed_seed():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    result = bootstrap_accuracy_ci(y_true, y_pred, n_resamples=200, seed=7)

    assert result["metric"] == "accuracy"
    assert result["estimate"] == pytest.approx(4 / 6)
    assert result["ci_low"] == pytest.approx(1 / 3)
    assert result["ci_high"] == pytest.approx(1.0)


def test_mcnemar_counts_handles_identical_and_one_sided_disagreements():
    y_true = np.array([0, 1, 1, 0])
    model_a = np.array([0, 1, 0, 0])
    model_b = np.array([0, 0, 1, 0])

    counts = mcnemar_counts(y_true, model_a, model_b)

    assert counts == {
        "both_correct": 2,
        "model_a_only": 1,
        "model_b_only": 1,
        "both_wrong": 0,
        "statistic": 0.0,
    }


def test_mcnemar_counts_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_counts(np.array([0, 1]), np.array([0]), np.array([0, 1]))


def test_exact_mcnemar_handles_identical_and_one_sided_disagreements():
    identical = exact_mcnemar_test([0, 1], [0, 1], [0, 1])
    assert identical["discordant"] == 0
    assert identical["p_value_exact"] == 1.0

    one_sided = exact_mcnemar_test(
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [9, 9, 9, 9, 4, 5],
    )
    assert one_sided["model_a_only"] == 4
    assert one_sided["model_b_only"] == 0
    assert one_sided["p_value_exact"] == pytest.approx(0.125)


def test_holm_adjust_preserves_ordered_monotonic_adjustment():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])

    with pytest.raises(ValueError, match="between zero and one"):
        holm_adjust([1.1])


def test_aggregate_seed_metrics_returns_mean_and_std():
    rows = [
        {"model": "efficientnetv2b3", "protocol": "paper_random", "seed": 1, "accuracy": 0.9},
        {"model": "efficientnetv2b3", "protocol": "paper_random", "seed": 2, "accuracy": 1.0},
    ]

    summary = aggregate_seed_metrics(rows, metric="accuracy")

    assert summary == [
        {
            "model": "efficientnetv2b3",
            "protocol": "paper_random",
            "metric": "accuracy",
            "seeds": 2,
            "mean": 0.95,
            "std": 0.07071067811865474,
        }
    ]


def test_compute_retrieval_metrics_reports_cmc_and_map():
    gallery_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    gallery_ids = np.array(["cow_a", "cow_a", "cow_b"])
    probe_embeddings = np.array(
        [
            [0.95, 0.05],
            [0.1, 0.9],
            [0.8, 0.2],
        ]
    )
    probe_ids = np.array(["cow_a", "cow_b", "cow_b"])

    metrics, predictions = compute_retrieval_metrics(
        gallery_embeddings,
        gallery_ids,
        probe_embeddings,
        probe_ids,
    )

    assert metrics["cmc_rank_1"] == pytest.approx(2 / 3)
    assert metrics["cmc_rank_5"] == 1.0
    assert metrics["mean_average_precision"] == pytest.approx((1.0 + 1.0 + 1 / 3) / 3)
    assert predictions["correct_rank_1"].tolist() == [True, True, False]
    assert predictions["first_correct_rank"].tolist() == [1, 1, 3]


def test_compute_retrieval_metrics_rejects_probe_identity_absent_from_gallery():
    with pytest.raises(ValueError, match="absent from gallery"):
        compute_retrieval_metrics(
            np.array([[1.0, 0.0]]),
            np.array(["cow_a"]),
            np.array([[0.0, 1.0]]),
            np.array(["cow_b"]),
        )


def test_bootstrap_group_mean_ci_is_deterministic_and_resamples_animals():
    values = np.array([1.0, 1.0, 0.0, 0.0])
    groups = np.array(["cow_a", "cow_a", "cow_b", "cow_b"])

    first = bootstrap_group_mean_ci(values, groups, n_resamples=200, seed=9)
    second = bootstrap_group_mean_ci(values, groups, n_resamples=200, seed=9)

    assert first == second
    assert first["estimate"] == 0.5
    assert first["groups"] == 2
    assert first["ci_low"] == 0.0
    assert first["ci_high"] == 1.0


def test_bootstrap_cluster_accuracy_ci_resamples_complete_groups():
    result = bootstrap_cluster_accuracy_ci(
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        ["cow_a", "cow_a", "cow_b", "cow_b"],
        n_resamples=200,
        seed=9,
    )

    assert result["metric"] == "accuracy"
    assert result["bootstrap_unit"] == "group"
    assert result["estimate"] == 0.5
    assert result["groups"] == 2
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 1.0


def test_bootstrap_seed_cluster_accuracy_preserves_seed_specific_observations():
    result = bootstrap_seed_cluster_accuracy_ci(
        [
            ([0, 0, 1, 1], [0, 1, 1, 1], ["cow_a", "cow_a", "cow_b", "cow_b"]),
            ([0, 0, 1, 1], [0, 0, 1, 1], ["cow_a", "cow_a", "cow_b", "cow_b"]),
        ],
        n_resamples=200,
        seed=9,
    )

    assert result["estimate"] == 0.875
    assert result["groups"] == 2
    assert result["runs"] == 2
    assert result["run_aggregation"] == "equal_run_mean"
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_bootstrap_seed_cluster_accuracy_rejects_different_cluster_sets():
    with pytest.raises(ValueError, match="identical cluster IDs"):
        bootstrap_seed_cluster_accuracy_ci(
            [
                ([0], [0], ["cow_a"]),
                ([1], [1], ["cow_b"]),
            ],
            n_resamples=10,
        )


def test_paired_group_mean_delta_ci_is_deterministic():
    first = paired_group_mean_delta_ci(
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        ["cow_a", "cow_a", "cow_b", "cow_b"],
        n_resamples=100,
        seed=4,
    )
    second = paired_group_mean_delta_ci(
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        ["cow_a", "cow_a", "cow_b", "cow_b"],
        n_resamples=100,
        seed=4,
    )

    assert first == second


def test_paired_run_group_mean_delta_ci_averages_fixed_runs_deterministically():
    runs = [
        (
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            ["cow_a", "cow_a", "cow_b", "cow_b"],
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            ["cow_a", "cow_a", "cow_b", "cow_b"],
        ),
    ]

    first = paired_run_group_mean_delta_ci(runs, n_resamples=100, seed=4)
    second = paired_run_group_mean_delta_ci(runs, n_resamples=100, seed=4)

    assert first == second
    assert first["estimate"] == pytest.approx(0.25)
    assert first["runs"] == 2
    assert first["groups"] == 2
    assert first["run_aggregation"] == "equal_run_mean"


def test_paired_run_group_mean_delta_ci_rejects_mismatched_group_sets():
    with pytest.raises(ValueError, match="identical group IDs"):
        paired_run_group_mean_delta_ci(
            [
                ([0.0, 1.0], [1.0, 1.0], ["cow_a", "cow_b"]),
                ([0.0, 1.0], [1.0, 1.0], ["cow_a", "cow_c"]),
            ],
            n_resamples=10,
        )


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_hierarchical_paired_metric_delta_resamples_seeds_and_classes(metric):
    runs = [
        ([0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1], ["0", "0", "1", "1"]),
        ([0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1], ["0", "0", "1", "1"]),
    ]

    result = hierarchical_paired_metric_delta_ci(
        runs,
        metric=metric,
        n_resamples=100,
        seed=8,
    )

    assert result["estimate"] > 0
    assert result["runs"] == 2
    assert result["groups"] == 2
    assert result["bootstrap_unit"] == "training_seed_then_group"


def test_hierarchical_paired_metric_delta_rejects_unmatched_groups():
    with pytest.raises(ValueError, match="identical group IDs"):
        hierarchical_paired_metric_delta_ci(
            [
                ([0], [0], [0], ["cow_a"]),
                ([1], [1], [1], ["cow_b"]),
            ],
            n_resamples=10,
        )


def _naive_hierarchical_delta(runs, *, metric, n_resamples, seed):
    prepared = [
        tuple(np.asarray(values) for values in run)
        for run in runs
    ]
    groups = np.asarray(sorted(set(prepared[0][3].astype(str))))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        selected_runs = rng.integers(0, len(prepared), size=len(prepared))
        selected_groups = rng.choice(groups, size=len(groups), replace=True)
        deltas = []
        for selected_run in selected_runs:
            y_true, reference, candidate, run_groups = prepared[int(selected_run)]
            run_groups = run_groups.astype(str)
            indices = np.concatenate(
                [np.flatnonzero(run_groups == str(group)) for group in selected_groups]
            )
            sampled_true = y_true[indices].astype(int)
            sampled_reference = reference[indices].astype(int)
            sampled_candidate = candidate[indices].astype(int)
            if metric == "accuracy":
                reference_score = np.mean(sampled_reference == sampled_true)
                candidate_score = np.mean(sampled_candidate == sampled_true)
            else:
                labels = sorted(np.unique(sampled_true).tolist())
                reference_score = f1_score(
                    sampled_true,
                    sampled_reference,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
                candidate_score = f1_score(
                    sampled_true,
                    sampled_candidate,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            deltas.append(float(candidate_score - reference_score))
        estimates.append(float(np.mean(deltas)))
    return np.asarray(estimates)


@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_hierarchical_paired_metric_delta_matches_naive_bootstrap(metric):
    runs = [
        (
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 0, 2, 1],
            [0, 0, 1, 1, 2, 2],
            ["a", "a", "b", "b", "c", "c"],
        ),
        (
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 2, 0, 2],
            [0, 0, 1, 1, 2, 2],
            ["a", "a", "b", "b", "c", "c"],
        ),
    ]
    n_resamples = 100
    seed = 19
    expected = _naive_hierarchical_delta(
        runs,
        metric=metric,
        n_resamples=n_resamples,
        seed=seed,
    )

    result = hierarchical_paired_metric_delta_ci(
        runs,
        metric=metric,
        n_resamples=n_resamples,
        seed=seed,
    )

    assert result["ci_low"] == pytest.approx(np.quantile(expected, 0.025))
    assert result["ci_high"] == pytest.approx(np.quantile(expected, 0.975))
