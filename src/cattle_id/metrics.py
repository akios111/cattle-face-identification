from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    effective_k = min(k, y_prob.shape[1])
    top_k = np.argsort(y_prob, axis=1)[:, -effective_k:]
    hits = [int(label in row) for label, row in zip(y_true, top_k, strict=True)]
    return float(np.mean(hits))


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    labels: Sequence[int] | None = None,
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(labels) if labels is not None else sorted(np.unique(y_true).tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top_5_accuracy": _top_k_accuracy(y_true, y_prob, k=5),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def bootstrap_accuracy_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    n_resamples: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("bootstrap requires at least one sample")

    rng = np.random.default_rng(seed)
    scores = []
    indices = np.arange(len(y_true))
    for _ in range(n_resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        scores.append(float(np.mean(y_true[sample] == y_pred[sample])))
    alpha = 1.0 - confidence
    return {
        "metric": "accuracy",
        "estimate": float(np.mean(y_true == y_pred)),
        "ci_low": float(np.quantile(scores, alpha / 2.0)),
        "ci_high": float(np.quantile(scores, 1.0 - alpha / 2.0)),
    }


def mcnemar_counts(
    y_true: Sequence[int],
    model_a_pred: Sequence[int],
    model_b_pred: Sequence[int],
) -> dict[str, int | float]:
    y_true = np.asarray(y_true)
    model_a_pred = np.asarray(model_a_pred)
    model_b_pred = np.asarray(model_b_pred)
    if not (len(y_true) == len(model_a_pred) == len(model_b_pred)):
        raise ValueError("y_true, model_a_pred and model_b_pred must have the same length")

    a_correct = model_a_pred == y_true
    b_correct = model_b_pred == y_true
    model_a_only = int(np.sum(a_correct & ~b_correct))
    model_b_only = int(np.sum(~a_correct & b_correct))
    denominator = model_a_only + model_b_only
    statistic = 0.0
    if denominator:
        statistic = float((max(0, abs(model_a_only - model_b_only) - 1) ** 2) / denominator)
    return {
        "both_correct": int(np.sum(a_correct & b_correct)),
        "model_a_only": model_a_only,
        "model_b_only": model_b_only,
        "both_wrong": int(np.sum(~a_correct & ~b_correct)),
        "statistic": statistic,
    }


def exact_mcnemar_test(
    y_true: Sequence[int],
    model_a_pred: Sequence[int],
    model_b_pred: Sequence[int],
) -> dict[str, int | float]:
    counts = mcnemar_counts(y_true, model_a_pred, model_b_pred)
    a_only = int(counts["model_a_only"])
    b_only = int(counts["model_b_only"])
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, index) for index in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        **counts,
        "discordant": discordant,
        "p_value_exact": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be between zero and one")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running_max = 0.0
    total = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def aggregate_seed_metrics(rows: Sequence[dict[str, object]], *, metric: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["protocol"]))
        grouped.setdefault(key, []).append(float(row[metric]))

    summaries = []
    for (model, protocol), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        summaries.append(
            {
                "model": model,
                "protocol": protocol,
                "metric": metric,
                "seeds": int(len(array)),
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            }
        )
    return summaries


def _normalized_embeddings(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero-length embedding")
    return array / norms


def compute_retrieval_metrics(
    gallery_embeddings: np.ndarray,
    gallery_ids: Sequence[object],
    probe_embeddings: np.ndarray,
    probe_ids: Sequence[object],
) -> tuple[dict[str, object], pd.DataFrame]:
    gallery = _normalized_embeddings(gallery_embeddings, name="gallery_embeddings")
    probe = _normalized_embeddings(probe_embeddings, name="probe_embeddings")
    gallery_ids = np.asarray(gallery_ids).astype(str)
    probe_ids = np.asarray(probe_ids).astype(str)
    if len(gallery_ids) != len(gallery):
        raise ValueError("gallery ids and embeddings must have the same length")
    if len(probe_ids) != len(probe):
        raise ValueError("probe ids and embeddings must have the same length")
    if gallery.shape[1] != probe.shape[1]:
        raise ValueError("gallery and probe embedding dimensions must match")
    missing_ids = sorted(set(probe_ids) - set(gallery_ids))
    if missing_ids:
        raise ValueError(f"probe identities absent from gallery: {missing_ids}")

    similarities = probe @ gallery.T
    rankings = np.argsort(-similarities, axis=1, kind="mergesort")
    rows: list[dict[str, object]] = []
    average_precisions: list[float] = []
    first_correct_ranks: list[int] = []
    for index, ranked_indices in enumerate(rankings):
        ranked_ids = gallery_ids[ranked_indices]
        relevant = ranked_ids == probe_ids[index]
        relevant_ranks = np.flatnonzero(relevant) + 1
        first_correct_rank = int(relevant_ranks[0])
        precision_at_relevant = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
        average_precision = float(np.mean(precision_at_relevant))
        predicted_id = str(ranked_ids[0])
        rows.append(
            {
                "animal_id": str(probe_ids[index]),
                "predicted_animal_id": predicted_id,
                "correct_rank_1": predicted_id == str(probe_ids[index]),
                "first_correct_rank": first_correct_rank,
                "average_precision": average_precision,
                "top_1_similarity": float(similarities[index, ranked_indices[0]]),
            }
        )
        average_precisions.append(average_precision)
        first_correct_ranks.append(first_correct_rank)

    ranks = np.asarray(first_correct_ranks)
    metrics = {
        "probe_images": int(len(probe)),
        "probe_identities": int(len(set(probe_ids))),
        "gallery_images": int(len(gallery)),
        "gallery_identities": int(len(set(gallery_ids))),
        "cmc_rank_1": float(np.mean(ranks <= 1)),
        "cmc_rank_5": float(np.mean(ranks <= min(5, len(gallery)))),
        "mean_average_precision": float(np.mean(average_precisions)),
    }
    return metrics, pd.DataFrame(rows)


def bootstrap_group_mean_ci(
    values: Sequence[float],
    groups: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length")
    if len(values) == 0:
        raise ValueError("group bootstrap requires at least one sample")
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        raise ValueError("group bootstrap requires at least one group")
    grouped_values = {group: values[groups == group] for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample = np.concatenate([grouped_values[group] for group in selected])
        estimates.append(float(np.mean(sample)))
    alpha = 1.0 - confidence
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "groups": int(len(unique_groups)),
    }


def bootstrap_cluster_accuracy_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred and groups must have the same length")
    result = bootstrap_group_mean_ci(
        (y_true == y_pred).astype(float),
        groups,
        n_resamples=n_resamples,
        seed=seed,
        confidence=confidence,
    )
    return {"metric": "accuracy", "bootstrap_unit": "group", **result}


def bootstrap_seed_cluster_accuracy_ci(
    runs: Sequence[tuple[Sequence[int], Sequence[int], Sequence[object]]],
    *,
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap shared clusters while preserving each seed's test observations."""
    if not runs:
        raise ValueError("seed-cluster bootstrap requires at least one run")

    grouped_correctness: list[dict[str, np.ndarray]] = []
    run_estimates: list[float] = []
    reference_groups: set[str] | None = None
    for run_index, (y_true, y_pred, groups) in enumerate(runs):
        true_array = np.asarray(y_true)
        pred_array = np.asarray(y_pred)
        group_array = np.asarray(groups).astype(str)
        if not (len(true_array) == len(pred_array) == len(group_array)):
            raise ValueError(
                f"run {run_index} y_true, y_pred and groups must have the same length"
            )
        if len(true_array) == 0:
            raise ValueError(f"run {run_index} has no predictions")

        unique_groups = set(np.unique(group_array).tolist())
        if reference_groups is None:
            reference_groups = unique_groups
        elif unique_groups != reference_groups:
            raise ValueError(
                "seed-cluster bootstrap requires identical cluster IDs across runs"
            )

        correctness = (true_array == pred_array).astype(float)
        grouped_correctness.append(
            {
                group: correctness[group_array == group]
                for group in sorted(unique_groups)
            }
        )
        run_estimates.append(float(np.mean(correctness)))

    assert reference_groups is not None
    ordered_groups = np.asarray(sorted(reference_groups))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        selected_groups = rng.choice(
            ordered_groups,
            size=len(ordered_groups),
            replace=True,
        )
        sampled_run_estimates = []
        for grouped_run in grouped_correctness:
            sampled = np.concatenate([grouped_run[str(group)] for group in selected_groups])
            sampled_run_estimates.append(float(np.mean(sampled)))
        estimates.append(float(np.mean(sampled_run_estimates)))

    alpha = 1.0 - confidence
    return {
        "metric": "accuracy",
        "bootstrap_unit": "group",
        "run_aggregation": "equal_run_mean",
        "estimate": float(np.mean(run_estimates)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "groups": int(len(ordered_groups)),
        "runs": int(len(runs)),
    }


def paired_group_mean_delta_ci(
    reference_values: Sequence[float],
    candidate_values: Sequence[float],
    groups: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    reference = np.asarray(reference_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    group_array = np.asarray(groups).astype(str)
    if not (len(reference) == len(candidate) == len(group_array)):
        raise ValueError("paired values and groups must have the same length")
    if len(reference) == 0:
        raise ValueError("paired group bootstrap requires at least one sample")
    unique_groups = np.unique(group_array)
    grouped_deltas = {
        group: candidate[group_array == group] - reference[group_array == group]
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        estimates.append(
            float(np.mean(np.concatenate([grouped_deltas[group] for group in selected])))
        )
    alpha = 1.0 - confidence
    return {
        "metric": "paired_mean_delta",
        "bootstrap_unit": "group",
        "estimate": float(np.mean(candidate - reference)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "groups": int(len(unique_groups)),
    }


def paired_run_group_mean_delta_ci(
    runs: Sequence[
        tuple[
            Sequence[float],
            Sequence[float],
            Sequence[object],
        ]
    ],
    *,
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap paired group deltas and average the estimate across fixed runs."""
    if not runs:
        raise ValueError("paired run-group bootstrap requires at least one run")

    grouped_run_deltas: list[dict[str, np.ndarray]] = []
    observed_deltas: list[float] = []
    reference_groups: set[str] | None = None
    for run_index, (reference_values, candidate_values, groups) in enumerate(runs):
        reference = np.asarray(reference_values, dtype=float)
        candidate = np.asarray(candidate_values, dtype=float)
        group_array = np.asarray(groups).astype(str)
        if not (len(reference) == len(candidate) == len(group_array)):
            raise ValueError(
                f"run {run_index} paired values and groups must have the same length"
            )
        if len(reference) == 0:
            raise ValueError(f"run {run_index} has no paired values")

        unique_groups = set(np.unique(group_array).tolist())
        if reference_groups is None:
            reference_groups = unique_groups
        elif unique_groups != reference_groups:
            raise ValueError(
                "paired run-group bootstrap requires identical group IDs across runs"
            )

        deltas = candidate - reference
        grouped_run_deltas.append(
            {
                group: deltas[group_array == group]
                for group in sorted(unique_groups)
            }
        )
        observed_deltas.append(float(np.mean(deltas)))

    assert reference_groups is not None
    ordered_groups = np.asarray(sorted(reference_groups))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        selected_groups = rng.choice(
            ordered_groups,
            size=len(ordered_groups),
            replace=True,
        )
        run_estimates = []
        for grouped_deltas in grouped_run_deltas:
            sampled = np.concatenate(
                [grouped_deltas[str(group)] for group in selected_groups]
            )
            run_estimates.append(float(np.mean(sampled)))
        estimates.append(float(np.mean(run_estimates)))

    alpha = 1.0 - confidence
    return {
        "metric": "paired_mean_delta",
        "bootstrap_unit": "group",
        "run_aggregation": "equal_run_mean",
        "estimate": float(np.mean(observed_deltas)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "groups": int(len(ordered_groups)),
        "runs": int(len(runs)),
    }


def hierarchical_paired_metric_delta_ci(
    runs: Sequence[
        tuple[
            Sequence[int],
            Sequence[int],
            Sequence[int],
            Sequence[object],
        ]
    ],
    *,
    metric: str = "accuracy",
    n_resamples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap paired ablation deltas across training seeds and classes."""
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("hierarchical paired bootstrap supports accuracy or macro_f1")
    if not runs:
        raise ValueError("hierarchical paired bootstrap requires at least one run")

    prepared = []
    reference_groups: set[str] | None = None
    observed_deltas: list[float] = []
    for run_index, (y_true, reference_pred, candidate_pred, groups) in enumerate(runs):
        true_array = np.asarray(y_true, dtype=int)
        reference_array = np.asarray(reference_pred, dtype=int)
        candidate_array = np.asarray(candidate_pred, dtype=int)
        group_array = np.asarray(groups).astype(str)
        if not (
            len(true_array)
            == len(reference_array)
            == len(candidate_array)
            == len(group_array)
        ):
            raise ValueError(f"run {run_index} paired predictions and groups must align")
        if len(true_array) == 0:
            raise ValueError(f"run {run_index} has no paired predictions")
        unique_groups = set(np.unique(group_array).tolist())
        if reference_groups is None:
            reference_groups = unique_groups
        elif unique_groups != reference_groups:
            raise ValueError("hierarchical paired bootstrap requires identical group IDs")
        prepared.append((true_array, reference_array, candidate_array, group_array))

        if metric == "accuracy":
            reference_score = float(np.mean(reference_array == true_array))
            candidate_score = float(np.mean(candidate_array == true_array))
        else:
            labels = sorted(np.unique(true_array).tolist())
            reference_score = float(
                f1_score(true_array, reference_array, labels=labels, average="macro", zero_division=0)
            )
            candidate_score = float(
                f1_score(true_array, candidate_array, labels=labels, average="macro", zero_division=0)
            )
        observed_deltas.append(candidate_score - reference_score)

    assert reference_groups is not None
    ordered_groups = np.asarray(sorted(reference_groups))
    group_positions = {str(group): index for index, group in enumerate(ordered_groups)}
    rng = np.random.default_rng(seed)
    selected_run_plans = np.empty((n_resamples, len(prepared)), dtype=np.int64)
    group_count_plans = np.zeros(
        (n_resamples, len(ordered_groups)),
        dtype=np.int64,
    )
    for resample in range(n_resamples):
        selected_run_plans[resample] = rng.integers(
            0,
            len(prepared),
            size=len(prepared),
        )
        selected_groups = rng.choice(
            ordered_groups,
            size=len(ordered_groups),
            replace=True,
        )
        selected_positions = np.fromiter(
            (group_positions[str(group)] for group in selected_groups),
            dtype=np.int64,
            count=len(selected_groups),
        )
        group_count_plans[resample] = np.bincount(
            selected_positions,
            minlength=len(ordered_groups),
        )

    run_delta_matrix = np.empty((n_resamples, len(prepared)), dtype=float)
    sparse_group_counts = csr_matrix(group_count_plans)
    for run_index, (true_array, reference_array, candidate_array, group_array) in enumerate(prepared):
        sample_groups = np.fromiter(
            (group_positions[str(group)] for group in group_array),
            dtype=np.int64,
            count=len(group_array),
        )
        if metric == "accuracy":
            group_sizes = np.bincount(
                sample_groups,
                minlength=len(ordered_groups),
            )
            reference_correct = np.bincount(
                sample_groups,
                weights=(reference_array == true_array).astype(float),
                minlength=len(ordered_groups),
            )
            candidate_correct = np.bincount(
                sample_groups,
                weights=(candidate_array == true_array).astype(float),
                minlength=len(ordered_groups),
            )
            denominators = group_count_plans @ group_sizes
            reference_scores = (group_count_plans @ reference_correct) / denominators
            candidate_scores = (group_count_plans @ candidate_correct) / denominators
        else:
            labels = np.asarray(sorted(np.unique(true_array).tolist()), dtype=int)
            label_positions = {int(label): index for index, label in enumerate(labels)}
            true_positions = np.fromiter(
                (label_positions[int(label)] for label in true_array),
                dtype=np.int64,
                count=len(true_array),
            )
            shape = (len(ordered_groups), len(labels))
            ones = np.ones(len(true_array), dtype=np.int64)
            support_by_group = csr_matrix(
                (ones, (sample_groups, true_positions)),
                shape=shape,
            )

            def bootstrap_macro_f1(predictions: np.ndarray) -> np.ndarray:
                known = np.asarray(
                    [int(label) in label_positions for label in predictions],
                    dtype=bool,
                )
                prediction_positions = np.fromiter(
                    (
                        label_positions[int(label)]
                        for label in predictions[known]
                    ),
                    dtype=np.int64,
                    count=int(np.sum(known)),
                )
                predicted_by_group = csr_matrix(
                    (
                        np.ones(int(np.sum(known)), dtype=np.int64),
                        (sample_groups[known], prediction_positions),
                    ),
                    shape=shape,
                )
                correct = known & (predictions == true_array)
                true_positive_by_group = csr_matrix(
                    (
                        np.ones(int(np.sum(correct)), dtype=np.int64),
                        (sample_groups[correct], true_positions[correct]),
                    ),
                    shape=shape,
                )
                support = (sparse_group_counts @ support_by_group).toarray()
                predicted = (sparse_group_counts @ predicted_by_group).toarray()
                true_positive = (sparse_group_counts @ true_positive_by_group).toarray()
                denominators = support + predicted
                f1_values = np.divide(
                    2.0 * true_positive,
                    denominators,
                    out=np.zeros_like(denominators, dtype=float),
                    where=denominators != 0,
                )
                present = support > 0
                return np.sum(f1_values * present, axis=1) / np.sum(present, axis=1)

            reference_scores = bootstrap_macro_f1(reference_array)
            candidate_scores = bootstrap_macro_f1(candidate_array)
        run_delta_matrix[:, run_index] = candidate_scores - reference_scores

    estimates = np.mean(
        np.take_along_axis(run_delta_matrix, selected_run_plans, axis=1),
        axis=1,
    )

    alpha = 1.0 - confidence
    return {
        "metric": metric,
        "bootstrap_unit": "training_seed_then_group",
        "estimate": float(np.mean(observed_deltas)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "groups": int(len(ordered_groups)),
        "runs": int(len(prepared)),
    }
