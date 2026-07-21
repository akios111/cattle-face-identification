from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def load_evidence_scope(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        scope = yaml.safe_load(handle)
    if not isinstance(scope, dict):
        raise ValueError("evidence scope must be a mapping")
    groups = scope.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("evidence scope must define at least one group")
    names: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each evidence group must be a mapping")
        name = str(group.get("name", "")).strip()
        role = str(group.get("role", "")).strip()
        kind = str(group.get("kind", "")).strip()
        if not name or name in names:
            raise ValueError(f"invalid or duplicate evidence group name: {name!r}")
        if role not in {
            "legacy_replication",
            "confirmatory",
            "confirmatory_hardening_v2",
            "exploratory",
            "excluded",
        }:
            raise ValueError(f"unsupported evidence role for {name}: {role}")
        if kind not in {"classification", "retrieval"}:
            raise ValueError(f"unsupported evidence kind for {name}: {kind}")
        if not isinstance(group.get("match"), dict):
            raise ValueError(f"evidence group {name} requires a match mapping")
        names.add(name)
    return scope


def evidence_kind(row: dict[str, object]) -> str:
    if "cmc_rank_1" in row or row.get("evaluation_protocol"):
        return "retrieval"
    return "classification"


def _normalised_values(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values}


def _matches_list(value: object, candidates: object) -> bool:
    allowed = _normalised_values(candidates)
    return not allowed or str(value).strip().lower() in allowed


def _matches_seed(value: object, candidates: object) -> bool:
    if not isinstance(candidates, list) or not candidates:
        return True
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return False
    return seed in {int(candidate) for candidate in candidates}


def group_for_row(row: dict[str, object], scope: dict[str, Any]) -> dict[str, Any] | None:
    kind = evidence_kind(row)
    model = row.get("model", "")
    protocol = row.get("protocol", "")
    source_protocol = row.get("source_training_protocol", protocol)
    evaluation_protocol = row.get("evaluation_protocol", "")
    for group in scope.get("groups", []):
        if str(group.get("kind")) != kind:
            continue
        match = group["match"]
        if not _matches_list(model, match.get("models")):
            continue
        if kind == "classification" and not _matches_list(protocol, match.get("protocols")):
            continue
        if kind == "retrieval":
            if not _matches_list(source_protocol, match.get("source_protocols")):
                continue
            if not _matches_list(evaluation_protocol, match.get("evaluation_protocols")):
                continue
        if not _matches_seed(row.get("seed"), match.get("seeds")):
            continue
        if not _matches_seed(
            row.get("training_seed", row.get("seed")), match.get("training_seeds")
        ):
            continue
        if not _matches_seed(row.get("split_seed"), match.get("split_seeds")):
            continue
        if not _matches_seed(row.get("augmentation_seed"), match.get("augmentation_seeds")):
            continue
        if not _matches_list(row.get("protocol_version", "legacy"), match.get("protocol_versions")):
            continue
        return group
    return None


def annotate_rows(rows: Iterable[dict[str, object]], scope: dict[str, Any]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        group = group_for_row(row, scope)
        if group is None:
            continue
        item = dict(row)
        item["evidence_group"] = str(group["name"])
        item["evidence_role"] = str(group["role"])
        item["appendix_only"] = bool(group.get("appendix_only", False))
        item["main_protocol"] = str(row.get("protocol", "")) in {
            str(protocol) for protocol in group.get("main_protocols", [])
        }
        annotated.append(item)
    return annotated


def filter_predictions(
    predictions: dict[str, list[dict[str, str]]],
    scoped_metrics: Iterable[dict[str, object]],
) -> dict[str, list[dict[str, str]]]:
    allowed = {str(row.get("run_id", "")) for row in scoped_metrics}
    return {run_id: rows for run_id, rows in predictions.items() if run_id in allowed}
