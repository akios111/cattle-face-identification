from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


LEGACY_PROTOCOL_VERSION = "legacy"
HARDENING_PROTOCOL_VERSION = "hardening_v2"


def _base_protocol_name(protocol: str) -> tuple[str, str]:
    protocol_name = protocol.lower()
    suffix = f"_{HARDENING_PROTOCOL_VERSION}"
    if protocol_name.endswith(suffix):
        return protocol_name[: -len(suffix)], HARDENING_PROTOCOL_VERSION
    return protocol_name, LEGACY_PROTOCOL_VERSION


def run_seeds_from_config(config: dict[str, Any]) -> dict[str, int]:
    """Return the independently controlled seeds with legacy fallbacks."""
    training = config.get("training", {})
    split = config.get("split", {})
    augmentation = config.get("augmentation", {})
    legacy_seed = int(training.get("seed", 2026))
    return {
        "training_seed": int(training.get("training_seed", legacy_seed)),
        "split_seed": int(split.get("seed", split.get("split_seed", legacy_seed))),
        "augmentation_seed": int(
            augmentation.get("seed", augmentation.get("augmentation_seed", legacy_seed))
        ),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def image_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    value = config.get("preprocessing", {}).get("image_size", [224, 224])
    if len(value) != 2:
        raise ValueError("preprocessing.image_size must contain two integers")
    return int(value[0]), int(value[1])


def config_for_run(
    config: dict[str, Any],
    *,
    protocol: str | None = None,
    seed: int | None = None,
    training_seed: int | None = None,
    split_seed: int | None = None,
    augmentation_seed: int | None = None,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output.setdefault("preprocessing", {})
    output.setdefault("augmentation", {})
    output.setdefault("split", {})
    output.setdefault("training", {})

    # ``seed`` preserves the historical API: one value controls the entire
    # pipeline. New hardening runs pass the three explicit seed values.
    if seed is not None:
        output["training"]["seed"] = int(seed)
        output["training"]["training_seed"] = int(seed)
        output["split"]["seed"] = int(seed)
        output["augmentation"]["seed"] = int(seed)

    if training_seed is not None:
        output["training"]["training_seed"] = int(training_seed)
        output["training"]["seed"] = int(training_seed)
    if split_seed is not None:
        output["split"]["seed"] = int(split_seed)
    if augmentation_seed is not None:
        output["augmentation"]["seed"] = int(augmentation_seed)

    if protocol is None:
        return output

    protocol_name, inferred_version = _base_protocol_name(protocol)
    selected_version = protocol_version or (
        inferred_version
        if inferred_version == HARDENING_PROTOCOL_VERSION
        else output.get("protocol_version", LEGACY_PROTOCOL_VERSION)
    )
    if selected_version not in {LEGACY_PROTOCOL_VERSION, HARDENING_PROTOCOL_VERSION}:
        raise ValueError(f"Unknown protocol version: {selected_version}")
    output["protocol_version"] = selected_version
    output["split"]["protocol"] = "paper_random"
    if protocol_name == "paper_random":
        output["split"]["protocol"] = "paper_random"
    elif protocol_name == "augmentation_ablation":
        output["split"]["protocol"] = "augmentation_ablation"
    elif protocol_name == "transform_holdout":
        output["split"]["protocol"] = "transform_holdout"
    elif protocol_name == "ablation_no_aug":
        output["augmentation"]["profile"] = "none"
    elif protocol_name == "ablation_geometric_only":
        output["augmentation"]["profile"] = "geometric"
    elif protocol_name == "ablation_no_cutout":
        if selected_version == HARDENING_PROTOCOL_VERSION:
            output["augmentation"]["profile"] = "all"
            output["training"]["exclude_augmentation_families"] = ["cutout"]
        else:
            output["augmentation"]["profile"] = "no_cutout"
    elif protocol_name == "ablation_cutout_only":
        output["augmentation"]["profile"] = "cutout_only"
    elif protocol_name == "ablation_224":
        output["preprocessing"]["image_size"] = [224, 224]
    elif protocol_name == "ablation_frozen":
        output["training"].setdefault("fine_tune", {})
        output["training"]["fine_tune"]["enabled"] = False
    elif protocol_name == "holstein2025_closed_set":
        output["split"]["protocol"] = "holstein2025_closed_set"
    else:
        raise ValueError(f"Unknown experiment protocol: {protocol}")
    return output


def metadata_filename(
    protocol: str,
    *,
    seed: int | None = None,
    split_seed: int | None = None,
    augmentation_seed: int | None = None,
) -> str:
    if split_seed is not None or augmentation_seed is not None:
        split_part = f"_split{int(split_seed)}" if split_seed is not None else ""
        augmentation_part = (
            f"_aug{int(augmentation_seed)}" if augmentation_seed is not None else ""
        )
        return f"metadata_{protocol}{split_part}{augmentation_part}.csv"
    suffix = f"_seed{int(seed)}" if seed is not None else ""
    return f"metadata_{protocol}{suffix}.csv"
