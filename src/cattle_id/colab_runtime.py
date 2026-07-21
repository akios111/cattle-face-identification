from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HolsteinRuntimeConfigs:
    readiness: Path
    training: Path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset"), dict):
        raise ValueError(f"Config must contain a dataset mapping: {path}")
    return payload


def _write_config(source: Path, destination: Path, dataset_root: Path) -> Path:
    payload = _load_config(source)
    payload["dataset"]["raw_dir"] = str(dataset_root)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_holstein_runtime_configs(
    *,
    readiness_source: str | Path,
    training_source: str | Path,
    runtime_dir: str | Path,
    dataset_root: str | Path,
) -> HolsteinRuntimeConfigs:
    """Create Colab-local configs while leaving persistent project configs unchanged."""
    readiness_source = Path(readiness_source)
    training_source = Path(training_source)
    runtime_dir = Path(runtime_dir)
    dataset_root = Path(dataset_root).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    readiness = _write_config(
        readiness_source,
        runtime_dir / "holstein2025_open_set.runtime.yaml",
        dataset_root,
    )
    training = _write_config(
        training_source,
        runtime_dir / "holstein2025_colab_proplus.runtime.yaml",
        dataset_root,
    )
    return HolsteinRuntimeConfigs(readiness=readiness, training=training)
