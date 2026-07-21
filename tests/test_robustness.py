from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cattle_id.robustness import (
    apply_region_variant,
    materialize_region_audit,
    materialize_severity_sweep,
)


def _metadata(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index, color in enumerate(((30, 60, 90), (100, 130, 160))):
        path = tmp_path / f"cow_{index}.png"
        Image.new("RGB", (20, 20), color).save(path)
        rows.append(
            {
                "sample_id": f"cow-{index}:original",
                "image_path": str(path),
                "class_id": index,
                "augmentation_id": "original",
                "augmentation_family": "original",
                "split": "test",
            }
        )
    return pd.DataFrame(rows)


def test_random_region_controls_are_deterministic_per_sample():
    image = Image.new("RGB", (20, 20), (20, 40, 60))
    first = apply_region_variant(image, "random_area_mask", sample_id="cow-1", seed=1)
    repeated = apply_region_variant(image, "random_area_mask", sample_id="cow-1", seed=1)
    other = apply_region_variant(image, "random_area_mask", sample_id="cow-2", seed=1)

    assert np.array_equal(np.asarray(first), np.asarray(repeated))
    assert not np.array_equal(np.asarray(first), np.asarray(other))


def test_region_audit_preserves_labels_and_expands_five_variants(tmp_path: Path):
    output = materialize_region_audit(_metadata(tmp_path), tmp_path / "regions")

    assert len(output) == 10
    assert output.groupby("original_sample_id")["region_variant"].nunique().eq(5).all()
    assert set(output["class_id"]) == {0, 1}
    assert output["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_severity_sweep_has_fixed_levels_and_both_rotation_directions(tmp_path: Path):
    output = materialize_severity_sweep(_metadata(tmp_path), tmp_path / "severity")

    # Per source: rotation has 1 zero + 4*2 directions; three other families have 5 levels.
    assert len(output) == 2 * 24
    rotation = output[output["severity_family"] == "rotation"]
    assert set(rotation["severity_value"]) == {0.0, 5.0, 10.0, 15.0, 20.0}
    assert set(rotation.loc[rotation["severity_value"] > 0, "severity_direction"]) == {-1, 1}
    assert set(output["split"]) == {"test"}
