from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))

import external_public_benchmark_summary


def _metadata(path: Path) -> Path:
    rows = []
    for class_id, animal_id in enumerate(["Cattle_1", "Cattle_2"]):
        for index, split in enumerate(["train", "train", "validation", "test"]):
            rows.append(
                {
                    "image_path": f"{animal_id}_{index}.jpg",
                    "class_id": class_id,
                    "animal_id": animal_id,
                    "external_dataset": "cattely",
                    "protocol": "external_public_face_benchmark",
                    "split": split,
                    "source_url": "https://github.com/example/cattely",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def test_collect_summary_marks_ready_when_thresholds_and_splits_are_present(tmp_path):
    metadata_path = _metadata(tmp_path / "metadata.csv")

    summary = external_public_benchmark_summary.collect_summary(
        metadata_path=metadata_path,
        expected_animals=2,
        expected_images_per_animal=4,
    )

    assert summary["ready"] is True
    assert summary["animals"] == 2
    assert summary["images"] == 8
    assert summary["min_images_per_animal"] == 4
    assert summary["split_counts"] == {"test": 2, "train": 4, "validation": 2}
    assert summary["issues"] == []


def test_collect_summary_marks_not_ready_when_thresholds_are_not_met(tmp_path):
    metadata_path = _metadata(tmp_path / "metadata.csv")

    summary = external_public_benchmark_summary.collect_summary(
        metadata_path=metadata_path,
        expected_animals=50,
        expected_images_per_animal=50,
    )

    assert summary["ready"] is False
    assert "animal count below threshold" in summary["issues"][0]
    assert "images per animal below threshold" in summary["issues"][1]


def test_collect_summary_preserves_excluded_low_image_count_split(tmp_path):
    metadata_path = _metadata(tmp_path / "metadata.csv")
    frame = pd.read_csv(metadata_path)
    frame.loc[len(frame)] = {
        "image_path": "Cattle_3_only.jpg",
        "class_id": 2,
        "animal_id": "Cattle_3",
        "external_dataset": "cattely",
        "protocol": "external_public_face_benchmark",
        "split": "excluded_low_image_count",
        "source_url": "https://github.com/example/cattely",
    }
    frame.to_csv(metadata_path, index=False)

    summary = external_public_benchmark_summary.collect_summary(
        metadata_path=metadata_path,
        expected_animals=2,
        expected_images_per_animal=1,
    )

    assert summary["split_counts"]["excluded_low_image_count"] == 1


def test_write_summary_outputs_markdown_csv_and_tex(tmp_path):
    metadata_path = _metadata(tmp_path / "metadata.csv")
    gates_dir = tmp_path / "gates"
    tables_dir = tmp_path / "tables"
    summary = external_public_benchmark_summary.collect_summary(
        metadata_path=metadata_path,
        expected_animals=2,
        expected_images_per_animal=4,
    )

    generated = external_public_benchmark_summary.write_summary(summary, gates_dir=gates_dir, tables_dir=tables_dir)

    assert {path.name for path in generated} == {
        "external-public-benchmark-readiness.md",
        "external_public_benchmark_summary.csv",
        "external_public_benchmark_summary.tex",
    }
    assert "External public benchmark ready True." in (
        gates_dir / "external-public-benchmark-readiness.md"
    ).read_text(encoding="utf-8")
