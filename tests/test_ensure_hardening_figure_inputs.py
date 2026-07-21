from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image

from cattle_id.augmentation import apply_augmentation, get_augmentation_specs, resize_image
from cattle_id.hashing import sha256_file


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "ensure_hardening_figure_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("ensure_hardening_figure_inputs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_restore_cattlessfr_montage_reproduces_expected_byte_hashes(tmp_path: Path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source_file = "source.png"
    source_path = source_dir / source_file
    Image.new("RGB", (53, 47), (90, 130, 170)).save(source_path)

    expected_dir = tmp_path / "expected"
    target_dir = tmp_path / "restored"
    rows = []
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    for spec in get_augmentation_specs("all"):
        expected = expected_dir / f"{spec.identifier}.png"
        expected.parent.mkdir(parents=True, exist_ok=True)
        resize_image(
            apply_augmentation(
                source,
                spec,
                seed=1,
                source_id=source_file,
                protocol_version="hardening_v2",
            ),
            (32, 32),
        ).save(expected)
        rows.append(
            {
                "sample_id": f"000:{source_file}:{spec.identifier}",
                "source_file": source_file,
                "source_sha256": sha256_file(source_path),
                "image_path": str(target_dir / f"{spec.identifier}.png"),
                "image_sha256": sha256_file(expected),
                "augmentation_id": spec.identifier,
                "dataset_commit_sha": "a" * 40,
            }
        )
    shutil.rmtree(expected_dir)
    metadata = pd.DataFrame(rows)

    restored = MODULE.restore_cattlessfr_montage(
        metadata,
        source_dir=source_dir,
        image_size=(32, 32),
        augmentation_seed=1,
        expected_commit="a" * 40,
    )

    assert len(restored) == 20
    assert {row["augmentation_id"] for row in restored} == {
        spec.identifier for spec in get_augmentation_specs("all")
    }


def test_select_and_validate_holstein_error_images(tmp_path: Path):
    tables = tmp_path / "tables"
    runs = tmp_path / "runs"
    dataset = tmp_path / "Holstein2025"
    tables.mkdir()
    image_ok = dataset / "datasets_v2" / "query1" / "animal-a" / "ok.jpg"
    image_error = dataset / "datasets_v2" / "query1" / "animal-b" / "error.jpg"
    image_ok.parent.mkdir(parents=True)
    image_error.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), (10, 20, 30)).save(image_ok)
    Image.new("RGB", (20, 20), (40, 50, 60)).save(image_error)

    run_id = "fine-tuned-run"
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "control_type": "fine_tuned",
                "source_protocol": "paper_random_hardening_v2",
                "training_seed": 1,
            }
        ]
    ).to_csv(tables / "hardening_holstein_runs.csv", index=False)
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "image_path": "/missing/ok.jpg",
                "relative_path": image_ok.relative_to(dataset).as_posix(),
                "sha256": sha256_file(image_ok),
                "animal_id": "animal-a",
                "predicted_animal_id": "animal-a",
                "correct_rank_1": True,
            },
            {
                "image_path": "/missing/error.jpg",
                "relative_path": image_error.relative_to(dataset).as_posix(),
                "sha256": sha256_file(image_error),
                "animal_id": "animal-b",
                "predicted_animal_id": "animal-a",
                "correct_rank_1": False,
            },
        ]
    ).to_csv(run_dir / f"predictions_{MODULE.HOLSTEIN_SUFFIX}.csv", index=False)

    selected_run, errors = MODULE.select_holstein_error_rows(
        tables_dir=tables, runs_dir=runs
    )
    verified = MODULE.validate_holstein_error_images(errors, dataset_root=dataset)

    assert selected_run == run_id
    assert len(errors) == 1
    assert verified[0]["relative_path"].endswith("error.jpg")


def test_holstein_error_selection_prioritizes_distinct_true_identities(tmp_path: Path):
    tables = tmp_path / "tables"
    runs = tmp_path / "runs"
    tables.mkdir()
    run_id = "fine_tuned_run"
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "control_type": "fine_tuned",
                "source_protocol": "paper_random_hardening_v2",
                "training_seed": 1,
            }
        ]
    ).to_csv(tables / "hardening_holstein_runs.csv", index=False)
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    rows = []
    for index, animal_id in enumerate(["animal-a", "animal-a", "animal-b", "animal-c"]):
        rows.append(
            {
                "image_path": f"/missing/{index}.jpg",
                "relative_path": f"query/{index}.jpg",
                "sha256": f"hash-{index}",
                "animal_id": animal_id,
                "predicted_animal_id": "other",
                "correct_rank_1": False,
                "first_correct_rank": 10 - index,
                "average_precision": 0.1 + index / 100,
            }
        )
    pd.DataFrame(rows).to_csv(
        run_dir / f"predictions_{MODULE.HOLSTEIN_SUFFIX}.csv",
        index=False,
    )

    _, errors = MODULE.select_holstein_error_rows(
        tables_dir=tables,
        runs_dir=runs,
        limit=3,
    )

    assert len(errors) == 3
    assert errors["animal_id"].nunique() == 3
