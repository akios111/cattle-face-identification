from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

tf = pytest.importorskip("tensorflow")

from cattle_id.tfdata import dataframe_to_dataset


def _make_metadata(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for class_id in range(4):
        image_path = tmp_path / f"class_{class_id}.png"
        Image.new("RGB", (12, 12), (class_id * 40, class_id * 20, class_id * 10)).save(image_path)
        rows.append({"image_path": str(image_path), "class_id": class_id})
    return pd.DataFrame(rows)


def test_evaluation_dataset_preserves_metadata_label_order(tmp_path):
    metadata = _make_metadata(tmp_path)

    dataset = dataframe_to_dataset(
        metadata,
        "mobilenetv2",
        image_size=(16, 16),
        batch_size=2,
        shuffle=False,
    )

    labels = []
    for _, batch_labels in dataset:
        labels.extend(int(label) for label in batch_labels.numpy())

    assert labels == [0, 1, 2, 3]
    assert dataset.options().experimental_deterministic is True


def test_training_dataset_allows_non_deterministic_parallel_loading(tmp_path):
    metadata = _make_metadata(tmp_path)

    dataset = dataframe_to_dataset(
        metadata,
        "mobilenetv2",
        image_size=(16, 16),
        batch_size=2,
        shuffle=True,
    )

    assert dataset.options().experimental_deterministic is False


def test_dataset_loads_jpeg_images_for_external_acquisition(tmp_path):
    image_path = tmp_path / "external.jpg"
    Image.new("RGB", (12, 12), (120, 80, 40)).save(image_path, format="JPEG")
    metadata = pd.DataFrame([{"image_path": str(image_path), "class_id": 0}])

    dataset = dataframe_to_dataset(
        metadata,
        "mobilenetv2",
        image_size=(16, 16),
        batch_size=1,
        shuffle=False,
    )
    images, labels = next(iter(dataset))

    assert tuple(images.shape) == (1, 16, 16, 3)
    assert labels.numpy().tolist() == [0]
