import numpy as np
import pandas as pd
from PIL import Image
import pytest

tf = pytest.importorskip("tensorflow")

from cattle_id.gradcam import (
    gradcam_faithfulness_curves,
    make_gradcam_heatmap,
    make_overlay,
    select_gradcam_samples,
)
from cattle_id.models import build_model, normalize_model_name, set_backbone_trainable


def test_build_model_uses_frozen_backbone_and_311_class_softmax():
    model = build_model(
        "mobilenetv2",
        num_classes=311,
        input_shape=(64, 64, 3),
        weights=None,
    )

    assert model.output_shape == (None, 311)
    assert model.get_layer("backbone").trainable is False

    output = model(tf.ones((1, 64, 64, 3)), training=False).numpy()
    assert output.shape == (1, 311)
    assert np.allclose(output.sum(axis=1), np.array([1.0]), atol=1e-5)


def test_modern_backbones_are_available_for_colab_pro_plus():
    assert normalize_model_name("efficientnetv2-b0") == "efficientnetv2b0"
    assert normalize_model_name("convnext-tiny") == "convnexttiny"

    model = build_model(
        "efficientnetv2b0",
        num_classes=7,
        input_shape=(64, 64, 3),
        weights=None,
    )

    assert model.output_shape == (None, 7)
    assert model.get_layer("classifier").dtype == "float32"


def test_set_backbone_trainable_unfreezes_only_last_layers_and_keeps_batchnorm_frozen():
    model = build_model(
        "mobilenetv2",
        num_classes=3,
        input_shape=(64, 64, 3),
        weights=None,
    )

    set_backbone_trainable(model, trainable=True, trainable_last_n=8, freeze_batchnorm=True)
    backbone = model.get_layer("backbone")
    trainable_layers = [layer for layer in backbone.layers if layer.trainable]

    assert backbone.trainable is True
    assert 0 < len(trainable_layers) <= 8
    assert all(not isinstance(layer, tf.keras.layers.BatchNormalization) for layer in trainable_layers)


def test_gradcam_returns_non_empty_heatmap_for_built_model():
    model = build_model(
        "mobilenetv2",
        num_classes=3,
        input_shape=(64, 64, 3),
        weights=None,
    )

    heatmap = make_gradcam_heatmap(model, tf.ones((1, 64, 64, 3)))

    assert heatmap.ndim == 2
    assert heatmap.shape[0] > 0
    assert heatmap.shape[1] > 0
    assert np.isfinite(heatmap).all()


def test_gradcam_uses_logits_when_softmax_is_saturated():
    inputs = tf.keras.Input(shape=(4, 4, 1), name="image")
    features = tf.keras.layers.Conv2D(
        1,
        1,
        use_bias=False,
        kernel_initializer="ones",
        name="backbone_feature_map",
    )(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="classifier")(pooled)
    model = tf.keras.Model(inputs, outputs)
    model.get_layer("classifier").set_weights(
        [np.array([[100.0, -100.0]], dtype=np.float32), np.zeros(2, dtype=np.float32)]
    )
    batch = tf.ones((1, 4, 4, 1), dtype=tf.float32)

    probabilities = model(batch, training=False).numpy()[0]
    heatmap = make_gradcam_heatmap(model, batch)

    assert probabilities[0] == pytest.approx(1.0)
    assert float(heatmap.max()) == pytest.approx(1.0)
    assert np.count_nonzero(heatmap) == heatmap.size


def test_gradcam_curated_selection_uses_exact_prediction_rows():
    metadata = pd.DataFrame(
        [
            {
                "source_file": "cow_001.png",
                "class_id": 1,
                "augmentation_id": "original",
                "split": "train",
                "image_path": "train_wrong_row.png",
            },
            {
                "source_file": "cow_001.png",
                "class_id": 1,
                "augmentation_id": "cutout_random",
                "split": "test",
                "image_path": "test_exact_row.png",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "source_file": "cow_001.png",
                "class_id": 1,
                "augmentation_id": "cutout_random",
                "split": "test",
                "predicted_class_id": 2,
                "confidence": 0.4,
            }
        ]
    )

    selected = select_gradcam_samples(metadata, predictions, samples="curated", limit=1)

    assert selected["image_path"].tolist() == ["test_exact_row.png"]


def test_gradcam_curated_selection_prefers_diverse_source_files():
    metadata = pd.DataFrame(
        [
            {
                "source_file": f"cow_{source}.png",
                "class_id": source,
                "augmentation_id": augmentation,
                "split": "test",
                "image_path": f"cow_{source}_{augmentation}.png",
            }
            for source in range(3)
            for augmentation in ["original", "blur"]
        ]
    )
    predictions = metadata.copy()
    predictions["predicted_class_id"] = predictions["class_id"]
    predictions["confidence"] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]

    selected = select_gradcam_samples(metadata, predictions, samples="curated", limit=3)

    assert selected["source_file"].nunique() == 3
    assert len(selected) == 3


def test_gradcam_overlay_matches_input_size():
    image = tf.ones((8, 8, 3)).numpy()
    pil_image = Image.fromarray(np.uint8(image * 127))
    heatmap = np.ones((4, 4), dtype=np.float32)

    overlay = make_overlay(pil_image, heatmap)

    assert overlay.size == pil_image.size


def test_gradcam_faithfulness_curves_include_area_matched_random_controls():
    class MeanModel:
        def predict(self, batch, verbose=0):
            values = np.asarray(batch, dtype=np.float32).mean(axis=(1, 2, 3)) / 255.0
            return np.stack([values, 1.0 - values], axis=1)

    image = np.full((8, 8, 3), 200.0, dtype=np.float32)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[:2, :2] = 1.0

    metrics, curves = gradcam_faithfulness_curves(
        MeanModel(),
        image,
        heatmap,
        lambda value: value,
        steps=4,
        seed=1,
    )

    assert set(curves["curve"]) == {
        "deletion_gradcam",
        "insertion_gradcam",
        "deletion_random",
        "insertion_random",
    }
    assert len(curves) == 4 * 5
    assert 0.0 <= metrics["deletion_gradcam_auc"] <= 1.0
    assert 0.0 <= metrics["insertion_random_auc"] <= 1.0
