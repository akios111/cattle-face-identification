from __future__ import annotations

from typing import Callable

import tensorflow as tf


_APPLICATIONS: dict[str, tuple[Callable[..., tf.keras.Model], Callable]] = {
    "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "efficientnetv2b0": (
        tf.keras.applications.EfficientNetV2B0,
        tf.keras.applications.efficientnet_v2.preprocess_input,
    ),
    "efficientnetv2b3": (
        tf.keras.applications.EfficientNetV2B3,
        tf.keras.applications.efficientnet_v2.preprocess_input,
    ),
    "convnexttiny": (
        tf.keras.applications.ConvNeXtTiny,
        tf.keras.applications.convnext.preprocess_input,
    ),
    "convnextbase": (
        tf.keras.applications.ConvNeXtBase,
        tf.keras.applications.convnext.preprocess_input,
    ),
}


def normalize_model_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "vgg16": "vgg16",
        "resnet50": "resnet50",
        "mobilenetv2": "mobilenetv2",
        "densenet121": "densenet121",
        "efficientnetb0": "efficientnetb0",
        "efficientnetv2b0": "efficientnetv2b0",
        "efficientnetv2b3": "efficientnetv2b3",
        "convnexttiny": "convnexttiny",
        "convnextbase": "convnextbase",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported model: {name}")
    return aliases[key]


def get_preprocess_input(name: str) -> Callable:
    return _APPLICATIONS[normalize_model_name(name)][1]


def build_model(
    name: str,
    num_classes: int = 311,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    dropout: float = 0.2,
) -> tf.keras.Model:
    model_name = normalize_model_name(name)
    application, _ = _APPLICATIONS[model_name]
    base = application(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    backbone = tf.keras.Model(base.input, base.output, name="backbone")
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.Activation("linear", name="backbone_feature_map")(features)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    dropped = tf.keras.layers.Dropout(dropout, name="dropout")(pooled)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classifier",
    )(dropped)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cattle_{model_name}")


def build_embedding_model(model: tf.keras.Model) -> tf.keras.Model:
    try:
        embedding_layer = model.get_layer("global_average_pooling")
    except ValueError as exc:
        raise ValueError("Model does not expose the global_average_pooling embedding layer") from exc
    return tf.keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output,
        name=f"{model.name}_embedding",
    )


def build_imagenet_embedding_model(
    name: str,
    *,
    input_shape: tuple[int, int, int] = (384, 384, 3),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build an unadapted ImageNet backbone with global-average embeddings."""
    model_name = normalize_model_name(name)
    application, _ = _APPLICATIONS[model_name]
    return application(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg",
    )


def set_backbone_trainable(
    model: tf.keras.Model,
    trainable: bool,
    trainable_last_n: int | None = None,
    freeze_batchnorm: bool = True,
) -> None:
    backbone = model.get_layer("backbone")
    if not trainable:
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False
        return

    backbone.trainable = True
    layers = list(backbone.layers)
    trainable_start = 0
    if trainable_last_n is not None:
        trainable_start = max(0, len(layers) - int(trainable_last_n))

    for index, layer in enumerate(layers):
        layer.trainable = index >= trainable_start
        if freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
