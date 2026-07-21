from __future__ import annotations

from typing import Any

import tensorflow as tf


def _gpu_names() -> list[str]:
    return [device.name for device in tf.config.list_physical_devices("GPU")]


def configure_tensorflow_acceleration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    xla = bool(config.get("xla", False))
    tf.config.optimizer.set_jit(xla)

    policy = str(config.get("mixed_precision", "float32")).lower()
    if policy == "auto":
        policy = "mixed_float16" if gpus else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)

    return {
        "gpus": _gpu_names(),
        "xla": xla,
        "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
    }
