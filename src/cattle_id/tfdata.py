from __future__ import annotations

import pandas as pd
import tensorflow as tf

from .models import get_preprocess_input


def dataframe_to_dataset(
    metadata: pd.DataFrame,
    model_name: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 2026,
    cache: bool | str = False,
    deterministic: bool | None = None,
) -> tf.data.Dataset:
    preprocess_input = get_preprocess_input(model_name)
    paths = metadata["image_path"].astype(str).tolist()
    labels = metadata["class_id"].astype("int32").tolist()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image, label

    deterministic = (not shuffle) if deterministic is None else deterministic
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic,
    )
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    dataset = dataset.with_options(options)
    if cache:
        dataset = dataset.cache(str(cache)) if isinstance(cache, str) else dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
