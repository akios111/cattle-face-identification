from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from .logging_utils import append_event, log_line
from .models import get_preprocess_input


def _validate_heatmap(heatmap: np.ndarray, sample_id: str) -> None:
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError(f"Grad-CAM heatmap is invalid for {sample_id}")
    if float(np.max(heatmap)) <= 0.0 or int(np.count_nonzero(heatmap)) == 0:
        raise ValueError(f"Grad-CAM heatmap is empty for {sample_id}")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    pred_index: int | None = None,
) -> np.ndarray:
    if len(image_batch.shape) == 3:
        image_batch = tf.expand_dims(image_batch, axis=0)

    layer_names = [layer.name for layer in model.layers]
    if "backbone_feature_map" in layer_names:
        feature_map = model.get_layer("backbone_feature_map").output
    else:
        conv_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            raise ValueError("No convolutional layer found for Grad-CAM")
        feature_map = conv_layers[-1].output

    classifier = None
    try:
        candidate = model.get_layer("classifier")
        if isinstance(candidate, tf.keras.layers.Dense):
            classifier = candidate
    except ValueError:
        pass

    if classifier is not None:
        # Differentiate the pre-softmax score. Near-perfect classifiers can round
        # softmax probabilities to exactly 1.0 and erase the useful gradient.
        grad_model = tf.keras.Model(
            model.inputs,
            [feature_map, classifier.input],
        )
    else:
        grad_model = tf.keras.Model(model.inputs, [feature_map, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, score_inputs = grad_model(image_batch, training=False)
        if classifier is not None:
            logits = tf.linalg.matmul(
                score_inputs,
                tf.cast(classifier.kernel, score_inputs.dtype),
            )
            if classifier.bias is not None:
                logits = tf.nn.bias_add(
                    logits,
                    tf.cast(classifier.bias, score_inputs.dtype),
                )
            if pred_index is None:
                pred_index = int(tf.argmax(logits[0]))
            class_channel = logits[:, pred_index]
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(score_inputs[0]))
            class_channel = score_inputs[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def _diverse_prediction_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.head(0).copy()
    ordered = frame.copy()
    if "confidence" in ordered.columns:
        ordered = ordered.sort_values("confidence", ascending=False)
    diversity_keys = [key for key in ["source_file", "class_id"] if key in ordered.columns]
    if diversity_keys:
        ordered = ordered.drop_duplicates(subset=diversity_keys)
    return ordered.head(limit).copy()


def select_gradcam_samples(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    samples: str = "curated",
    limit: int = 12,
) -> pd.DataFrame:
    if predictions is None or samples != "curated":
        return (
            metadata[metadata["split"] == "test"]
            .drop_duplicates(subset=["source_file", "class_id"])
            .head(limit)
            .copy()
        )

    correct = _diverse_prediction_rows(
        predictions[predictions["class_id"] == predictions["predicted_class_id"]],
        limit // 2,
    )
    wrong = _diverse_prediction_rows(
        predictions[predictions["class_id"] != predictions["predicted_class_id"]],
        limit - len(correct),
    )
    selected_predictions = pd.concat([correct, wrong], ignore_index=True)
    if len(selected_predictions) < limit:
        selected_sources = set(selected_predictions["source_file"].tolist())
        remaining = predictions[
            ~predictions["source_file"].isin(selected_sources)
        ]
        filler = _diverse_prediction_rows(remaining, limit - len(selected_predictions))
        selected_predictions = pd.concat([selected_predictions, filler], ignore_index=True)
    if selected_predictions.empty:
        return (
            metadata[metadata["split"] == "test"]
            .drop_duplicates(subset=["source_file", "class_id"])
            .head(limit)
            .copy()
        )

    if "image_path" in selected_predictions.columns:
        selected = selected_predictions.copy()
    else:
        join_keys = ["source_file", "class_id", "augmentation_id", "split"]
        selected = selected_predictions.merge(
            metadata,
            on=join_keys,
            how="left",
            suffixes=("", "_metadata"),
        )
    if selected["image_path"].isna().any():
        missing = selected[selected["image_path"].isna()][["source_file", "augmentation_id", "split"]]
        raise ValueError(f"Could not resolve Grad-CAM image paths for rows: {missing.to_dict(orient='records')}")
    return selected.head(limit).copy()


def make_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0
    image_array = np.asarray(image).astype(np.float32)
    red_overlay = np.zeros_like(image_array)
    red_overlay[..., 0] = 255.0
    blended = image_array * (1.0 - alpha * heatmap_array[..., None]) + red_overlay * (
        alpha * heatmap_array[..., None]
    )
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def gradcam_faithfulness_curves(
    model: tf.keras.Model,
    image: np.ndarray,
    heatmap: np.ndarray,
    preprocess,
    *,
    steps: int = 20,
    seed: int = 1,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Compute deletion/insertion curves with an area-matched random control."""
    if steps < 2:
        raise ValueError("faithfulness curves require at least two steps")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("faithfulness image must have shape (height, width, 3)")
    resized_heatmap = np.asarray(
        Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
            (image.shape[1], image.shape[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    saliency_order = np.argsort(-resized_heatmap.reshape(-1), kind="mergesort")
    random_order = np.random.default_rng(seed).permutation(len(saliency_order))
    original_probability = np.asarray(
        model.predict(preprocess(image[None, ...]), verbose=0)
    )[0]
    target_class = int(np.argmax(original_probability))
    neutral = np.full_like(image, 127.0)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    batches: list[np.ndarray] = []
    labels: list[tuple[str, float]] = []
    for fraction in fractions:
        count = int(round(float(fraction) * len(saliency_order)))
        for control_name, order in (("gradcam", saliency_order), ("random", random_order)):
            selected = order[:count]
            deletion = image.copy().reshape(-1, 3)
            deletion[selected] = 127.0
            insertion = neutral.copy().reshape(-1, 3)
            source = image.reshape(-1, 3)
            insertion[selected] = source[selected]
            batches.extend([deletion.reshape(image.shape), insertion.reshape(image.shape)])
            labels.extend(
                [
                    (f"deletion_{control_name}", float(fraction)),
                    (f"insertion_{control_name}", float(fraction)),
                ]
            )
    probabilities = np.asarray(
        model.predict(preprocess(np.stack(batches)), verbose=0)
    )[:, target_class]
    rows = [
        {"curve": curve, "fraction": fraction, "target_probability": float(probability)}
        for (curve, fraction), probability in zip(labels, probabilities, strict=True)
    ]
    frame = pd.DataFrame(rows)
    metrics: dict[str, float | int] = {"target_class": target_class, "steps": int(steps)}
    for curve, group in frame.groupby("curve"):
        ordered = group.sort_values("fraction")
        metrics[f"{curve}_auc"] = float(
            np.trapezoid(ordered["target_probability"], ordered["fraction"])
        )
    return metrics, frame


def save_gradcam_for_run(
    run_dir: str | Path,
    samples: str = "curated",
    limit: int = 12,
    faithfulness_steps: int = 20,
) -> Path:
    run_dir = Path(run_dir)
    log_path = run_dir / "gradcam.log"
    log_line(log_path, "gradcam", f"run_dir={run_dir} samples={samples} limit={limit}")
    append_event(run_dir, "gradcam_started", samples=samples, limit=limit)
    log_line(log_path, "gradcam", "loading manifest")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    log_line(log_path, "gradcam", f"reading metadata={manifest['metadata_path']}")
    metadata = pd.read_csv(manifest["metadata_path"])
    predictions_path = run_dir / "predictions.csv"
    if predictions_path.exists() and samples == "curated":
        predictions = pd.read_csv(predictions_path)
        selected = select_gradcam_samples(metadata, predictions, samples=samples, limit=limit)
    else:
        selected = select_gradcam_samples(metadata, samples=samples, limit=limit)

    log_line(log_path, "gradcam", f"selected_samples={len(selected)}")
    log_line(log_path, "gradcam", "loading model")
    model = tf.keras.models.load_model(run_dir / "model.keras")
    preprocess = get_preprocess_input(manifest["model"])
    output_dir = run_dir / "gradcam"
    output_dir.mkdir(exist_ok=True)
    image_size = tuple(manifest["image_size"])
    sample_rows: list[dict[str, object]] = []
    faithfulness_rows: list[pd.DataFrame] = []
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        image = Image.open(row["image_path"]).convert("RGB").resize(image_size)
        array = np.asarray(image).astype(np.float32)
        batch = preprocess(np.expand_dims(array, axis=0))
        heatmap = make_gradcam_heatmap(model, tf.convert_to_tensor(batch))
        sample_id = str(row.get("sample_id", Path(row["image_path"]).stem))
        _validate_heatmap(heatmap, sample_id)
        heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image_size)
        overlay_image = make_overlay(image, heatmap)
        output_stem = Path(row["image_path"]).stem
        original_path = output_dir / f"{output_stem}_original.png"
        heatmap_path = output_dir / f"{output_stem}_heatmap.png"
        overlay_path = output_dir / f"{output_stem}_overlay.png"
        image.save(original_path)
        heatmap_image.save(heatmap_path)
        overlay_image.save(overlay_path)
        faithfulness, curves = gradcam_faithfulness_curves(
            model,
            array,
            heatmap,
            preprocess,
            steps=faithfulness_steps,
            seed=index,
        )
        curves.insert(0, "sample_id", str(row.get("sample_id", output_stem)))
        faithfulness_rows.append(curves)
        sample_row = row.to_dict()
        sample_row["original_path"] = str(original_path)
        sample_row["heatmap_path"] = str(heatmap_path)
        sample_row["overlay_path"] = str(overlay_path)
        sample_row["gradcam_score_space"] = "pre_softmax_logit"
        sample_row["heatmap_nonzero_fraction"] = float(np.count_nonzero(heatmap) / heatmap.size)
        sample_row.update(faithfulness)
        sample_rows.append(sample_row)
        log_line(log_path, "gradcam", f"saved {index}/{len(selected)} {overlay_path}")
    pd.DataFrame(sample_rows).to_csv(output_dir / "gradcam_samples.csv", index=False)
    if faithfulness_rows:
        pd.concat(faithfulness_rows, ignore_index=True).to_csv(
            output_dir / "gradcam_faithfulness_curves.csv",
            index=False,
        )
    log_line(log_path, "gradcam", f"samples_written={output_dir / 'gradcam_samples.csv'}")
    append_event(run_dir, "gradcam_completed", output_dir=str(output_dir), samples=int(len(selected)))
    return output_dir


def _resolve_saved_artifact(path_value: object, run_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    candidates.append(run_dir / "gradcam" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def refresh_saved_gradcam(
    run_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    faithfulness_steps: int = 20,
) -> Path:
    """Recompute Grad-CAM outputs from retained original images without metadata materialization."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "gradcam"
    samples_path = output_dir / "gradcam_samples.csv"
    samples = pd.read_csv(samples_path)
    if samples.empty:
        raise ValueError(f"No saved Grad-CAM samples in {samples_path}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(Path(model_path) if model_path else run_dir / "model.keras")
    preprocess = get_preprocess_input(manifest["model"])
    image_size = tuple(manifest["image_size"])
    refreshed_rows: list[dict[str, object]] = []
    faithfulness_rows: list[pd.DataFrame] = []

    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        sample_id = str(row.get("sample_id", index))
        original_path = _resolve_saved_artifact(row["original_path"], run_dir)
        heatmap_path = _resolve_saved_artifact(row["heatmap_path"], run_dir)
        overlay_path = _resolve_saved_artifact(row["overlay_path"], run_dir)
        image = Image.open(original_path).convert("RGB").resize(image_size)
        array = np.asarray(image).astype(np.float32)
        batch = preprocess(np.expand_dims(array, axis=0))
        heatmap = make_gradcam_heatmap(model, tf.convert_to_tensor(batch))
        _validate_heatmap(heatmap, sample_id)
        Image.fromarray(np.uint8(255 * heatmap)).resize(image_size).save(heatmap_path)
        make_overlay(image, heatmap).save(overlay_path)
        faithfulness, curves = gradcam_faithfulness_curves(
            model,
            array,
            heatmap,
            preprocess,
            steps=faithfulness_steps,
            seed=index,
        )
        curves.insert(0, "sample_id", sample_id)
        faithfulness_rows.append(curves)
        sample_row = row.to_dict()
        sample_row["gradcam_score_space"] = "pre_softmax_logit"
        sample_row["heatmap_nonzero_fraction"] = float(np.count_nonzero(heatmap) / heatmap.size)
        sample_row.update(faithfulness)
        refreshed_rows.append(sample_row)

    pd.DataFrame(refreshed_rows).to_csv(samples_path, index=False)
    pd.concat(faithfulness_rows, ignore_index=True).to_csv(
        output_dir / "gradcam_faithfulness_curves.csv",
        index=False,
    )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for a run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--samples", default="curated")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--faithfulness-steps", type=int, default=20)
    parser.add_argument("--refresh-saved", action="store_true")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args(argv)
    if args.refresh_saved:
        print(
            refresh_saved_gradcam(
                args.run,
                model_path=args.model,
                faithfulness_steps=args.faithfulness_steps,
            )
        )
        return
    print(
        save_gradcam_for_run(
            args.run,
            samples=args.samples,
            limit=args.limit,
            faithfulness_steps=args.faithfulness_steps,
        )
    )


if __name__ == "__main__":
    main()
