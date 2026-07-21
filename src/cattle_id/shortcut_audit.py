from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from skimage.feature import hog

from .hashing import sha256_file
from .metrics import bootstrap_cluster_accuracy_ci
from .models import build_imagenet_embedding_model
from .tfdata import dataframe_to_dataset


FEATURE_METHODS = (
    "phash",
    "pixels32",
    "hsv_histogram",
    "hog",
    "imagenet_efficientnetv2b3",
)


def _open_rgb(path: object) -> Image.Image:
    with Image.open(str(path)) as image:
        return image.convert("RGB")


def _phash_feature(path: object) -> np.ndarray:
    value = imagehash.phash(_open_rgb(path), hash_size=8)
    return np.asarray(value.hash, dtype=np.float32).reshape(-1)


def _pixels_feature(path: object) -> np.ndarray:
    image = _open_rgb(path).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def _hsv_histogram_feature(path: object) -> np.ndarray:
    image = _open_rgb(path).convert("HSV").resize((128, 128), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3) / 255.0
    histogram, _ = np.histogramdd(
        pixels,
        bins=(16, 4, 4),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    flattened = histogram.astype(np.float32).reshape(-1)
    total = float(flattened.sum())
    return flattened / total if total else flattened


def _hog_feature(path: object) -> np.ndarray:
    image = _open_rgb(path).convert("L").resize((96, 96), Image.Resampling.BILINEAR)
    return np.asarray(
        hog(
            np.asarray(image, dtype=np.float32) / 255.0,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ),
        dtype=np.float32,
    )


def extract_handcrafted_features(
    metadata: pd.DataFrame,
    method: str,
) -> np.ndarray:
    extractors: dict[str, Callable[[object], np.ndarray]] = {
        "phash": _phash_feature,
        "pixels32": _pixels_feature,
        "hsv_histogram": _hsv_histogram_feature,
        "hog": _hog_feature,
    }
    if method not in extractors:
        raise ValueError(f"Unsupported handcrafted shortcut method: {method}")
    if metadata.empty:
        raise ValueError("feature extraction requires at least one image")
    return np.stack([extractors[method](path) for path in metadata["image_path"]])


def extract_imagenet_features(
    metadata: pd.DataFrame,
    *,
    image_size: tuple[int, int] = (384, 384),
    batch_size: int = 64,
) -> np.ndarray:
    model = build_imagenet_embedding_model(
        "efficientnetv2b3",
        input_shape=(image_size[0], image_size[1], 3),
    )
    dataset = dataframe_to_dataset(
        metadata,
        "efficientnetv2b3",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=1,
        deterministic=True,
    )
    return np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)


def _distance_block(query: np.ndarray, train: np.ndarray, metric: str) -> np.ndarray:
    if metric == "chi2":
        numerator = (query[:, None, :] - train[None, :, :]) ** 2
        denominator = query[:, None, :] + train[None, :, :] + 1e-12
        return 0.5 * np.sum(numerator / denominator, axis=2)
    return cdist(query, train, metric=metric)


def deterministic_one_nearest_neighbor(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sample_ids: list[str],
    test_features: np.ndarray,
    *,
    metric: str,
    chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=int)
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("1-NN features must be two-dimensional")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test feature dimensions must match")
    if len(train_features) != len(train_labels) or len(train_features) != len(train_sample_ids):
        raise ValueError("training features, labels and sample IDs must align")
    if not len(train_features) or not len(test_features):
        raise ValueError("1-NN requires non-empty train and test features")

    order = np.argsort(np.asarray(train_sample_ids, dtype=str), kind="mergesort")
    ordered_features = train_features[order]
    ordered_labels = train_labels[order]
    ordered_ids = np.asarray(train_sample_ids, dtype=str)[order]
    predictions: list[int] = []
    distances: list[float] = []
    neighbours: list[str] = []
    for start in range(0, len(test_features), chunk_size):
        block = _distance_block(test_features[start : start + chunk_size], ordered_features, metric)
        nearest = np.argmin(block, axis=1)
        predictions.extend(ordered_labels[nearest].tolist())
        distances.extend(block[np.arange(len(nearest)), nearest].astype(float).tolist())
        neighbours.extend(ordered_ids[nearest].tolist())
    return np.asarray(predictions, dtype=int), np.asarray(distances), neighbours


def _method_metric(method: str) -> str:
    return {
        "phash": "hamming",
        "pixels32": "euclidean",
        "hsv_histogram": "chi2",
        "hog": "cosine",
        "imagenet_efficientnetv2b3": "cosine",
    }[method]


def run_shortcut_method(
    metadata: pd.DataFrame,
    *,
    method: str,
    bootstrap_resamples: int = 2000,
    feature_extractor: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if method not in FEATURE_METHODS:
        raise ValueError(f"Unknown shortcut method: {method}")
    required = {"image_path", "class_id", "split", "sample_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"shortcut metadata missing columns: {', '.join(missing)}")
    train = metadata[metadata["split"] == "train"].sort_values("sample_id").reset_index(drop=True)
    test = metadata[metadata["split"] == "test"].sort_values("sample_id").reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("shortcut audit requires non-empty train and test splits")

    if feature_extractor is None:
        if method == "imagenet_efficientnetv2b3":
            feature_extractor = extract_imagenet_features
        else:
            feature_extractor = lambda frame: extract_handcrafted_features(frame, method)
    train_features = feature_extractor(train)
    test_features = feature_extractor(test)
    predicted, distances, neighbour_ids = deterministic_one_nearest_neighbor(
        train_features,
        train["class_id"].to_numpy(),
        train["sample_id"].astype(str).tolist(),
        test_features,
        metric=_method_metric(method),
    )
    y_true = test["class_id"].astype(int).to_numpy()
    interval = bootstrap_cluster_accuracy_ci(
        y_true,
        predicted,
        test["class_id"].astype(str),
        n_resamples=bootstrap_resamples,
        seed=1,
    )
    predictions = test.copy()
    predictions["predicted_class_id"] = predicted
    predictions["nearest_train_sample_id"] = neighbour_ids
    predictions["nearest_distance"] = distances
    predictions["correct"] = predicted == y_true
    result: dict[str, object] = {
        "method": method,
        "distance": _method_metric(method),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "raw_errors": int(np.sum(predicted != y_true)),
        "accuracy_ci_low": float(interval["ci_low"]),
        "accuracy_ci_high": float(interval["ci_high"]),
        "bootstrap_unit": "class_id",
        "bootstrap_resamples": int(bootstrap_resamples),
    }
    return result, predictions


def run_shortcut_audit(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    methods: tuple[str, ...] = FEATURE_METHODS,
    bootstrap_resamples: int = 2000,
) -> Path:
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        metrics, predictions = run_shortcut_method(
            metadata,
            method=method,
            bootstrap_resamples=bootstrap_resamples,
        )
        metrics["metadata_path"] = str(metadata_path)
        metrics["metadata_sha256"] = sha256_file(metadata_path)
        (output_dir / f"metrics_{method}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        predictions.to_csv(output_dir / f"predictions_{method}.csv", index=False)
        rows.append(metrics)
    summary = output_dir / "shortcut_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic source-image shortcut baselines.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", choices=FEATURE_METHODS, default=list(FEATURE_METHODS))
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    output = run_shortcut_audit(
        args.metadata,
        args.out,
        methods=tuple(args.methods),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
