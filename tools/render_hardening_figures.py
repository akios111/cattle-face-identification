from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_id.hardening_matrix import metadata_path_for_job, run_dir_for_job
from cattle_id.holstein_audit import (
    DEFAULT_SUFFIX as HOLSTEIN_SUFFIX,
    select_diverse_rank1_errors,
)
from cattle_id.run_matrix import expand_experiment_matrix, load_experiment_matrix
from cattle_id.augmentation import get_augmentation_specs


COLORS = {
    "green": "#167D5A",
    "red": "#B7413E",
    "blue": "#276FBF",
    "gold": "#C58B19",
    "gray": "#5D6670",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_pipeline(path: str | Path) -> Path:
    path = Path(path)
    labels = [
        "Pinned CattleSSFR\n311 source images",
        "hardening_v2\n6,220 byte-hashed PNGs",
        "Frozen splits and\nindependent seeds",
        "29 EfficientNetV2B3\ntrainings",
        "Shortcut, ablation and\nrobustness audits",
        "Holstein2025 controls\nand checkpoint audit",
        "Hash-verified evidence\nand thesis tables",
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, label in enumerate(labels):
        color = COLORS["green"] if index in {0, 1, 6} else COLORS["blue"]
        rectangle = plt.Rectangle(
            (index + 0.08, 0.28),
            0.82,
            0.44,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(rectangle)
        ax.text(index + 0.49, 0.50, label, ha="center", va="center", fontsize=8.5)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(index + 1.06, 0.50),
                xytext=(index + 0.91, 0.50),
                arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.5},
            )
    return _save(fig, path)


def render_augmentation_montage(metadata_path: str | Path, path: str | Path) -> Path:
    metadata = pd.read_csv(metadata_path)
    source_file = metadata["source_file"].astype(str).sort_values().iloc[0]
    selected = metadata[metadata["source_file"].astype(str) == source_file].copy()
    canonical_ids = [spec.identifier for spec in get_augmentation_specs("all")]
    selected["augmentation_id"] = pd.Categorical(
        selected["augmentation_id"], categories=canonical_ids, ordered=True
    )
    selected = selected.sort_values("augmentation_id")
    if len(selected) != 20:
        raise ValueError("augmentation montage requires all twenty canonical variants")
    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    for axis, (_, row) in zip(axes.flat, selected.iterrows(), strict=True):
        with Image.open(row["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(str(row["augmentation_id"]).replace("_", " "), fontsize=8)
        axis.axis("off")
    fig.suptitle("Canonical source image and nineteen deterministic transformations", fontsize=12)
    return _save(fig, Path(path))


def render_split_diagram(metadata_paths: dict[str, Path], path: str | Path) -> Path:
    rows = []
    for protocol, metadata_path in metadata_paths.items():
        counts = pd.read_csv(metadata_path)["split"].value_counts()
        for split in ("train", "validation", "test"):
            rows.append({"protocol": protocol, "split": split, "samples": int(counts.get(split, 0))})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="protocol", columns="split", values="samples").fillna(0)
    pivot = pivot[["train", "validation", "test"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    pivot.plot(
        kind="barh",
        stacked=True,
        color=[COLORS["green"], COLORS["gold"], COLORS["red"]],
        ax=ax,
    )
    ax.set_xlabel("Images")
    ax.set_ylabel("")
    ax.legend(title="Split", frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, Path(path))


def render_shortcut_comparison(tables_dir: Path, path: Path) -> Path:
    frame = pd.read_csv(tables_dir / "hardening_shortcut_summary.csv")
    protocols = list(frame["protocol"].drop_duplicates())
    fig, axes = plt.subplots(1, len(protocols), figsize=(12, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, protocol in zip(axes, protocols, strict=True):
        selected = frame[frame["protocol"] == protocol].sort_values("accuracy")
        axis.barh(selected["method"], selected["accuracy"], color=COLORS["blue"])
        axis.set_title(protocol.replace("_hardening_v2", "").replace("_", " "))
        axis.set_xlim(0, 1.0)
        axis.grid(axis="x", alpha=0.2)
        axis.set_xlabel("Accuracy")
    return _save(fig, path)


def render_learning_curves(run_dirs: list[Path], path: Path) -> Path:
    rows = []
    for run_dir in run_dirs:
        history = pd.read_csv(run_dir / "history.csv")
        history["epoch"] = np.arange(1, len(history) + 1)
        history["run_id"] = run_dir.name
        rows.append(history)
    frame = pd.concat(rows, ignore_index=True)
    required = {"accuracy", "val_accuracy", "loss", "val_loss"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training histories missing columns: {missing}")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, train_metric, validation_metric, title in (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Loss"),
    ):
        for metric, color, label in (
            (train_metric, COLORS["blue"], "Training"),
            (validation_metric, COLORS["red"], "Validation"),
        ):
            grouped = frame.groupby("epoch")[metric]
            mean = grouped.mean()
            std = grouped.std(ddof=1).fillna(0)
            axis.plot(mean.index, mean.values, color=color, label=label)
            axis.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.16)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"Five-seed {title.lower()} curve")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    return _save(fig, path)


def render_robustness(tables_dir: Path, path: Path) -> Path:
    region = pd.read_csv(tables_dir / "hardening_region_audit_summary.csv")
    severity = pd.read_csv(tables_dir / "hardening_severity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ordered_region = region.sort_values("accuracy_mean")
    axes[0].barh(
        ordered_region["region_variant"],
        ordered_region["accuracy_mean"],
        xerr=ordered_region["accuracy_sample_std"],
        color=COLORS["green"],
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Mean accuracy across five seeds")
    axes[0].set_title("Geometric image-region controls")
    for family, group in severity.groupby("severity_family"):
        group = group.sort_values("severity_value")
        axes[1].plot(group["severity_value"], group["accuracy_mean"], marker="o", label=family)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Transformation severity")
    axes[1].set_ylabel("Mean accuracy across five seeds")
    axes[1].set_title("Same-source severity sweep")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, path)


def render_holstein_curves(
    tables_dir: Path,
    runs_dir: Path,
    path: Path,
) -> Path:
    runs = pd.read_csv(tables_dir / "hardening_holstein_runs.csv")
    curve_rows = []
    map_rows = []
    for row in runs.to_dict(orient="records"):
        predictions_path = runs_dir / row["run_id"] / f"predictions_{HOLSTEIN_SUFFIX}.csv"
        predictions = pd.read_csv(predictions_path)
        maximum_rank = int(predictions["first_correct_rank"].max())
        for rank in range(1, maximum_rank + 1):
            curve_rows.append(
                {
                    "control_type": row["control_type"],
                    "source_protocol": row["source_protocol"],
                    "rank": rank,
                    "cmc": float(np.mean(predictions["first_correct_rank"] <= rank)),
                }
            )
        per_identity = predictions.groupby("animal_id")["average_precision"].mean()
        for value in per_identity:
            map_rows.append({"control_type": row["control_type"], "mAP": float(value)})
    curves = pd.DataFrame(curve_rows)
    maps = pd.DataFrame(map_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for (control_type, source_protocol), group in curves.groupby(
        ["control_type", "source_protocol"]
    ):
        mean_curve = group.groupby("rank")["cmc"].mean()
        axes[0].plot(
            mean_curve.index,
            mean_curve.values,
            label=f"{control_type}: {source_protocol.replace('_hardening_v2', '')}",
        )
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("CMC")
    axes[0].set_ylim(0, 1.01)
    axes[0].set_title("Holstein2025 cumulative matching curve")
    axes[0].legend(frameon=False, fontsize=7)
    groups = [group["mAP"].to_numpy() for _, group in maps.groupby("control_type")]
    labels = [name for name, _ in maps.groupby("control_type")]
    axes[1].boxplot(groups, tick_labels=labels, showfliers=False)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_ylabel("Per-identity mAP")
    axes[1].set_title("Holstein2025 identity-level distribution")
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, path)


def render_checkpoint_heatmap(tables_dir: Path, path: Path) -> Path:
    frame = pd.read_csv(tables_dir / "hardening_holstein_checkpoint_pairwise.csv")
    labels = sorted(set(frame["run_a"]) | set(frame["run_b"]))
    matrix = np.eye(len(labels), dtype=float)
    index = {label: position for position, label in enumerate(labels)}
    for row in frame.to_dict(orient="records"):
        left, right = index[row["run_a"]], index[row["run_b"]]
        matrix[left, right] = matrix[right, left] = float(row["correct_set_jaccard"])
    def short_label(label: str) -> str:
        prefix = "PR" if "paper_random" in label else "TH"
        match = re.search(r"_train(\d+)_", label)
        if match is None:
            raise ValueError(f"cannot infer checkpoint seed from run ID: {label}")
        return f"{prefix}{match.group(1)}"

    short = [short_label(label) for label in labels]
    off_diagonal = matrix[~np.eye(len(labels), dtype=bool)]
    lower = max(0.0, float(np.floor((off_diagonal.min() - 0.01) * 100.0) / 100.0))
    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, vmin=lower, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=short, fontsize=8)
    ax.set_yticks(range(len(labels)), labels=short, fontsize=8)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            color = "white" if value < (lower + 1.0) / 2.0 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    ax.set_title("Jaccard overlap of correctly retrieved Holstein2025 probes", fontsize=10)
    ax.set_xlabel("PR: random split; TH: transform-family holdout; number: training seed", fontsize=8)
    fig.colorbar(image, ax=ax, label="Jaccard overlap", shrink=0.84, pad=0.03)
    return _save(fig, path)


def render_error_gallery(
    runs_dir: Path,
    run_id: str,
    path: Path,
    limit: int = 8,
    holstein_root: Path = Path("/content/cattle_runtime/raw/Holstein2025"),
) -> Path:
    predictions = pd.read_csv(runs_dir / run_id / f"predictions_{HOLSTEIN_SUFFIX}.csv")
    errors = select_diverse_rank1_errors(predictions, limit=limit)
    if errors.empty:
        raise ValueError("Holstein error gallery requires at least one rank-1 error")
    columns = 4
    rows = int(np.ceil(len(errors) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, (_, row) in zip(axes, errors.iterrows()):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file() and "relative_path" in row:
            image_path = holstein_root / str(row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Holstein2025 error-gallery image not found: {image_path}")
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"True: {row['animal_id']}\nTop-1: {row['predicted_animal_id']}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes[len(errors) :]:
        axis.axis("off")
    return _save(fig, path)


def _load_gradcam_heatmap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        heatmap = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if float(heatmap.max()) <= 0.0 or int(np.count_nonzero(heatmap)) == 0:
        raise ValueError(f"blank Grad-CAM heatmap: {path}")
    return heatmap


def render_gradcam_triptych(run_dir: Path, path: Path, limit: int = 6) -> Path:
    samples = pd.read_csv(run_dir / "gradcam" / "gradcam_samples.csv").head(limit)
    if samples.empty:
        raise ValueError("Grad-CAM figure requires saved samples")

    def artifact_path(value: object) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    for row in samples.to_dict(orient="records"):
        _load_gradcam_heatmap(artifact_path(row["heatmap_path"]))

    correct = samples["class_id"].astype(str) == samples["predicted_class_id"].astype(str)
    chosen: list[int] = []
    preferred_groups = [
        samples.index[correct & (samples["augmentation_id"].astype(str) == "original")],
        samples.index[~correct],
        samples.index[correct],
    ]
    for group in preferred_groups:
        for index in group:
            if int(index) not in chosen:
                chosen.append(int(index))
                break
    for index in samples.index:
        if int(index) not in chosen:
            chosen.append(int(index))
        if len(chosen) >= 3:
            break
    representatives = samples.loc[chosen[:3]].reset_index(drop=True)

    fig = plt.figure(figsize=(10.5, 3.0 * len(representatives) + 3.2))
    grid = fig.add_gridspec(
        len(representatives) + 1,
        3,
        height_ratios=[1.0] * len(representatives) + [1.15],
    )
    image_axes: list[list[plt.Axes]] = []
    for row_index, (_, row) in enumerate(representatives.iterrows()):
        row_axes = [fig.add_subplot(grid[row_index, column]) for column in range(3)]
        image_axes.append(row_axes)
        with Image.open(artifact_path(row["original_path"])) as image:
            row_axes[0].imshow(image.convert("RGB"))
        row_axes[1].imshow(
            _load_gradcam_heatmap(artifact_path(row["heatmap_path"])),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        with Image.open(artifact_path(row["overlay_path"])) as image:
            row_axes[2].imshow(image.convert("RGB"))
        prediction_ok = str(row["class_id"]) == str(row["predicted_class_id"])
        row_axes[0].set_ylabel(
            f"class {row['class_id']}\n{row['augmentation_id']}\n"
            f"{'correct' if prediction_ok else 'incorrect'}",
            fontsize=8,
        )
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(image_axes[0], ("Original", "Grad-CAM", "Overlay"), strict=True):
        axis.set_title(title)

    curves = pd.read_csv(run_dir / "gradcam" / "gradcam_faithfulness_curves.csv")
    summary = (
        curves.groupby(["curve", "fraction"])["target_probability"]
        .agg(["mean", "std"])
        .reset_index()
    )
    curve_axis = fig.add_subplot(grid[-1, :])
    styles = {
        "deletion_gradcam": (COLORS["red"], "-", "Deletion: Grad-CAM"),
        "deletion_random": (COLORS["red"], "--", "Deletion: random"),
        "insertion_gradcam": (COLORS["blue"], "-", "Insertion: Grad-CAM"),
        "insertion_random": (COLORS["blue"], "--", "Insertion: random"),
    }
    for curve, (color, linestyle, label) in styles.items():
        group = summary[summary["curve"] == curve].sort_values("fraction")
        if group.empty:
            raise ValueError(f"missing Grad-CAM faithfulness curve: {curve}")
        x = group["fraction"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        std = group["std"].fillna(0.0).to_numpy(dtype=float)
        curve_axis.plot(x, mean, color=color, linestyle=linestyle, linewidth=2, label=label)
        curve_axis.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color,
            alpha=0.08,
        )
    curve_axis.set_title(f"Faithfulness curves across {len(samples)} samples")
    curve_axis.set_xlabel("Fraction of pixels removed or inserted")
    curve_axis.set_ylabel("Predicted-class probability")
    curve_axis.set_xlim(0, 1)
    curve_axis.set_ylim(0, 1.02)
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, path)


def render_all(
    *,
    matrix_path: str | Path = "configs/experiment_matrix_hardening_v2.yaml",
    tables_dir: str | Path = "thesis/tables/hardening_v2",
    output_dir: str | Path = "thesis/figures/hardening_v2",
    runs_dir: str | Path = "artifacts/runs",
    holstein_root: str | Path = "/content/cattle_runtime/raw/Holstein2025",
) -> list[Path]:
    jobs = expand_experiment_matrix(load_experiment_matrix(matrix_path))
    paper_jobs = sorted(
        [
            job
            for job in jobs
            if job["protocol"] == "paper_random_hardening_v2" and int(job["split_seed"]) == 1
        ],
        key=lambda job: int(job["training_seed"]),
    )
    transform_job = next(
        job
        for job in jobs
        if job["protocol"] == "transform_holdout_hardening_v2"
        and int(job["training_seed"]) == 1
        and int(job["split_seed"]) == 1
    )
    tables_dir = Path(tables_dir)
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    holstein_root = Path(holstein_root)
    paper_metadata = metadata_path_for_job(paper_jobs[0])
    transform_metadata = metadata_path_for_job(transform_job)
    first_paper_run = runs_dir / run_dir_for_job(paper_jobs[0]).name
    holstein_runs = pd.read_csv(tables_dir / "hardening_holstein_runs.csv")
    error_candidates = holstein_runs[
        holstein_runs["control_type"] == "fine_tuned"
    ].sort_values(["source_protocol", "training_seed", "run_id"])
    error_run = error_candidates.iloc[0]["run_id"]
    outputs = [
        render_pipeline(output_dir / "pipeline.png"),
        render_augmentation_montage(paper_metadata, output_dir / "augmentation_montage.png"),
        render_split_diagram(
            {"paper random": paper_metadata, "transform holdout": transform_metadata},
            output_dir / "split_protocols.png",
        ),
        render_shortcut_comparison(tables_dir, output_dir / "shortcut_comparison.png"),
        render_learning_curves(
            [runs_dir / run_dir_for_job(job).name for job in paper_jobs],
            output_dir / "learning_curves.png",
        ),
        render_robustness(tables_dir, output_dir / "robustness_controls.png"),
        render_holstein_curves(tables_dir, runs_dir, output_dir / "holstein_curves.png"),
        render_checkpoint_heatmap(tables_dir, output_dir / "checkpoint_overlap.png"),
        render_error_gallery(
            runs_dir,
            str(error_run),
            output_dir / "holstein_error_gallery.png",
            holstein_root=holstein_root,
        ),
        render_gradcam_triptych(first_paper_run, output_dir / "gradcam_triptych.png"),
    ]
    manifest = {
        "figures": [str(path) for path in outputs],
        "count": len(outputs),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render hardening_v2 computer-vision figures.")
    parser.add_argument("--matrix", default="configs/experiment_matrix_hardening_v2.yaml")
    parser.add_argument("--tables", default="thesis/tables/hardening_v2")
    parser.add_argument("--out", default="thesis/figures/hardening_v2")
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument(
        "--holstein-root", default="/content/cattle_runtime/raw/Holstein2025"
    )
    args = parser.parse_args(argv)
    outputs = render_all(
        matrix_path=args.matrix,
        tables_dir=args.tables,
        output_dir=args.out,
        runs_dir=args.runs,
        holstein_root=args.holstein_root,
    )
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
