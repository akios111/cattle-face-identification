from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "external_public_face_benchmark.yaml"
DEFAULT_GATES_DIR = PROJECT_ROOT / "thesis" / "gates"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "thesis" / "tables"
EXPECTED_PROTOCOL = "external_public_face_benchmark"
EXPECTED_SPLITS = ("train", "validation", "test")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _required_columns() -> set[str]:
    return {
        "image_path",
        "class_id",
        "animal_id",
        "external_dataset",
        "protocol",
        "split",
        "source_url",
    }


def collect_summary(
    *,
    metadata_path: str | Path,
    expected_animals: int,
    expected_images_per_animal: int,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    frame = pd.read_csv(metadata_path)
    missing_columns = sorted(_required_columns() - set(frame.columns))
    issues: list[str] = []
    if missing_columns:
        issues.append(f"metadata missing required columns: {', '.join(missing_columns)}")
        return {
            "ready": False,
            "metadata_path": str(metadata_path),
            "animals": 0,
            "images": int(len(frame)),
            "min_images_per_animal": 0,
            "max_images_per_animal": 0,
            "mean_images_per_animal": 0.0,
            "split_counts": {},
            "source_url": "",
            "issues": issues,
        }

    per_animal = frame.groupby("animal_id", sort=True)["image_path"].count()
    split_counts = {
        split: int(count)
        for split, count in frame["split"].value_counts().sort_index().to_dict().items()
    }
    animals = int(per_animal.shape[0])
    images = int(frame.shape[0])
    min_images = int(per_animal.min()) if animals else 0
    max_images = int(per_animal.max()) if animals else 0
    mean_images = float(per_animal.mean()) if animals else 0.0
    protocols = set(frame["protocol"].astype(str))
    missing_splits = [split for split in EXPECTED_SPLITS if split_counts.get(split, 0) == 0]
    blank_paths = int(frame["image_path"].isna().sum() + (frame["image_path"].astype(str).str.strip() == "").sum())
    duplicate_paths = int(frame["image_path"].duplicated().sum())

    if animals < expected_animals:
        issues.append(f"animal count below threshold: {animals} < {expected_animals}")
    if min_images < expected_images_per_animal:
        issues.append(
            f"images per animal below threshold: {min_images} < {expected_images_per_animal}"
        )
    if protocols != {EXPECTED_PROTOCOL}:
        issues.append(f"unexpected protocol values: {sorted(protocols)}")
    if missing_splits:
        issues.append(f"missing split assignments: {', '.join(missing_splits)}")
    if blank_paths:
        issues.append(f"blank image paths: {blank_paths}")
    if duplicate_paths:
        issues.append(f"duplicate image paths: {duplicate_paths}")

    source_values = [value for value in frame["source_url"].dropna().astype(str).unique() if value]
    return {
        "ready": not issues,
        "metadata_path": str(metadata_path),
        "animals": animals,
        "images": images,
        "min_images_per_animal": min_images,
        "max_images_per_animal": max_images,
        "mean_images_per_animal": mean_images,
        "split_counts": split_counts,
        "source_url": source_values[0] if source_values else "",
        "issues": issues,
    }


def collect_from_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    outputs = config["outputs"]
    summary = collect_summary(
        metadata_path=_project_path(outputs["metadata"]),
        expected_animals=int(dataset["expected_animals"]),
        expected_images_per_animal=int(dataset["expected_images_per_animal"]),
    )
    summary["config_path"] = str(config_path)
    summary["dataset_name"] = dataset.get("name", "unknown")
    summary["expected_animals"] = int(dataset["expected_animals"])
    summary["expected_images_per_animal"] = int(dataset["expected_images_per_animal"])
    summary["hash_manifest"] = str(_project_path(outputs["hash_manifest"]))
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    split_counts = summary.get("split_counts", {})
    split_keys = list(EXPECTED_SPLITS) + sorted(key for key in split_counts if key not in EXPECTED_SPLITS)
    split_text = ", ".join(f"{key}={split_counts.get(key, 0)}" for key in split_keys)
    issue_lines = "\n".join(f"- {issue}" for issue in summary["issues"]) if summary["issues"] else "- none"
    return (
        "# External Public Benchmark Readiness\n\n"
        f"External public benchmark ready {summary['ready']}.\n\n"
        "This gate covers the public Cattely face-image benchmark as a separate "
        "`external_public_face_benchmark` protocol. It is evidence for cross-dataset "
        "public benchmarking, not a same-identity external acquisition holdout for CattleSSFR.\n\n"
        f"- Metadata: `{summary['metadata_path']}`\n"
        f"- Source URL: {summary.get('source_url', '')}\n"
        f"- Animals: {summary['animals']}\n"
        f"- Images: {summary['images']}\n"
        f"- Images per animal: min={summary['min_images_per_animal']}, "
        f"mean={summary['mean_images_per_animal']:.2f}, max={summary['max_images_per_animal']}\n"
        f"- Split counts: {split_text}\n\n"
        "## Issues\n\n"
        f"{issue_lines}\n"
    )


def _summary_row(summary: dict[str, Any]) -> dict[str, object]:
    split_counts = summary.get("split_counts", {})
    return {
        "protocol": EXPECTED_PROTOCOL,
        "ready": summary["ready"],
        "animals": summary["animals"],
        "images": summary["images"],
        "min_images_per_animal": summary["min_images_per_animal"],
        "mean_images_per_animal": f"{summary['mean_images_per_animal']:.2f}",
        "max_images_per_animal": summary["max_images_per_animal"],
        "train_images": split_counts.get("train", 0),
        "validation_images": split_counts.get("validation", 0),
        "test_images": split_counts.get("test", 0),
        "excluded_low_image_count": split_counts.get("excluded_low_image_count", 0),
        "issues": "; ".join(summary["issues"]) if summary["issues"] else "none",
    }


def _write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_tex(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Protocol",
        "Ready",
        "Animals",
        "Images",
        "Min/img",
        "Mean/img",
        "Train",
        "Val.",
        "Test",
        "Excluded",
    ]
    values = [
        row["protocol"],
        row["ready"],
        row["animals"],
        row["images"],
        row["min_images_per_animal"],
        row["mean_images_per_animal"],
        row["train_images"],
        row["validation_images"],
        row["test_images"],
        row["excluded_low_image_count"],
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Σύνοψη εξωτερικού δημόσιου benchmark Cattely.}",
        r"\label{tab:external-public-benchmark-summary}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        " & ".join(latex_escape(value) for value in values) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    summary: dict[str, Any],
    *,
    gates_dir: str | Path = DEFAULT_GATES_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
) -> list[Path]:
    gates_dir = Path(gates_dir)
    tables_dir = Path(tables_dir)
    markdown_path = gates_dir / "external-public-benchmark-readiness.md"
    csv_path = tables_dir / "external_public_benchmark_summary.csv"
    tex_path = tables_dir / "external_public_benchmark_summary.tex"

    gates_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    row = _summary_row(summary)
    _write_csv(csv_path, row)
    _write_tex(tex_path, row)
    return [markdown_path, csv_path, tex_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gates-dir", type=Path, default=DEFAULT_GATES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_from_config(args.config)
    generated = write_summary(summary, gates_dir=args.gates_dir, tables_dir=args.tables_dir)
    print(f"External public benchmark ready {summary['ready']}.")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
