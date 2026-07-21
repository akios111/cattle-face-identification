from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .logging_utils import log_line


SUMMARY_COLUMNS = [
    "model",
    "protocol",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "top_5_accuracy",
    "seconds_per_image",
    "parameter_count",
    "model_size_bytes",
]


def collect_run_metrics(runs_dir: str | Path) -> list[dict[str, object]]:
    runs_dir = Path(runs_dir)
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["run_id"] = metrics_path.parent.name
        rows.append(row)
    return rows


def _format_latex_table(frame: pd.DataFrame) -> str:
    return frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}")


def write_summary_tables(rows: Iterable[dict[str, object]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        frame = pd.DataFrame(columns=SUMMARY_COLUMNS)
    selected = [column for column in SUMMARY_COLUMNS if column in frame.columns]
    frame = frame[selected]
    csv_path = output_dir / "model_comparison.csv"
    latex_path = output_dir / "model_comparison.tex"
    frame.to_csv(csv_path, index=False)
    latex_path.write_text(_format_latex_table(frame), encoding="utf-8")
    return {"csv": csv_path, "latex": latex_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate CattleSSFR run metrics.")
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--out", default="thesis/tables")
    args = parser.parse_args(argv)
    log_path = Path(args.out) / "report.log"
    log_line(log_path, "report", f"collecting runs={args.runs}")
    rows = collect_run_metrics(args.runs)
    log_line(log_path, "report", f"metrics_files={len(rows)}")
    written = write_summary_tables(rows, args.out)
    log_line(log_path, "report", f"csv_written={written['csv']}")
    log_line(log_path, "report", f"latex_written={written['latex']}")
    print(written["csv"])
    print(written["latex"])


if __name__ == "__main__":
    main()
