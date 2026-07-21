from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_id.augmentation import get_augmentation_specs
from cattle_id.config import load_config


def _latex(value: object) -> str:
    return str(value).replace("_", r"\_\allowbreak{}").replace("%", r"\%")


def render_settings(
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    config = load_config(config_path)
    training = config["training"]
    fine_tune = training["fine_tune"]
    rows = []
    for spec in get_augmentation_specs("all"):
        parameters = ", ".join(f"{key}={value}" for key, value in spec.params.items()) or "--"
        rows.append(("Μετασχηματισμός", spec.identifier, spec.family, parameters))
    rows.extend(
        [
            ("Δεδομένα", "input_size", "canonical", "384 x 384"),
            ("Δεδομένα", "interpolation", "loader", "bilinear"),
            ("Δεδομένα", "fill_mode", "geometric", "constant black"),
            ("Seeds", "augmentation_seed", "materialization", config["augmentation"]["seed"]),
            ("Seeds", "split_seed", "fixed/repeated", config["split"]["seed"]),
            ("Seeds", "training_seed", "model", training["training_seed"]),
            ("Εκπαίδευση", "optimizer_head", "head", training["optimizer"]),
            ("Εκπαίδευση", "learning_rate_head", "head", training["learning_rate"]),
            ("Εκπαίδευση", "epochs_head", "head", training["epochs"]),
            ("Εκπαίδευση", "optimizer_finetune", "fine-tune", training["optimizer"]),
            ("Εκπαίδευση", "learning_rate_finetune", "fine-tune", fine_tune["learning_rate"]),
            ("Εκπαίδευση", "epochs_finetune", "fine-tune", fine_tune["epochs"]),
            ("Εκπαίδευση", "weight_decay", "all stages", training["weight_decay"]),
            ("Εκπαίδευση", "dropout", "classifier", training["dropout"]),
            ("Εκπαίδευση", "batch_size", "all stages", training["batch_size"]),
            (
                "Εκπαίδευση",
                "early_stopping_patience",
                "validation loss",
                training["early_stopping"]["patience"],
            ),
            ("Εκπαίδευση", "checkpoint_metric", "selection", training["checkpoint_metric"]),
            ("Fine-tuning", "trainable_last_n", "backbone", fine_tune["trainable_last_n"]),
            ("Fine-tuning", "freeze_batchnorm", "backbone", fine_tune["freeze_batchnorm"]),
        ]
    )
    lines = [
        r"\small",
        r"\begin{longtable}{@{}p{0.22\textwidth}p{0.23\textwidth}p{0.16\textwidth}p{0.25\textwidth}@{}}",
        r"\caption{Πλήρες συμβόλαιο μετασχηματισμών και εκπαίδευσης του \texttt{hardening\_v2}}\label{tab:hardening-settings}\\",
        r"\toprule",
        r"Κατηγορία & Ρύθμιση & Πεδίο & Τιμή \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Κατηγορία & Ρύθμιση & Πεδίο & Τιμή \\",
        r"\midrule",
        r"\endhead",
    ]
    for category, name, field, value in rows:
        lines.append(
            f"{_latex(category)} & \\texttt{{{_latex(name)}}} & {_latex(field)} & {_latex(value)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the configuration-derived thesis settings table.")
    parser.add_argument("--config", default="configs/cattlessfr_hardening_v2_colab_proplus.yaml")
    parser.add_argument("--out", default="thesis/tables/hardening_v2/experiment_settings_compact.tex")
    args = parser.parse_args(argv)
    print(render_settings(args.config, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
