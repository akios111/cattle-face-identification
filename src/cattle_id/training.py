from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from .logging_utils import append_event, log_line
from .models import set_backbone_trainable


@dataclass(frozen=True)
class TrainingStage:
    name: str
    epochs: int
    learning_rate: float
    trainable_backbone: bool
    trainable_last_n: int | None = None
    freeze_batchnorm: bool = True


class RunEventLogger(tf.keras.callbacks.Callback):
    def __init__(self, run_dir: str | Path, stage: TrainingStage) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.log_path = self.run_dir / "run.log"

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        log_line(
            self.log_path,
            "train",
            f"stage={self.stage.name} train_begin epochs={self.stage.epochs}",
        )
        append_event(
            self.run_dir,
            "stage_train_begin",
            stage=self.stage.name,
            epochs=self.stage.epochs,
            learning_rate=self.stage.learning_rate,
            trainable_backbone=self.stage.trainable_backbone,
            trainable_last_n=self.stage.trainable_last_n,
        )

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        log_line(
            self.log_path,
            "train",
            f"stage={self.stage.name} epoch={epoch + 1}/{self.stage.epochs} begin",
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        metrics = {key: float(value) for key, value in (logs or {}).items()}
        metric_text = " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        log_line(
            self.log_path,
            "train",
            f"stage={self.stage.name} epoch={epoch + 1}/{self.stage.epochs} end {metric_text}".strip(),
        )
        append_event(
            self.run_dir,
            "epoch_end",
            stage=self.stage.name,
            epoch=epoch + 1,
            metrics=metrics,
        )

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        log_line(self.log_path, "train", f"stage={self.stage.name} train_end")
        append_event(self.run_dir, "stage_train_end", stage=self.stage.name)


def build_training_stages(training_cfg: dict[str, Any], epochs_override: int | None = None) -> list[TrainingStage]:
    head_epochs = int(epochs_override or training_cfg.get("epochs", 100))
    stages = [
        TrainingStage(
            name="head",
            epochs=head_epochs,
            learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
            trainable_backbone=False,
        )
    ]
    fine_tune_cfg = training_cfg.get("fine_tune", {})
    if bool(fine_tune_cfg.get("enabled", False)) and epochs_override is None:
        stages.append(
            TrainingStage(
                name="finetune",
                epochs=int(fine_tune_cfg.get("epochs", 30)),
                learning_rate=float(fine_tune_cfg.get("learning_rate", 1e-5)),
                trainable_backbone=True,
                trainable_last_n=fine_tune_cfg.get("trainable_last_n"),
                freeze_batchnorm=bool(fine_tune_cfg.get("freeze_batchnorm", True)),
            )
        )
    return stages


def build_optimizer(training_cfg: dict[str, Any], learning_rate: float) -> tf.keras.optimizers.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def compile_for_stage(
    model: tf.keras.Model,
    training_cfg: dict[str, Any],
    stage: TrainingStage,
) -> None:
    set_backbone_trainable(
        model,
        trainable=stage.trainable_backbone,
        trainable_last_n=stage.trainable_last_n,
        freeze_batchnorm=stage.freeze_batchnorm,
    )
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    compile_kwargs: dict[str, Any] = {
        "optimizer": build_optimizer(training_cfg, stage.learning_rate),
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
        "metrics": [
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    }
    if "steps_per_execution" in training_cfg:
        compile_kwargs["steps_per_execution"] = int(training_cfg["steps_per_execution"])
    if "jit_compile" in training_cfg:
        compile_kwargs["jit_compile"] = bool(training_cfg["jit_compile"])
    model.compile(**compile_kwargs)
    if label_smoothing:
        # SparseCategoricalCrossentropy has no label_smoothing argument in some TF builds.
        # Keep this explicit so configs do not silently claim unsupported smoothing.
        raise ValueError("label_smoothing is not supported with sparse labels in this pipeline")


def build_callbacks(run_dir: str | Path, stage: TrainingStage, training_cfg: dict[str, Any]) -> list[tf.keras.callbacks.Callback]:
    run_dir = Path(run_dir)
    checkpoint_metric = str(training_cfg.get("checkpoint_metric", "val_accuracy"))
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=str(training_cfg.get("lr_monitor", "val_loss")),
            factor=float(training_cfg.get("lr_factor", 0.2)),
            patience=int(training_cfg.get("lr_patience", 5)),
            min_lr=float(training_cfg.get("min_lr", 1e-6)),
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "model.keras"),
            monitor=checkpoint_metric,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"history_{stage.name}.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard" / stage.name)),
        RunEventLogger(run_dir, stage),
    ]
    early_stopping = training_cfg.get("early_stopping", {})
    if bool(early_stopping.get("enabled", True)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=str(early_stopping.get("monitor", "val_loss")),
                patience=int(early_stopping.get("patience", 12)),
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def fit_stages(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    validation_ds: tf.data.Dataset,
    stages: list[TrainingStage],
    run_dir: str | Path,
    training_cfg: dict[str, Any],
) -> pd.DataFrame:
    histories = []
    run_dir = Path(run_dir)
    for stage in stages:
        log_line(
            run_dir / "run.log",
            "train",
            (
                f"stage={stage.name} compile "
                f"epochs={stage.epochs} lr={stage.learning_rate} "
                f"trainable_backbone={stage.trainable_backbone} "
                f"trainable_last_n={stage.trainable_last_n}"
            ),
        )
        append_event(run_dir, "stage_compile", stage=stage.__dict__)
        compile_for_stage(model, training_cfg, stage)
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=stage.epochs,
            callbacks=build_callbacks(run_dir, stage, training_cfg),
        )
        frame = pd.DataFrame(history.history)
        frame.insert(0, "stage", stage.name)
        frame.insert(1, "stage_epoch", range(1, len(frame) + 1))
        histories.append(frame)
    return pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
