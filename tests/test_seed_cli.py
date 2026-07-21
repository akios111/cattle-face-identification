from __future__ import annotations

from pathlib import Path

from cattle_id import data, train


def test_prepare_cli_passes_seed_override(monkeypatch, capsys):
    calls = []

    def fake_prepare(config_path, protocol=None, seed=None):
        calls.append({"config": config_path, "protocol": protocol, "seed": seed})
        return Path("metadata.csv")

    monkeypatch.setattr(data, "prepare_from_config", fake_prepare)

    data.main(["--config", "configs/cattlessfr_colab_proplus.yaml", "--protocol", "paper_random", "--seed", "5"])

    assert calls == [
        {
            "config": "configs/cattlessfr_colab_proplus.yaml",
            "protocol": "paper_random",
            "seed": 5,
        }
    ]
    assert "metadata.csv" in capsys.readouterr().out


def test_prepare_cli_passes_independent_hardening_seeds(monkeypatch):
    calls = []

    def fake_prepare(config_path, **kwargs):
        calls.append({"config": config_path, **kwargs})
        return Path("metadata.csv")

    monkeypatch.setattr(data, "prepare_from_config", fake_prepare)

    data.main(
        [
            "--config",
            "configs/cattlessfr_hardening_v2_colab_proplus.yaml",
            "--protocol",
            "paper_random_hardening_v2",
            "--training-seed",
            "5",
            "--split-seed",
            "3",
            "--augmentation-seed",
            "1",
        ]
    )

    assert calls == [
        {
            "config": "configs/cattlessfr_hardening_v2_colab_proplus.yaml",
            "protocol": "paper_random_hardening_v2",
            "seed": None,
            "training_seed": 5,
            "split_seed": 3,
            "augmentation_seed": 1,
        }
    ]


def test_train_cli_passes_seed_override(monkeypatch, capsys):
    calls = []

    def fake_train(
        config_path,
        model_name,
        protocol,
        epochs=None,
        weights="imagenet",
        seed=None,
        metadata_path=None,
        run_id=None,
        skip_completed=False,
    ):
        calls.append(
            {
                "config": config_path,
                "model": model_name,
                "protocol": protocol,
                "epochs": epochs,
                "weights": weights,
                "seed": seed,
                "metadata_path": metadata_path,
                "run_id": run_id,
                "skip_completed": skip_completed,
            }
        )
        return Path("artifacts/runs/run_seed5")

    monkeypatch.setattr(train, "train_from_config", fake_train)

    train.main(
        [
            "--config",
            "configs/cattlessfr_colab_proplus.yaml",
            "--model",
            "efficientnetb0",
            "--protocol",
            "paper_random",
            "--seed",
            "5",
        ]
    )

    assert calls == [
        {
            "config": "configs/cattlessfr_colab_proplus.yaml",
            "model": "efficientnetb0",
            "protocol": "paper_random",
            "epochs": None,
            "weights": "imagenet",
            "seed": 5,
            "metadata_path": None,
            "run_id": None,
            "skip_completed": False,
        }
    ]
    output = capsys.readouterr().out.replace("\\", "/")
    assert "artifacts/runs/run_seed5" in output


def test_train_cli_passes_resumable_matrix_arguments(monkeypatch):
    calls = []

    def fake_train(*args, **kwargs):
        calls.append(kwargs)
        return Path("artifacts/runs/matrix_run")

    monkeypatch.setattr(train, "train_from_config", fake_train)

    train.main(
        [
            "--config",
            "configs/holstein2025_colab_proplus.yaml",
            "--model",
            "convnexttiny",
            "--protocol",
            "holstein2025_closed_set",
            "--seed",
            "2",
            "--metadata",
            "artifacts/metadata/holstein2025_open_set.csv",
            "--run-id",
            "matrix_holstein_convnext_seed2",
            "--skip-completed",
        ]
    )

    assert calls[0]["metadata_path"] == "artifacts/metadata/holstein2025_open_set.csv"
    assert calls[0]["run_id"] == "matrix_holstein_convnext_seed2"
    assert calls[0]["skip_completed"] is True
