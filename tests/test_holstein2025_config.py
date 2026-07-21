from pathlib import Path

import yaml


def test_holstein2025_colab_config_targets_validated_metadata_and_gpu_runs():
    config = yaml.safe_load(Path("configs/holstein2025_colab_proplus.yaml").read_text(encoding="utf-8"))

    assert config["split"]["protocol"] == "holstein2025_closed_set"
    assert config["preprocessing"]["image_size"] == [384, 384]
    assert config["training"]["batch_size"] <= 64
    assert config["training"]["seed"] == 2026
    assert config["output"]["run_dir"] == "artifacts/runs"
