from cattle_id.runtime import configure_tensorflow_acceleration
from cattle_id.training import build_training_stages


def test_build_training_stages_supports_head_then_finetune_plan():
    stages = build_training_stages(
        {
            "learning_rate": 1e-3,
            "epochs": 20,
            "fine_tune": {
                "enabled": True,
                "epochs": 7,
                "learning_rate": 1e-5,
                "trainable_last_n": 32,
            },
        }
    )

    assert [stage.name for stage in stages] == ["head", "finetune"]
    assert stages[0].epochs == 20
    assert stages[0].learning_rate == 1e-3
    assert stages[0].trainable_backbone is False
    assert stages[1].epochs == 7
    assert stages[1].learning_rate == 1e-5
    assert stages[1].trainable_backbone is True
    assert stages[1].trainable_last_n == 32


def test_build_training_stages_can_disable_finetune_for_fast_smoke_runs():
    stages = build_training_stages(
        {
            "learning_rate": 1e-3,
            "epochs": 2,
            "fine_tune": {"enabled": False},
        }
    )

    assert len(stages) == 1
    assert stages[0].name == "head"


def test_configure_tensorflow_acceleration_handles_cpu_safe_policy():
    info = configure_tensorflow_acceleration({"mixed_precision": "float32", "xla": False})

    assert info["mixed_precision_policy"] == "float32"
    assert "gpus" in info
