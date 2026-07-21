from cattle_id.hardening_postprocess import execute_hardening_postprocess


def test_hardening_postprocess_dry_run_has_all_required_audits_and_evaluations():
    result = execute_hardening_postprocess(dry_run=True)
    commands = result["planned_commands"]

    assert result["training_jobs"] == 29
    assert result["holstein_evaluations"] == 16
    assert len([command for command in commands if "cattle_id.shortcut_audit" in command]) == 2
    assert len([command for command in commands if "cattle_id.robustness" in command]) == 2
    assert len(
        [
            command
            for command in commands
            if "cattle_id.evaluate" in command and "--output-suffix" in command
        ]
    ) == 10
    assert len([command for command in commands if "cattle_id.open_set_evaluate" in command]) == 15
    assert len([command for command in commands if "imagenet-control" in command]) == 1
    assert len([command for command in commands if "checkpoint-audit" in command]) == 1
    assert len([command for command in commands if "control-deltas" in command]) == 1
    assert len([command for command in commands if "cattle_id.gradcam" in command]) == 2
