import json

from cattle_id.logging_utils import append_event, log_line, write_json


def test_logging_utils_write_human_and_jsonl_artifacts(tmp_path):
    log_path = tmp_path / "run.log"
    run_dir = tmp_path / "run"

    log_line(log_path, "train", "stage=head epoch=1 end accuracy=0.5")
    append_event(run_dir, "epoch_end", stage="head", epoch=1, metrics={"accuracy": 0.5})
    append_event(run_dir, "run_completed", run_dir=str(run_dir))
    write_json(run_dir / "config_resolved.json", {"training": {"batch_size": 128}})

    log_text = log_path.read_text(encoding="utf-8")
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))

    assert "[train]" in log_text
    assert "epoch=1" in log_text
    assert events[0]["event"] == "epoch_end"
    assert events[0]["metrics"]["accuracy"] == 0.5
    assert events[1]["event"] == "run_completed"
    assert events[1]["run_dir"] == str(run_dir)
    assert config["training"]["batch_size"] == 128
