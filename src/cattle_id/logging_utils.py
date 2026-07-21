from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_line(log_path: str | Path, component: str, message: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}][{component}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_event(event_dir: str | Path, event: str, **payload: Any) -> None:
    path = Path(event_dir) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": timestamp(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        snapshot["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        snapshot["nvidia_smi"] = f"unavailable: {exc}"
    return snapshot
