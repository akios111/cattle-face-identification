from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


DATASETS = {
    "cattlessfr": {
        "url": "https://github.com/MachineLearningVisionRG/CattleSSFR.git",
        "commit": "099d749e9a766ff0c9b9fbc49112c6b77b29542e",
    },
    "holstein2025": {
        "url": "https://github.com/JZM-shuimu/Cattle-ID.git",
        "commit": "b905600ca4153e8435c1a2c33306a2783de6fbdf",
    },
}


def acquire(name: str, destination: Path) -> Path:
    specification = DATASETS[name]
    if destination.exists():
        if not (destination / ".git").exists():
            raise ValueError(f"existing destination is not a Git checkout: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", specification["url"], str(destination)], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", specification["commit"]], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", specification["commit"]], cwd=destination, check=True)
    if name == "holstein2025":
        subprocess.run(["git", "lfs", "pull"], cwd=destination, check=True)
    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != specification["commit"]:
        raise RuntimeError(f"dataset commit mismatch: {resolved} != {specification['commit']}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire pinned upstream datasets without redistributing them.")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(acquire(args.dataset, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
