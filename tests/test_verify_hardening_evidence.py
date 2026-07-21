import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from verify_hardening_evidence import REQUIRED_SANITY_FILES, REQUIRED_TABLES, verify_bundle


def _bundle(path: Path, *, corrupt: bool = False) -> None:
    files = {
        "MODEL_HASHES.json": json.dumps(
            [
                {"run_id": f"run-{index}", "sha256": hashlib.sha256(str(index).encode()).hexdigest()}
                for index in range(30)
            ]
        ).encode(),
    }
    for table in REQUIRED_TABLES:
        files[f"thesis/tables/hardening_v2/{table}"] = b"column\nvalue\n"
    sanity_rows = [
        "state_audit_complete,backbone_hash_equal,batchnorm_hash_equal,"
        "backbone_max_abs_difference,batchnorm_max_abs_difference",
        *["True,True,True,0.0,0.0" for _ in range(5)],
    ]
    files["artifacts/audits/holstein/frozen_imagenet_sanity.csv"] = (
        "\n".join(sanity_rows) + "\n"
    ).encode()
    files["artifacts/audits/holstein/frozen_imagenet_sanity.json"] = json.dumps(
        {
            "complete": True,
            "backbone_hashes_equal": True,
            "batchnorm_moving_statistics_equal": True,
        }
    ).encode()
    files["artifacts/audits/holstein/frozen_imagenet_sanity.md"] = b"complete\n"
    assert REQUIRED_SANITY_FILES.issubset(files)
    rows = [
        {
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in sorted(files.items())
    ]
    if corrupt:
        rows[0]["sha256"] = "0" * 64
    manifest = {
        "protocol_version": "hardening_v2",
        "training_jobs": 29,
        "holstein_evaluations": 16,
        "raw_images_included": False,
        "files": rows,
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("MANIFEST.json", json.dumps(manifest))


def test_verify_hardening_bundle_accepts_complete_hash_manifest(tmp_path: Path):
    path = tmp_path / "evidence.zip"
    _bundle(path)

    result = verify_bundle(str(path))

    assert result["verified"] is True
    assert result["models"] == 30


def test_verify_hardening_bundle_rejects_hash_mismatch(tmp_path: Path):
    path = tmp_path / "evidence.zip"
    _bundle(path, corrupt=True)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_bundle(str(path))
