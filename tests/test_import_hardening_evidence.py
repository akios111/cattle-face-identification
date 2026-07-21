from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))

import import_hardening_evidence
import verify_hardening_evidence


def _write_bundle(path: Path, config_data: dict[str, bytes], *, extra: dict[str, bytes] | None = None) -> None:
    files = dict(config_data)
    for name in verify_hardening_evidence.REQUIRED_TABLES:
        files[f"thesis/tables/hardening_v2/{name}"] = b"column\nvalue\n"
    sanity_rows = [
        "state_audit_complete,backbone_hash_equal,batchnorm_hash_equal,"
        "backbone_max_abs_difference,batchnorm_max_abs_difference"
    ]
    sanity_rows.extend("true,true,true,0.0,0.0" for _ in range(5))
    files["artifacts/audits/holstein/frozen_imagenet_sanity.csv"] = (
        "\n".join(sanity_rows) + "\n"
    ).encode()
    files["artifacts/audits/holstein/frozen_imagenet_sanity.json"] = (
        json.dumps(
            {
                "complete": True,
                "backbone_hashes_equal": True,
                "batchnorm_moving_statistics_equal": True,
            }
        )
        + "\n"
    ).encode()
    files["artifacts/audits/holstein/frozen_imagenet_sanity.md"] = b"# Frozen sanity audit\n"
    files["thesis/chapters/generated/hardening_v2_results.tex"] = b"results\n"
    files.update(extra or {})
    model_hashes = [{"sha256": f"{index:064x}"} for index in range(30)]
    files["MODEL_HASHES.json"] = (json.dumps(model_hashes) + "\n").encode()
    manifest = {
        "protocol_version": "hardening_v2",
        "training_jobs": 29,
        "holstein_evaluations": 16,
        "raw_images_included": False,
        "files": [
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(files.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("MANIFEST.json", json.dumps(manifest))


def _configs(root: Path) -> dict[str, bytes]:
    values = {}
    for name in import_hardening_evidence.IMMUTABLE_CONFIGS:
        data = f"config: {name}\n".encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        values[name] = data
    return values


def test_import_hardening_evidence_verifies_and_extracts_whitelist(tmp_path):
    configs = _configs(tmp_path)
    bundle = tmp_path / "download" / "evidence.zip"
    _write_bundle(bundle, configs)

    result = import_hardening_evidence.import_evidence(bundle, root=tmp_path)

    assert result["training_jobs"] == 29
    assert (tmp_path / "artifacts/evidence/hardening_v2_evidence.zip").is_file()
    assert (tmp_path / "artifacts/evidence/hardening_v2_MANIFEST.json").is_file()
    assert (tmp_path / "thesis/tables/hardening_v2/hardening_primary_runs.csv").is_file()
    assert (tmp_path / "thesis/chapters/generated/hardening_v2_results.tex").is_file()


def test_import_hardening_evidence_rejects_path_traversal(tmp_path):
    configs = _configs(tmp_path)
    bundle = tmp_path / "download" / "evidence.zip"
    _write_bundle(bundle, configs, extra={"../escape.txt": b"escape"})

    with pytest.raises(ValueError, match="unsafe hardening evidence paths"):
        import_hardening_evidence.import_evidence(bundle, root=tmp_path)


def test_import_hardening_evidence_rejects_changed_local_config(tmp_path):
    configs = _configs(tmp_path)
    bundle = tmp_path / "download" / "evidence.zip"
    _write_bundle(bundle, configs)
    changed = tmp_path / sorted(import_hardening_evidence.IMMUTABLE_CONFIGS)[0]
    changed.write_text("changed: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable config differs"):
        import_hardening_evidence.import_evidence(bundle, root=tmp_path)
