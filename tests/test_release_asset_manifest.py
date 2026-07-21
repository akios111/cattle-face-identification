import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
import release_asset_manifest as release_assets


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = root / "artifacts/runs" / release_assets.PRIMARY_RUN_ID / "model.keras"
    evidence = root / "artifacts/evidence/hardening_v2_evidence.zip"
    visual = root / "artifacts/evidence/hardening_v2_visual_addendum.zip"
    model.parent.mkdir(parents=True)
    evidence.parent.mkdir(parents=True)
    model.write_bytes(b"primary-checkpoint")
    evidence.write_bytes(b"verified-evidence")
    visual.write_bytes(b"verified-visuals")
    records = [
        {
            "run_id": release_assets.PRIMARY_RUN_ID,
            "model": "efficientnetv2b3",
            "protocol": "paper_random_hardening_v2",
            "training_seed": 1,
            "split_seed": 1,
            "augmentation_seed": 1,
            "size_bytes": model.stat().st_size,
            "sha256": _sha(model.read_bytes()),
        }
    ]
    (root / "artifacts/evidence/hardening_v2_MODEL_HASHES.json").write_text(
        json.dumps(records), encoding="utf-8"
    )

    monkeypatch.setattr(
        release_assets,
        "verify_bundle",
        lambda path: {
            "verified": True,
            "files": 507,
            "models": 30,
            "training_jobs": 29,
            "holstein_evaluations": 16,
        },
    )

    def verify_visual(path, *, source_evidence):
        return (
            {
                "verified": True,
                "files": 49,
                "gradcam_runs": 2,
                "holstein_error_identities": [str(index) for index in range(8)],
                "source_evidence_sha256": release_assets.sha256_file(
                    Path(source_evidence)
                ),
                "zip_sha256": release_assets.sha256_file(Path(path)),
            },
            {},
        )

    monkeypatch.setattr(release_assets, "verify_visual_addendum", verify_visual)


def test_release_asset_manifest_builds_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fixture(tmp_path, monkeypatch)

    built = release_assets.build_release_asset_manifest(tmp_path)
    verified = release_assets.verify_release_asset_manifest(tmp_path)
    payload = json.loads(
        (tmp_path / release_assets.DEFAULT_MANIFEST).read_text(encoding="utf-8")
    )

    assert built["assets"] == 3
    assert verified["verified"] is True
    assert [row["role"] for row in payload["assets"]] == [
        "primary_checkpoint",
        "hardening_evidence",
        "visual_addendum",
    ]
    assert payload["raw_dataset_images_included"] is False
    assert len((tmp_path / "release/v1.0.0/SHA256SUMS").read_text().splitlines()) == 3


def test_release_asset_manifest_rejects_changed_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fixture(tmp_path, monkeypatch)
    release_assets.build_release_asset_manifest(tmp_path)
    (tmp_path / "artifacts/evidence/hardening_v2_visual_addendum.zip").write_bytes(
        b"changed-visuals"
    )

    with pytest.raises(ValueError, match="differs from verified local assets"):
        release_assets.verify_release_asset_manifest(tmp_path)


def test_release_asset_manifest_rejects_primary_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fixture(tmp_path, monkeypatch)
    records_path = tmp_path / "artifacts/evidence/hardening_v2_MODEL_HASHES.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records[0]["sha256"] = "0" * 64
    records_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="primary checkpoint SHA-256"):
        release_assets.build_release_asset_manifest(tmp_path)
