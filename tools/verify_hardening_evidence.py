from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import PurePosixPath
import zipfile


REQUIRED_TABLES = {
    "hardening_primary_runs.csv",
    "hardening_training_seed_stability.csv",
    "hardening_split_sensitivity_runs.csv",
    "hardening_split_sensitivity_summary.csv",
    "hardening_ablation_summary.csv",
    "hardening_shortcut_summary.csv",
    "hardening_shortcut_mcnemar.csv",
    "hardening_region_audit_summary.csv",
    "hardening_severity_summary.csv",
    "hardening_holstein_runs.csv",
    "hardening_holstein_control_deltas.csv",
    "hardening_holstein_group_control_deltas.csv",
    "hardening_holstein_checkpoint_pairwise.csv",
    "hardening_frozen_imagenet_sanity.csv",
}

REQUIRED_SANITY_FILES = {
    "artifacts/audits/holstein/frozen_imagenet_sanity.csv",
    "artifacts/audits/holstein/frozen_imagenet_sanity.json",
    "artifacts/audits/holstein/frozen_imagenet_sanity.md",
}


def verify_bundle(path: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "MANIFEST.json" not in names or "MODEL_HASHES.json" not in names:
            raise ValueError("hardening evidence bundle lacks its manifests")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if manifest.get("protocol_version") != "hardening_v2":
            raise ValueError("hardening evidence protocol version mismatch")
        if manifest.get("training_jobs") != 29 or manifest.get("holstein_evaluations") != 16:
            raise ValueError("hardening evidence job contract mismatch")
        if manifest.get("raw_images_included") is not False:
            raise ValueError("hardening evidence must not redistribute raw images")
        declared = {row["path"]: row for row in manifest.get("files", [])}
        if set(declared) != names - {"MANIFEST.json"}:
            raise ValueError("hardening evidence manifest file list mismatch")
        for name, row in declared.items():
            data = archive.read(name)
            if len(data) != int(row["size_bytes"]):
                raise ValueError(f"hardening evidence size mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"hardening evidence SHA-256 mismatch: {name}")
        table_names = {PurePosixPath(name).name for name in names if "/tables/hardening_v2/" in name}
        missing_tables = sorted(REQUIRED_TABLES.difference(table_names))
        if missing_tables:
            raise ValueError(f"hardening evidence missing thesis tables: {missing_tables}")
        missing_sanity = sorted(REQUIRED_SANITY_FILES.difference(names))
        if missing_sanity:
            raise ValueError(f"hardening evidence missing frozen sanity audit: {missing_sanity}")
        sanity = json.loads(
            archive.read("artifacts/audits/holstein/frozen_imagenet_sanity.json")
        )
        if not (
            sanity.get("complete") is True
            and sanity.get("backbone_hashes_equal") is True
            and sanity.get("batchnorm_moving_statistics_equal") is True
        ):
            raise ValueError("hardening evidence frozen sanity audit is incomplete")
        sanity_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "artifacts/audits/holstein/frozen_imagenet_sanity.csv"
                    ).decode("utf-8")
                )
            )
        )
        if len(sanity_rows) != 5:
            raise ValueError("hardening evidence frozen sanity audit requires five rows")
        if any(
            row.get("state_audit_complete", "").lower() != "true"
            or row.get("backbone_hash_equal", "").lower() != "true"
            or row.get("batchnorm_hash_equal", "").lower() != "true"
            or float(row.get("backbone_max_abs_difference", "nan")) != 0.0
            or float(row.get("batchnorm_max_abs_difference", "nan")) != 0.0
            for row in sanity_rows
        ):
            raise ValueError("hardening evidence frozen sanity state comparison failed")
        model_hashes = json.loads(archive.read("MODEL_HASHES.json"))
        if len(model_hashes) != 30 or len({row["sha256"] for row in model_hashes}) != 30:
            raise ValueError("hardening evidence requires 30 unique checkpoint hashes")
        forbidden = [
            name
            for name in names
            if "/processed/" in name
            or "/external_public/" in name
            or "region_images/" in name
            or "severity_images/" in name
        ]
        if forbidden:
            raise ValueError(f"hardening evidence contains raw/materialized image roots: {forbidden[:5]}")
    return {
        "verified": True,
        "files": len(declared),
        "models": len(model_hashes),
        "training_jobs": 29,
        "holstein_evaluations": 16,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a hardening_v2 evidence bundle.")
    parser.add_argument("--zip", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_bundle(args.zip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
