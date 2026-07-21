from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from import_hardening_visual_addendum import verify_visual_addendum
from verify_hardening_evidence import verify_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "v1.0.0"
PROTOCOL_VERSION = "hardening_v2"
PRIMARY_RUN_ID = (
    "matrix_efficientnetv2b3_paper_random_hardening_v2_train1_split1_aug1"
)
DEFAULT_MANIFEST = Path("release/v1.0.0/ASSET_MANIFEST.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rooted_file(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release asset escaped project root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _primary_model_record(root: Path) -> dict[str, object]:
    model_hashes_path = _rooted_file(
        root, "artifacts/evidence/hardening_v2_MODEL_HASHES.json"
    )
    records = json.loads(model_hashes_path.read_text(encoding="utf-8"))
    matches = [row for row in records if row.get("run_id") == PRIMARY_RUN_ID]
    if len(matches) != 1:
        raise ValueError(
            f"expected one primary checkpoint hash record, found {len(matches)}"
        )
    record = matches[0]
    checkpoint = _rooted_file(
        root, f"artifacts/runs/{PRIMARY_RUN_ID}/model.keras"
    )
    actual_sha = sha256_file(checkpoint)
    actual_size = checkpoint.stat().st_size
    if actual_sha != str(record.get("sha256")):
        raise ValueError("primary checkpoint SHA-256 differs from MODEL_HASHES.json")
    if actual_size != int(record.get("size_bytes", -1)):
        raise ValueError("primary checkpoint size differs from MODEL_HASHES.json")
    return {
        "path": checkpoint,
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "record": record,
    }


def collect_release_asset_manifest(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, object]:
    root = Path(root).resolve()
    primary = _primary_model_record(root)
    evidence = _rooted_file(root, "artifacts/evidence/hardening_v2_evidence.zip")
    visual = _rooted_file(
        root, "artifacts/evidence/hardening_v2_visual_addendum.zip"
    )

    evidence_verification = verify_bundle(str(evidence))
    visual_verification, _ = verify_visual_addendum(
        visual,
        source_evidence=evidence,
    )
    evidence_sha = sha256_file(evidence)
    visual_sha = sha256_file(visual)
    if evidence_verification.get("verified") is not True:
        raise ValueError("hardening evidence verification did not complete")
    if visual_verification.get("verified") is not True:
        raise ValueError("visual addendum verification did not complete")
    if visual_verification.get("source_evidence_sha256") != evidence_sha:
        raise ValueError("visual addendum is not bound to the release evidence ZIP")
    if visual_verification.get("zip_sha256") != visual_sha:
        raise ValueError("visual addendum verifier returned a different ZIP hash")

    record = primary["record"]
    assets = [
        {
            "role": "primary_checkpoint",
            "release_name": "efficientnetv2b3_paper_random_hardening_v2_seed1.keras",
            "source_path": f"artifacts/runs/{PRIMARY_RUN_ID}/model.keras",
            "size_bytes": primary["size_bytes"],
            "sha256": primary["sha256"],
            "metadata": {
                "run_id": PRIMARY_RUN_ID,
                "model": record["model"],
                "protocol": record["protocol"],
                "training_seed": int(record["training_seed"]),
                "split_seed": int(record["split_seed"]),
                "augmentation_seed": int(record["augmentation_seed"]),
            },
        },
        {
            "role": "hardening_evidence",
            "release_name": "hardening_v2_evidence.zip",
            "source_path": "artifacts/evidence/hardening_v2_evidence.zip",
            "size_bytes": evidence.stat().st_size,
            "sha256": evidence_sha,
            "metadata": {
                "training_jobs": int(evidence_verification["training_jobs"]),
                "holstein_evaluations": int(
                    evidence_verification["holstein_evaluations"]
                ),
                "model_hashes": int(evidence_verification["models"]),
                "manifested_files": int(evidence_verification["files"]),
            },
        },
        {
            "role": "visual_addendum",
            "release_name": "hardening_v2_visual_addendum.zip",
            "source_path": "artifacts/evidence/hardening_v2_visual_addendum.zip",
            "size_bytes": visual.stat().st_size,
            "sha256": visual_sha,
            "metadata": {
                "source_evidence_sha256": evidence_sha,
                "manifested_files": int(visual_verification["files"]),
                "gradcam_runs": int(visual_verification["gradcam_runs"]),
                "holstein_error_identities": int(
                    len(visual_verification["holstein_error_identities"])
                ),
            },
        },
    ]
    release_names = [str(row["release_name"]) for row in assets]
    if len(release_names) != len(set(release_names)):
        raise ValueError("release asset names are not unique")
    return {
        "schema_version": 1,
        "release_version": RELEASE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "raw_dataset_images_included": False,
        "assets": assets,
    }


def _checksum_text(manifest: dict[str, object]) -> str:
    assets = manifest.get("assets", [])
    return "".join(
        f"{row['sha256']}  {row['release_name']}\n"
        for row in sorted(assets, key=lambda item: str(item["release_name"]))
    )


def build_release_asset_manifest(
    root: str | Path = PROJECT_ROOT,
    output: str | Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    root = Path(root).resolve()
    manifest = collect_release_asset_manifest(root)
    output = Path(output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release manifest escaped project root: {output}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = output.parent / "SHA256SUMS"
    checksums.write_text(_checksum_text(manifest), encoding="utf-8")
    return {
        "verified": True,
        "manifest": output.relative_to(root).as_posix(),
        "checksums": checksums.relative_to(root).as_posix(),
        "assets": len(manifest["assets"]),
        "manifest_sha256": sha256_file(output),
    }


def verify_release_asset_manifest(
    root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    manifest_path = (
        manifest_path.resolve()
        if manifest_path.is_absolute()
        else (root / manifest_path).resolve()
    )
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = collect_release_asset_manifest(root)
    if actual != expected:
        raise ValueError("release asset manifest differs from verified local assets")
    checksums = manifest_path.parent / "SHA256SUMS"
    if checksums.read_text(encoding="utf-8") != _checksum_text(expected):
        raise ValueError("release SHA256SUMS differs from the asset manifest")
    return {
        "verified": True,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "checksums": checksums.relative_to(root).as_posix(),
        "assets": len(expected["assets"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the v1.0.0 public release asset manifest."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_release_asset_manifest(args.root, args.out)
    else:
        result = build_release_asset_manifest(args.root, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
