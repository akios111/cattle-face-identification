from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"archive path uses backslashes: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def verify_visual_addendum(
    zip_path: str | Path,
    *,
    source_evidence: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, bytes]]:
    zip_path = Path(zip_path).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("visual addendum contains duplicate archive paths")
        for name in names:
            _safe_archive_name(name)
        required = {"ADDENDUM.json", "SHA256SUMS"}
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"visual addendum missing control files: {missing}")
        manifest = json.loads(archive.read("ADDENDUM.json").decode("utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported visual addendum schema")
        if manifest.get("artifact") != "hardening_v2_visual_addendum":
            raise ValueError("unexpected visual addendum artifact type")
        source = manifest.get("source_evidence", {})
        if source.get("immutable") is not True:
            raise ValueError("source evidence is not marked immutable")

        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("visual addendum has no file manifest")
        payloads: dict[str, bytes] = {}
        checksum_lines: list[str] = []
        for entry in entries:
            name = str(entry["path"])
            _safe_archive_name(name)
            if name not in names:
                raise ValueError(f"manifested file missing from ZIP: {name}")
            payload = archive.read(name)
            actual_sha = _sha256_bytes(payload)
            if actual_sha != str(entry["sha256"]):
                raise ValueError(f"SHA-256 mismatch for {name}")
            if len(payload) != int(entry["size_bytes"]):
                raise ValueError(f"size mismatch for {name}")
            payloads[name] = payload
            checksum_lines.append(f"{actual_sha}  {name}\n")
        expected_names = {str(entry["path"]) for entry in entries} | required
        extras = sorted(set(names).difference(expected_names))
        if extras:
            raise ValueError(f"unmanifested ZIP entries: {extras}")
        expected_checksums = "".join(checksum_lines).encode("utf-8")
        if archive.read("SHA256SUMS") != expected_checksums:
            raise ValueError("SHA256SUMS does not match ADDENDUM.json")

    validations = manifest.get("gradcam_validation", [])
    if len(validations) != 2:
        raise ValueError("visual addendum must validate two Grad-CAM runs")
    for validation in validations:
        if validation.get("score_space") != "pre_softmax_logit":
            raise ValueError("visual addendum contains non-logit Grad-CAM")
        if int(validation.get("samples", 0)) != 6:
            raise ValueError("visual addendum requires six Grad-CAM samples per run")
        if float(validation.get("minimum_heatmap_nonzero_fraction", 0.0)) <= 0.0:
            raise ValueError("visual addendum contains an empty heatmap")
    errors = manifest.get("visual_inputs", {}).get("holstein2025", {}).get("verified_errors", [])
    identities = [str(row["animal_id"]) for row in errors]
    if len(identities) != 8 or len(set(identities)) != 8:
        raise ValueError("Holstein2025 error gallery must contain eight distinct identities")

    if source_evidence is not None:
        source_path = Path(source_evidence).resolve()
        actual_source_sha = _sha256_file(source_path)
        if actual_source_sha != str(source.get("sha256")):
            raise ValueError("source evidence SHA-256 mismatch")

    result = {
        "verified": True,
        "zip": str(zip_path),
        "zip_sha256": _sha256_file(zip_path),
        "files": int(len(payloads)),
        "gradcam_runs": int(len(validations)),
        "holstein_error_identities": identities,
        "source_evidence_sha256": str(source.get("sha256")),
    }
    return result, payloads


def extract_verified_payloads(
    payloads: dict[str, bytes],
    *,
    output_root: str | Path,
) -> list[Path]:
    root = Path(output_root).resolve()
    written: list[Path] = []
    for name, payload in payloads.items():
        relative = _safe_archive_name(name)
        target = root.joinpath(*relative.parts).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"refusing extraction outside output root: {target}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and import hardening_v2 visual evidence.")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--extract-root", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args(argv)
    result, payloads = verify_visual_addendum(
        args.zip,
        source_evidence=args.source_evidence,
    )
    if args.extract_root:
        written = extract_verified_payloads(payloads, output_root=args.extract_root)
        result["extracted_files"] = int(len(written))
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
