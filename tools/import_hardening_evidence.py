from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from verify_hardening_evidence import verify_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ZIP = Path("artifacts/evidence/hardening_v2_evidence.zip")
IMPORT_MANIFEST = Path("artifacts/evidence/hardening_v2_import_manifest.json")
IMMUTABLE_CONFIGS = {
    "configs/cattlessfr_hardening_v2_colab_proplus.yaml",
    "configs/experiment_matrix_hardening_v2.yaml",
    "configs/experiment_matrix_holstein_hardening_v2.yaml",
    "configs/final_evidence_scope_hardening_v2.yaml",
}
ALLOWED_PREFIXES = (
    "artifacts/runs/",
    "artifacts/metadata/hardening_v2/",
    "artifacts/matrix/",
    "artifacts/audits/",
    "thesis/tables/hardening_v2/",
    "thesis/figures/hardening_v2/",
    "thesis/chapters/generated/hardening_v2_",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name)


def _destination_for(name: str) -> Path | None:
    if name == "MANIFEST.json":
        return Path("artifacts/evidence/hardening_v2_MANIFEST.json")
    if name == "MODEL_HASHES.json":
        return Path("artifacts/evidence/hardening_v2_MODEL_HASHES.json")
    if name in IMMUTABLE_CONFIGS or any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return Path(PurePosixPath(name))
    return None


def import_evidence(source: str | Path, *, root: str | Path = PROJECT_ROOT) -> dict[str, object]:
    root = Path(root).resolve()
    source = Path(source).resolve()
    verification = verify_bundle(str(source))
    source_sha256 = sha256_file(source)

    staged: list[tuple[Path, Path]] = []
    imported: list[str] = []
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        invalid = [name for name in names if not _safe_archive_name(name)]
        if invalid:
            raise ValueError(f"unsafe hardening evidence paths: {invalid[:5]}")
        unknown = [name for name in names if _destination_for(name) is None]
        if unknown:
            raise ValueError(f"unexpected hardening evidence paths: {unknown[:5]}")

        for name in IMMUTABLE_CONFIGS:
            bundled = archive.read(name)
            local = root / name
            if not local.is_file() or local.read_bytes() != bundled:
                raise ValueError(f"local immutable config differs from verified evidence: {name}")

        evidence_dir = root / "artifacts" / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix="hardening_import_", dir=evidence_dir))
        try:
            for name in names:
                if name in IMMUTABLE_CONFIGS:
                    continue
                destination = _destination_for(name)
                if destination is None:
                    continue
                staged_path = staging_root / destination
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                staged_path.write_bytes(archive.read(name))
                staged.append((staged_path, root / destination))
                imported.append(destination.as_posix())

            for staged_path, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged_path, destination)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    canonical = root / CANONICAL_ZIP
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if source != canonical.resolve():
        shutil.copy2(source, canonical)

    payload = {
        "version": 2,
        "protocol_version": "hardening_v2",
        "source": str(source),
        "source_sha256": source_sha256,
        "training_jobs": verification["training_jobs"],
        "holstein_evaluations": verification["holstein_evaluations"],
        "models": verification["models"],
        "imported_files": sorted(imported),
        "imported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    manifest = root / IMPORT_MANIFEST
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**payload, "canonical_zip": CANONICAL_ZIP.as_posix(), "manifest": IMPORT_MANIFEST.as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and import a hardening_v2 evidence ZIP from Drive.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    result = import_evidence(args.source, root=args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
