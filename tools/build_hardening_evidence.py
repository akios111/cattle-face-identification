from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_id.hardening_matrix import classification_run_complete, run_dir_for_job
from cattle_id.run_matrix import expand_experiment_matrix, load_experiment_matrix


ZIP_TIMESTAMP = (2026, 7, 12, 0, 0, 0)
RUN_FILE_PREFIXES = (
    "metrics",
    "predictions",
    "confusion_matrix",
    "gallery_embeddings",
    "probe_embeddings",
    "runtime_snapshot",
)
RUN_FILES = {
    "manifest.json",
    "config_resolved.json",
    "history.csv",
    "run_complete.json",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include_run_file(path: Path) -> bool:
    if path.name in RUN_FILES:
        return True
    if any(path.name.startswith(prefix) for prefix in RUN_FILE_PREFIXES):
        return path.suffix.lower() in {".json", ".csv", ".npy"}
    return "gradcam" in path.parts and path.suffix.lower() in {".csv", ".png"}


def collect_hardening_files(
    *,
    root: Path = PROJECT_ROOT,
    runs_dir: Path | None = None,
    matrix_path: Path | None = None,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    root = root.resolve()
    runs_dir = (runs_dir or root / "artifacts" / "runs").resolve()
    matrix_path = (matrix_path or root / "configs" / "experiment_matrix_hardening_v2.yaml").resolve()
    jobs = expand_experiment_matrix(load_experiment_matrix(matrix_path))
    if len(jobs) != 29:
        raise ValueError("hardening evidence requires the exact 29-job matrix")
    files: dict[str, Path] = {}
    model_hashes = []
    for job in jobs:
        run_name = run_dir_for_job(job).name
        run_dir = runs_dir / run_name
        if not classification_run_complete(run_dir):
            raise ValueError(f"incomplete hardening run: {run_dir}")
        model = run_dir / "model.keras"
        model_hashes.append(
            {
                "run_id": run_name,
                "model": str(job["model"]),
                "protocol": str(job["protocol"]),
                "training_seed": int(job["training_seed"]),
                "split_seed": int(job["split_seed"]),
                "augmentation_seed": int(job["augmentation_seed"]),
                "size_bytes": int(model.stat().st_size),
                "sha256": _sha256_file(model),
            }
        )
        for path in run_dir.rglob("*"):
            if path.is_file() and _include_run_file(path):
                relative = path.relative_to(runs_dir).as_posix()
                files[f"artifacts/runs/{relative}"] = path
    imagenet_dir = runs_dir / "imagenet_only_efficientnetv2b3_hardening_v2"
    imagenet_model = imagenet_dir / "imagenet_only_efficientnetv2b3.keras"
    if not imagenet_model.is_file():
        raise ValueError("hardening evidence is missing the ImageNet-only control checkpoint")
    model_hashes.append(
        {
            "run_id": imagenet_dir.name,
            "model": "efficientnetv2b3",
            "protocol": "imagenet_only",
            "training_seed": 0,
            "split_seed": 1,
            "augmentation_seed": 1,
            "size_bytes": int(imagenet_model.stat().st_size),
            "sha256": _sha256_file(imagenet_model),
        }
    )
    for path in imagenet_dir.rglob("*"):
        if path.is_file() and _include_run_file(path):
            relative = path.relative_to(runs_dir).as_posix()
            files[f"artifacts/runs/{relative}"] = path

    roots = (
        root / "artifacts" / "metadata" / "hardening_v2",
        root / "artifacts" / "matrix",
        root / "artifacts" / "audits",
        root / "thesis" / "tables" / "hardening_v2",
        root / "thesis" / "figures" / "hardening_v2",
    )
    for source_root in roots:
        if not source_root.exists():
            raise ValueError(f"hardening evidence root is missing: {source_root}")
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue
            if "region_images" in path.parts or "severity_images" in path.parts:
                continue
            files[path.relative_to(root).as_posix()] = path
    for config_name in (
        "cattlessfr_hardening_v2_colab_proplus.yaml",
        "experiment_matrix_hardening_v2.yaml",
        "experiment_matrix_holstein_hardening_v2.yaml",
        "final_evidence_scope_hardening_v2.yaml",
    ):
        path = root / "configs" / config_name
        files[path.relative_to(root).as_posix()] = path
    for fragment_name in (
        "hardening_v2_results.tex",
        "hardening_v2_abstract_el.tex",
        "hardening_v2_abstract_en.tex",
        "hardening_v2_discussion.tex",
        "hardening_v2_conclusion.tex",
    ):
        fragment = root / "thesis" / "chapters" / "generated" / fragment_name
        if not fragment.is_file():
            raise ValueError(f"hardening evidence is missing the generated fragment: {fragment_name}")
        files[fragment.relative_to(root).as_posix()] = fragment
    return files, model_hashes


def _write_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_hardening_bundle(
    output_path: str | Path,
    *,
    root: Path = PROJECT_ROOT,
    runs_dir: Path | None = None,
    matrix_path: Path | None = None,
) -> dict[str, object]:
    files, model_hashes = collect_hardening_files(
        root=root,
        runs_dir=runs_dir,
        matrix_path=matrix_path,
    )
    entries = []
    data_by_name = {}
    for name, path in sorted(files.items()):
        data = path.read_bytes()
        data_by_name[name] = data
        entries.append({"path": name, "size_bytes": len(data), "sha256": _sha256_bytes(data)})
    model_hash_data = (json.dumps(model_hashes, indent=2, sort_keys=True) + "\n").encode("utf-8")
    data_by_name["MODEL_HASHES.json"] = model_hash_data
    entries.append(
        {
            "path": "MODEL_HASHES.json",
            "size_bytes": len(model_hash_data),
            "sha256": _sha256_bytes(model_hash_data),
        }
    )
    manifest = {
        "bundle_version": 2,
        "protocol_version": "hardening_v2",
        "training_jobs": 29,
        "holstein_evaluations": 16,
        "files": sorted(entries, key=lambda row: row["path"]),
        "model_hashes": model_hashes,
        "raw_images_included": False,
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for name, data in sorted(data_by_name.items()):
            _write_entry(archive, name, data)
        _write_entry(archive, "MANIFEST.json", manifest_data)
    return {
        "path": output_path,
        "sha256": _sha256_file(output_path),
        "files": len(entries),
        "models": len(model_hashes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the hash-manifested hardening_v2 evidence bundle.")
    parser.add_argument("--out", default="artifacts/evidence/hardening_v2_evidence.zip")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args(argv)
    result = build_hardening_bundle(
        args.out,
        runs_dir=args.runs,
        matrix_path=args.matrix,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
