from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Cannot hash missing path: {path}")
    return sorted({path.resolve() for path in files}, key=lambda item: item.as_posix())


def _display_path(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def collect_hash_rows(paths: list[str | Path], *, root: str | Path | None = None) -> list[dict[str, object]]:
    root_path = Path(root).resolve() if root is not None else None
    return [
        {
            "path": _display_path(path, root_path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _iter_files(paths)
    ]


def collect_hash_rows_from_csv_manifest(
    metadata_path: str | Path,
    *,
    image_path_column: str = "image_path",
    root: str | Path | None = None,
) -> list[dict[str, object]]:
    metadata = Path(metadata_path)
    with metadata.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or image_path_column not in reader.fieldnames:
            raise ValueError(f"CSV manifest is missing image path column: {image_path_column}")
        paths = [Path(row[image_path_column]) for row in reader if row.get(image_path_column, "").strip()]
    return collect_hash_rows(paths, root=root)


def _write_hash_rows(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_sha256_manifest(
    paths: list[str | Path],
    output_path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    rows = collect_hash_rows(paths, root=root)
    return _write_hash_rows(rows, output_path)


def write_sha256_manifest_from_csv_manifest(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    image_path_column: str = "image_path",
    root: str | Path | None = None,
) -> Path:
    rows = collect_hash_rows_from_csv_manifest(
        metadata_path,
        image_path_column=image_path_column,
        root=root,
    )
    return _write_hash_rows(rows, output_path)
