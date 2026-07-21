from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from cattle_id.hashing import collect_hash_rows, collect_hash_rows_from_csv_manifest, write_sha256_manifest


def test_collect_hash_rows_hashes_files_and_directories_relative_to_root(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "nested" / "b.txt"
    file_b.parent.mkdir()
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    rows = collect_hash_rows([file_b.parent, file_a], root=tmp_path)

    assert [row["path"] for row in rows] == ["a.txt", "nested/b.txt"]
    assert rows[0]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert rows[1]["sha256"] == hashlib.sha256(b"beta").hexdigest()


def test_write_sha256_manifest_writes_csv(tmp_path: Path):
    file_path = tmp_path / "annotation.csv"
    file_path.write_text("image_path,status\nx.jpg,present\n", encoding="utf-8")
    output_path = tmp_path / "hashes.csv"

    written = write_sha256_manifest([file_path], output_path, root=tmp_path)

    with written.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert written == output_path
    assert rows[0]["path"] == "annotation.csv"
    assert rows[0]["size_bytes"] == str(file_path.stat().st_size)


def test_collect_hash_rows_from_csv_manifest_hashes_only_declared_paths(tmp_path: Path):
    kept = tmp_path / "kept.jpg"
    extra = tmp_path / "extra.jpg"
    kept.write_bytes(b"kept")
    extra.write_bytes(b"extra")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text(f"image_path\n{kept}\n", encoding="utf-8")

    rows = collect_hash_rows_from_csv_manifest(manifest, image_path_column="image_path", root=tmp_path)

    assert rows == [
        {
            "path": "kept.jpg",
            "size_bytes": 4,
            "sha256": hashlib.sha256(b"kept").hexdigest(),
        }
    ]
