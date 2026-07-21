from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))

import reproduce_release


def test_canonical_text_sha256_normalizes_line_endings(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_bytes(b"a,b\r\n1,2\r\n")
    right.write_bytes(b"a,b\n1,2\n")

    assert reproduce_release.canonical_text_sha256(left) == reproduce_release.canonical_text_sha256(right)


def test_compare_text_outputs_fails_closed_on_difference(tmp_path):
    generated = tmp_path / "generated"
    expected = tmp_path / "expected"
    generated.mkdir()
    expected.mkdir()
    (generated / "table.csv").write_text("value\n1\n", encoding="utf-8")
    (expected / "table.csv").write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from verified evidence"):
        reproduce_release.compare_text_outputs(generated, expected, ["table.csv"])


def test_compare_text_outputs_reports_matching_hash(tmp_path):
    generated = tmp_path / "generated"
    expected = tmp_path / "expected"
    generated.mkdir()
    expected.mkdir()
    (generated / "table.csv").write_text("value\n1\n", encoding="utf-8")
    (expected / "table.csv").write_bytes(b"value\r\n1\r\n")

    rows = reproduce_release.compare_text_outputs(generated, expected, ["table.csv"])

    assert rows == [
        {
            "path": "table.csv",
            "canonical_sha256": reproduce_release.canonical_text_sha256(generated / "table.csv"),
            "match": True,
        }
    ]


def test_compare_csv_outputs_allows_sub_tolerance_float_roundoff(tmp_path):
    generated = tmp_path / "generated"
    expected = tmp_path / "expected"
    generated.mkdir()
    expected.mkdir()
    (generated / "table.csv").write_text("label,value\na,0.1000000000001\n", encoding="utf-8")
    (expected / "table.csv").write_text("label,value\na,0.1\n", encoding="utf-8")

    rows = reproduce_release.compare_csv_outputs(generated, expected, ["table.csv"])

    assert rows[0]["match"] is True
    assert rows[0]["max_abs_numeric_difference"] < 1e-12


def test_compare_csv_outputs_rejects_meaningful_numeric_difference(tmp_path):
    generated = tmp_path / "generated"
    expected = tmp_path / "expected"
    generated.mkdir()
    expected.mkdir()
    (generated / "table.csv").write_text("label,value\na,0.10000001\n", encoding="utf-8")
    (expected / "table.csv").write_text("label,value\na,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="numeric values differ"):
        reproduce_release.compare_csv_outputs(generated, expected, ["table.csv"])


def test_summary_commands_use_temporary_outputs_under_root(tmp_path):
    tables = tmp_path / "artifacts" / "reproduction" / "tables"

    commands = reproduce_release.summary_commands(
        root=tmp_path,
        tables_dir=tables,
        python="python-test",
    )

    assert len(commands) == 1
    assert commands[0][0] == "python-test"
    assert str(Path("artifacts") / "reproduction" / "tables") in commands[0]
    assert commands[0][-2:] == [
        "--verified-model-hashes",
        "artifacts/evidence/hardening_v2_MODEL_HASHES.json",
    ]
