from pathlib import Path
import runpy
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.validation import validate_parallel_files


def test_validate_parallel_files_accepts_matching_non_empty_rows(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    output.write_text("x\ny\n", encoding="utf-8")

    validate_parallel_files(source, output)


def test_validate_parallel_files_allows_blank_output_for_blank_source_row(
    tmp_path: Path,
):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\n\nb\n", encoding="utf-8")
    output.write_text("x\n\ny\n", encoding="utf-8")

    validate_parallel_files(source, output)


def test_validate_parallel_files_rejects_line_count_mismatch(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    output.write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line count mismatch"):
        validate_parallel_files(source, output)


def test_validate_parallel_files_rejects_empty_output_rows(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    output.write_text("x\n   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty output rows"):
        validate_parallel_files(source, output)


def test_validate_outputs_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/validate_outputs.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--source" in result.stdout
    assert "--output" in result.stdout


def test_validate_outputs_cli_returns_zero_on_success(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    output.write_text("x\ny\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_outputs.py",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_validate_outputs_cli_returns_non_zero_on_failure(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    output.write_text("x\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_outputs.py",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "line count mismatch" in result.stderr


def test_validate_outputs_cli_invokes_validation_function(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\n", encoding="utf-8")
    output.write_text("x\n", encoding="utf-8")
    captured = {}

    def fake_validate_parallel_files(source_path, output_path):
        captured["call"] = {
            "source_path": source_path,
            "output_path": output_path,
        }

    module_globals = runpy.run_path("scripts/validate_outputs.py", run_name="__test__")
    exit_code = module_globals["main"](
        ["--source", str(source), "--output", str(output)],
        validate_parallel_files_fn=fake_validate_parallel_files,
    )

    assert exit_code == 0
    assert captured == {
        "call": {
            "source_path": source,
            "output_path": output,
        }
    }
