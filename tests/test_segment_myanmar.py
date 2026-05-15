from pathlib import Path
import runpy
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.segmentation import (
    CommandTokenizer,
    FallbackTokenizer,
    build_tokenizer,
    segment_file,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def segment(self, text: str) -> str:
        self.calls.append(text)
        return "|".join(text.split())


class NewlineTokenizer:
    def segment(self, text: str) -> str:
        return "tok1\ntok2 " + text


def test_segment_file_writes_one_segmented_line_per_input_line(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("alpha beta\ngamma delta\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    tokenizer = FakeTokenizer()

    segment_file(src, out, tokenizer)

    assert out.read_text(encoding="utf-8").splitlines() == [
        "alpha|beta",
        "gamma|delta",
    ]
    assert tokenizer.calls == ["alpha beta", "gamma delta"]


def test_segment_file_preserves_blank_line_alignment(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("alpha beta\n\ngamma\n", encoding="utf-8")
    out = tmp_path / "nested" / "out.txt"
    tokenizer = FakeTokenizer()

    segment_file(src, out, tokenizer)

    assert out.read_text(encoding="utf-8").splitlines() == [
        "alpha|beta",
        "",
        "gamma",
    ]
    assert tokenizer.calls == ["alpha beta", "gamma"]


def test_segment_file_normalizes_multiline_tokenizer_output(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("alpha\n", encoding="utf-8")
    out = tmp_path / "out.txt"

    segment_file(src, out, NewlineTokenizer())

    assert out.read_text(encoding="utf-8").splitlines() == ["tok1 tok2 alpha"]


def test_command_tokenizer_uses_runner_and_normalizes_output():
    calls = []

    def fake_runner(command, text):
        calls.append((command, text))
        return "tok1\ntok2"

    tokenizer = CommandTokenizer("python fake_segmenter.py", runner=fake_runner)

    assert tokenizer.segment("abc") == "tok1\ntok2"
    assert calls == [("python fake_segmenter.py", "abc")]


def test_build_tokenizer_prefers_explicit_command():
    tokenizer = build_tokenizer(
        backend="command",
        command="python fake_segmenter.py",
    )

    assert isinstance(tokenizer, CommandTokenizer)
    assert tokenizer.command == "python fake_segmenter.py"


def test_build_tokenizer_rejects_missing_command():
    with pytest.raises(ValueError, match="requires --command"):
        build_tokenizer(backend="command", command=None)


def test_build_tokenizer_uses_pyidaungsu_module_when_available():
    captured = {}

    class FakePyidaungsuModule:
        @staticmethod
        def tokenize(text: str, **kwargs) -> list[str]:
            captured["kwargs"] = kwargs
            return ["py", text]

    tokenizer = build_tokenizer(
        backend="pyidaungsu",
        pyidaungsu_module=FakePyidaungsuModule(),
    )

    assert tokenizer.segment("abc") == "py abc"
    assert captured["kwargs"] == {"form": "word"}


def test_build_tokenizer_fails_clearly_when_pyidaungsu_missing():
    with pytest.raises(NotImplementedError, match="pyidaungsu is not installed"):
        build_tokenizer(
            backend="pyidaungsu",
            pyidaungsu_module=None,
            pyidaungsu_loader=lambda: None,
        )


def test_build_tokenizer_auto_falls_back_when_no_formal_backend_available():
    tokenizer = build_tokenizer(
        backend="auto",
        command=None,
        pyidaungsu_module=None,
        pyidaungsu_loader=lambda: None,
    )

    assert isinstance(tokenizer, FallbackTokenizer)


def test_build_tokenizer_auto_prefers_command_before_pyidaungsu():
    class FakePyidaungsuModule:
        @staticmethod
        def tokenize(text: str, **kwargs) -> list[str]:
            return ["py", text]

    tokenizer = build_tokenizer(
        backend="auto",
        command="python fake_segmenter.py",
        pyidaungsu_module=FakePyidaungsuModule(),
    )

    assert isinstance(tokenizer, CommandTokenizer)
    assert tokenizer.command == "python fake_segmenter.py"


def test_build_tokenizer_auto_uses_pyidaungsu_when_command_missing():
    class FakePyidaungsuModule:
        @staticmethod
        def tokenize(text: str, **kwargs) -> list[str]:
            return ["py", text]

    tokenizer = build_tokenizer(
        backend="auto",
        command=None,
        pyidaungsu_module=FakePyidaungsuModule(),
    )

    assert tokenizer.segment("abc") == "py abc"


def test_segment_myanmar_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/segment_myanmar.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--command" in result.stdout
    assert "fallback is for local smoke tests" in result.stdout


def test_segment_myanmar_cli_succeeds_with_explicit_fallback_backend(tmp_path: Path):
    src = tmp_path / "src.txt"
    out = tmp_path / "out.txt"
    src.write_text("alpha beta\n\ngamma\nxyz\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/segment_myanmar.py",
            "--backend",
            "fallback",
            "--input",
            str(src),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert out.read_text(encoding="utf-8").splitlines() == [
        "alpha beta",
        "",
        "g a m m a",
        "x y z",
    ]


def test_segment_myanmar_cli_builds_tokenizer_and_calls_segment_file(
    tmp_path: Path,
):
    src = tmp_path / "src.txt"
    out = tmp_path / "out.txt"
    src.write_text("alpha beta\n", encoding="utf-8")
    captured = {}

    class BuiltTokenizer:
        pass

    tokenizer = BuiltTokenizer()

    def fake_build_tokenizer(*, backend, command):
        captured["build"] = {
            "backend": backend,
            "command": command,
        }
        return tokenizer

    def fake_segment_file(input_path, output_path, tokenizer):
        captured["call"] = {
            "input_path": input_path,
            "output_path": output_path,
            "tokenizer": tokenizer,
        }

    module_globals = runpy.run_path("scripts/segment_myanmar.py", run_name="__test__")
    module_globals["main"](
        [
            "--backend",
            "command",
            "--command",
            "python fake_segmenter.py",
            "--input",
            str(src),
            "--output",
            str(out),
        ],
        build_tokenizer_fn=fake_build_tokenizer,
        segment_file_fn=fake_segment_file,
    )

    assert captured == {
        "build": {
            "backend": "command",
            "command": "python fake_segmenter.py",
        },
        "call": {
            "input_path": src,
            "output_path": out,
            "tokenizer": tokenizer,
        },
    }
