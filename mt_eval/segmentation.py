from pathlib import Path
import os
import shlex
import subprocess
from typing import Any, Callable, Protocol


class Segmenter(Protocol):
    def segment(self, text: str) -> str:
        ...


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _normalize_segmented_line(text: str) -> str:
    parts = text.splitlines()
    if not parts:
        return ""
    return " ".join(part.strip() for part in parts).strip()


class FallbackTokenizer:
    """Local smoke-test fallback, not the preferred formal BLEU backend."""

    def segment(self, text: str) -> str:
        parts = text.split()
        if len(parts) > 1:
            return " ".join(parts)
        if not parts:
            return ""
        return " ".join(parts[0])


def _normalize_command(command: str) -> str | list[str]:
    if os.name == "nt":
        return command
    return shlex.split(command, posix=True)


def _run_command(command: str, text: str) -> str:
    result = subprocess.run(
        _normalize_command(command),
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = stderr or f"exit code {result.returncode}"
        raise RuntimeError(f"segmentation command failed: {message}")
    return result.stdout


class CommandTokenizer:
    def __init__(
        self,
        command: str,
        runner: Callable[[str, str], str] | None = None,
    ):
        self.command = command
        self._runner = _run_command if runner is None else runner

    def segment(self, text: str) -> str:
        return self._runner(self.command, text)


class PyidaungsuTokenizer:
    def __init__(self, module: Any):
        self._module = module

    def segment(self, text: str) -> str:
        tokenize = getattr(self._module, "tokenize", None)
        if tokenize is None:
            raise AttributeError("pyidaungsu backend must provide tokenize(text)")
        try:
            result = tokenize(text, form="word")
        except TypeError:
            result = tokenize(text)
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return " ".join(str(item) for item in result)
        raise TypeError("pyidaungsu tokenize(text) must return a string or list")


def _load_pyidaungsu_module() -> Any | None:
    try:
        import pyidaungsu  # type: ignore
    except ImportError:
        return None
    return pyidaungsu


def build_tokenizer(
    backend: str,
    command: str | None = None,
    pyidaungsu_module: Any | None = None,
    pyidaungsu_loader: Callable[[], Any | None] | None = None,
) -> Segmenter:
    module_loader = (
        _load_pyidaungsu_module if pyidaungsu_loader is None else pyidaungsu_loader
    )
    if backend == "command":
        if not command:
            raise ValueError("command backend requires --command")
        return CommandTokenizer(command)

    if backend == "pyidaungsu":
        module = module_loader() if pyidaungsu_module is None else pyidaungsu_module
        if module is None:
            raise NotImplementedError(
                "pyidaungsu is not installed; select --backend command or --backend fallback"
            )
        return PyidaungsuTokenizer(module)

    if backend == "fallback":
        return FallbackTokenizer()

    if backend == "auto":
        if command:
            return CommandTokenizer(command)
        module = module_loader() if pyidaungsu_module is None else pyidaungsu_module
        if module is not None:
            return PyidaungsuTokenizer(module)
        return FallbackTokenizer()

    raise ValueError(f"unsupported backend: {backend}")


def segment_file(input_path: Path, output_path: Path, tokenizer: Segmenter) -> None:
    source_lines = _read_lines(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for line in source_lines:
            if line:
                segmented = _normalize_segmented_line(tokenizer.segment(line))
            else:
                segmented = ""
            handle.write(segmented + "\n")
