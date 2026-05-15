from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def validate_parallel_files(source_path: Path, output_path: Path) -> None:
    source_lines = _read_lines(source_path)
    output_lines = _read_lines(output_path)

    if len(source_lines) != len(output_lines):
        raise ValueError(
            f"line count mismatch: source has {len(source_lines)} lines, "
            f"output has {len(output_lines)} lines"
        )

    empty_rows = [
        str(index)
        for index, (source_line, output_line) in enumerate(
            zip(source_lines, output_lines), start=1
        )
        if source_line.strip() and not output_line.strip()
    ]
    if empty_rows:
        raise ValueError(
            "empty output rows: " + ", ".join(empty_rows)
        )
