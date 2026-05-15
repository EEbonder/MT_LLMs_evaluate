import argparse
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.validation import validate_parallel_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    validate_parallel_files_fn: Callable[[Path, Path], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    validate = (
        validate_parallel_files
        if validate_parallel_files_fn is None
        else validate_parallel_files_fn
    )
    try:
        validate(args.source, args.output)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
