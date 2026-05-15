import argparse
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.segmentation import Segmenter, build_tokenizer, segment_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment Myanmar text for zh2my BLEU preparation. "
            "Use command or pyidaungsu for formal runs; fallback is for local smoke tests."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "command", "pyidaungsu", "fallback"],
        default="auto",
    )
    parser.add_argument("--command")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    build_tokenizer_fn: Callable[..., Segmenter] | None = None,
    segment_file_fn: Callable[[Path, Path, Segmenter], None] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    tokenizer_factory = build_tokenizer if build_tokenizer_fn is None else build_tokenizer_fn
    segment = segment_file if segment_file_fn is None else segment_file_fn
    tokenizer = tokenizer_factory(backend=args.backend, command=args.command)
    segment(args.input, args.output, tokenizer)


if __name__ == "__main__":
    main()
