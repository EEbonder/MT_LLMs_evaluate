import argparse
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import VALID_DIRECTIONS
from mt_eval.comet_eval import run_two_reference_comet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--ref1", type=Path, required=True)
    parser.add_argument("--ref2", type=Path, required=True)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--direction", choices=sorted(VALID_DIRECTIONS))
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(
    argv: list[str] | None = None,
    run_two_reference_comet_fn: (
        Callable[[Path, Path, Path, Path, str], dict] | None
    ) = None,
) -> int:
    args = build_parser().parse_args(argv)
    run_eval = (
        run_two_reference_comet
        if run_two_reference_comet_fn is None
        else run_two_reference_comet_fn
    )
    result = run_eval(args.src, args.hyp, args.ref1, args.ref2, args.model)
    payload = dict(result)
    if args.model_name:
        payload["model"] = args.model_name
    if args.direction:
        payload["direction"] = args.direction
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
