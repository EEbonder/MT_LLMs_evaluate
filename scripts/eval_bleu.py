import argparse
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import VALID_DIRECTIONS
from mt_eval.bleu_eval import compute_bleu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=sorted(VALID_DIRECTIONS), required=True)
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--refs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name")
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(
    argv: list[str] | None = None,
    compute_bleu_fn: Callable[[str, Path, list[Path]], dict] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    compute = compute_bleu if compute_bleu_fn is None else compute_bleu_fn
    result = compute(args.direction, args.hyp, args.refs)
    payload = dict(result)
    if args.model_name:
        payload["model"] = args.model_name
    payload["direction"] = args.direction
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
