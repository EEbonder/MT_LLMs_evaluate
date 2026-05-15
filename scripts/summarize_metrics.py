import argparse
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.summary import SummaryResult, summarize_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/summary"),
    )
    return parser


def main(
    argv: list[str] | None = None,
    summarize_metrics_fn: Callable[..., SummaryResult] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    summarize = summarize_metrics if summarize_metrics_fn is None else summarize_metrics_fn
    result = summarize(metrics_root=args.metrics_root, output_dir=args.output_dir)
    print(f"main_csv={result.main_csv_path}")
    print(f"detail_csv={result.detail_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
