import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import load_runtime_config
from mt_eval.download import build_download_jobs, download_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--models", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_root = args.config_root or project_path("configs")
    cfg = load_runtime_config(
        config_root / "models.yaml",
        config_root / "prompts.yaml",
        config_root / "generation.yaml",
    )
    jobs = build_download_jobs(
        cfg,
        cache_dir=args.cache_dir,
        selected_models=args.models,
    )
    for job in jobs:
        download_job(job)
    return 0


if __name__ == "__main__":
    main()
