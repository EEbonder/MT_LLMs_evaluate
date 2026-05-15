import argparse
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.pipeline import (
    PipelineExecutionError,
    PipelinePlan,
    build_pipeline_plan,
    execute_pipeline_plan,
    run_subprocess_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--directions",
        nargs="+",
        help="Translation directions to run. Defaults to all configured directions.",
    )
    parser.add_argument("--smoke-lines", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--infer-python")
    parser.add_argument("--eval-python")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    return parser


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(
    argv: list[str] | None = None,
    build_pipeline_plan_fn: Callable[..., PipelinePlan] | None = None,
    execute_pipeline_plan_fn: Callable[..., object] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.plan_only:
        raise ValueError("choose either --execute or --plan-only, not both")

    build = build_pipeline_plan if build_pipeline_plan_fn is None else build_pipeline_plan_fn
    execute = (
        execute_pipeline_plan
        if execute_pipeline_plan_fn is None
        else execute_pipeline_plan_fn
    )
    project_root = _resolve_project_root()
    infer_python_executable = args.infer_python or sys.executable
    eval_python_executable = args.eval_python or sys.executable
    plan = build(
        mode=args.mode,
        smoke_lines=args.smoke_lines,
        project_root=project_root,
        infer_python_executable=infer_python_executable,
        eval_python_executable=eval_python_executable,
        config_root=args.config_root,
        output_root=args.output_root,
        models_root=args.models_root,
        data_root=args.data_root,
        selected_models=args.models,
        selected_directions=args.directions,
    )

    if not args.execute:
        print(json.dumps(plan.to_dict(), ensure_ascii=True, indent=2))
        return 0

    try:
        execute(
            plan,
            project_root=project_root,
            command_runner=run_subprocess_command,
        )
    except PipelineExecutionError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
