from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Callable

from mt_eval.config import VALID_DIRECTIONS, load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"
DEFAULT_PROMPTS_CONFIG = PROJECT_ROOT / "configs" / "prompts.yaml"
DEFAULT_GENERATION_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_SUMMARY_DIR = DEFAULT_OUTPUT_ROOT / "metrics" / "summary"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEGMENTATION_BACKEND = "pyidaungsu"
DEFAULT_DIRECTIONS = ["my2zh", "zh2my", "my2en", "en2my"]
DEFAULT_COMET_MODEL = str(
    PROJECT_ROOT
    / "root-cache"
    / "huggingface"
    / "hub"
    / "models--Unbabel--wmt22-comet-da"
    / "snapshots"
    / "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
)


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    source_filename: str
    ref_filenames: tuple[str, str]
    target_is_myanmar: bool = False


DIRECTION_SPECS = {
    "my2zh": DirectionSpec(
        name="my2zh",
        source_filename="my.txt",
        ref_filenames=("my2zh1.txt", "my2zh2.txt"),
    ),
    "zh2my": DirectionSpec(
        name="zh2my",
        source_filename="zh.txt",
        ref_filenames=("zh2my1_seg.txt", "zh2my2_seg.txt"),
        target_is_myanmar=True,
    ),
    "my2en": DirectionSpec(
        name="my2en",
        source_filename="my.txt",
        ref_filenames=("my2en1.txt", "my2en2.txt"),
    ),
    "en2my": DirectionSpec(
        name="en2my",
        source_filename="en.txt",
        ref_filenames=("en2my1_seg.txt", "en2my2_seg.txt"),
        target_is_myanmar=True,
    ),
}


@dataclass(frozen=True)
class PipelineStep:
    name: str
    stage: str
    description: str
    command: list[str] | None = None
    callable_runner: Callable[[], object] | None = None

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "stage": self.stage,
            "description": self.description,
            "kind": "callable" if self.callable_runner is not None else "command",
        }
        if self.command is not None:
            payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True)
class PipelinePlan:
    mode: str
    models: list[str]
    summary_dir: Path
    steps: list[PipelineStep]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "models": list(self.models),
            "summary_dir": str(self.summary_dir),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PipelineStepResult:
    step_name: str
    stage: str
    detail: str


class PipelineExecutionError(RuntimeError):
    def __init__(self, step_name: str, message: str):
        super().__init__(f"{step_name}: {message}")
        self.step_name = step_name
        self.message = message


def _script_path(project_root: Path, script_name: str) -> str:
    return str(project_root / "scripts" / script_name)


def _select_models(
    *,
    models_config_path: Path,
    prompts_config_path: Path,
    generation_config_path: Path,
    selected_models: list[str] | None,
) -> list[str]:
    if selected_models is not None and not models_config_path.exists():
        return selected_models

    cfg = load_runtime_config(
        models_config_path,
        prompts_config_path,
        generation_config_path,
    )
    available_models = list(cfg.models.keys())
    if selected_models is None:
        return available_models

    unknown = [name for name in selected_models if name not in cfg.models]
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    return selected_models


def _select_directions(selected_directions: list[str] | None) -> list[str]:
    directions = list(DEFAULT_DIRECTIONS) if selected_directions is None else selected_directions
    unknown = [
        direction
        for direction in directions
        if direction not in VALID_DIRECTIONS or direction not in DIRECTION_SPECS
    ]
    if unknown:
        raise ValueError(f"unknown directions: {', '.join(unknown)}")
    return directions


def _append_optional_path(
    command: list[str],
    flag: str,
    path: Path | None,
) -> None:
    if path is not None:
        command.extend([flag, str(path)])


def _smoke_copy_step(
    *,
    direction: str,
    source_path: Path,
    target_path: Path,
    max_lines: int,
) -> PipelineStep:
    def _copy_limited_lines() -> str:
        lines = source_path.read_text(encoding="utf-8").splitlines()[:max_lines]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(f"{line}\n" for line in lines)
        target_path.write_text(text, encoding="utf-8")
        return str(target_path)

    return PipelineStep(
        name=f"prepare_{direction}_{target_path.stem}",
        stage="prepare",
        description=f"Prepare smoke input {target_path.name}",
        callable_runner=_copy_limited_lines,
    )


def _translation_step(
    *,
    project_root: Path,
    infer_python_executable: str,
    config_root: Path,
    models_root: Path,
    model_name: str,
    direction: str,
    input_path: Path,
    output_path: Path,
    smoke_lines: int | None,
) -> PipelineStep:
    command = [
        infer_python_executable,
        _script_path(project_root, "translate_batch.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model-name",
        model_name,
        "--model-path",
        str(models_root / model_name),
        "--direction",
        direction,
        "--config-root",
        str(config_root),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--resume",
    ]
    if smoke_lines is not None:
        command.extend(["--max-lines", str(smoke_lines)])
    return PipelineStep(
        name=f"translate_{model_name}_{direction}",
        stage="translate",
        description=f"Translate {direction} with {model_name}",
        command=command,
    )


def _build_steps(
    *,
    mode: str,
    smoke_lines: int,
    project_root: Path,
    infer_python_executable: str,
    eval_python_executable: str,
    config_root: Path,
    models_root: Path,
    output_root: Path,
    data_root: Path,
    models: list[str],
    directions: list[str],
) -> list[PipelineStep]:
    smoke_limit = smoke_lines if mode == "smoke" else None
    raw_dir = output_root / "translations" / "raw"
    segmented_dir = output_root / "translations" / "bleu_ready"
    metrics_bleu_dir = output_root / "metrics" / "bleu"
    metrics_comet_dir = output_root / "metrics" / "comet"
    summary_dir = output_root / "metrics" / "summary"
    smoke_root = output_root / "smoke_inputs"

    def source_path(direction: str) -> Path:
        spec = DIRECTION_SPECS[direction]
        if mode == "smoke":
            return smoke_root / direction / spec.source_filename
        return data_root / direction / spec.source_filename

    def ref_paths(direction: str) -> tuple[Path, Path]:
        spec = DIRECTION_SPECS[direction]
        root = smoke_root if mode == "smoke" else data_root
        return (
            root / direction / spec.ref_filenames[0],
            root / direction / spec.ref_filenames[1],
        )

    steps = [
        PipelineStep(
            name="download_models",
            stage="download",
            description="Download configured model snapshots",
            command=[
                infer_python_executable,
                _script_path(project_root, "download_models.py"),
                "--cache-dir",
                str(models_root),
                "--config-root",
                str(config_root),
                "--models",
                *models,
            ],
        )
    ]
    if mode == "smoke":
        for direction in directions:
            spec = DIRECTION_SPECS[direction]
            steps.append(
                _smoke_copy_step(
                    direction=direction,
                    source_path=data_root / direction / spec.source_filename,
                    target_path=source_path(direction),
                    max_lines=smoke_lines,
                )
            )
            target_ref1, target_ref2 = ref_paths(direction)
            steps.extend(
                [
                    _smoke_copy_step(
                        direction=direction,
                        source_path=data_root / direction / spec.ref_filenames[0],
                        target_path=target_ref1,
                        max_lines=smoke_lines,
                    ),
                    _smoke_copy_step(
                        direction=direction,
                        source_path=data_root / direction / spec.ref_filenames[1],
                        target_path=target_ref2,
                        max_lines=smoke_lines,
                    ),
                ]
            )

    for model_name in models:
        for direction in directions:
            spec = DIRECTION_SPECS[direction]
            raw_output = raw_dir / f"{model_name}_{direction}.txt"
            metric_hyp = raw_output
            if spec.target_is_myanmar:
                metric_hyp = segmented_dir / f"{model_name}_{direction}.seg.txt"
            ref1, ref2 = ref_paths(direction)

            steps.append(
                _translation_step(
                    project_root=project_root,
                    infer_python_executable=infer_python_executable,
                    config_root=config_root,
                    models_root=models_root,
                    model_name=model_name,
                    direction=direction,
                    input_path=source_path(direction),
                    output_path=raw_output,
                    smoke_lines=smoke_limit,
                )
            )
            if spec.target_is_myanmar:
                steps.append(
                    PipelineStep(
                        name=f"segment_{model_name}_{direction}",
                        stage="segment",
                        description=f"Segment {direction} output for {model_name}",
                        command=[
                            eval_python_executable,
                            _script_path(project_root, "segment_myanmar.py"),
                            "--backend",
                            DEFAULT_SEGMENTATION_BACKEND,
                            "--input",
                            str(raw_output),
                            "--output",
                            str(metric_hyp),
                        ],
                    )
                )

            steps.extend(
                [
                    PipelineStep(
                        name=f"validate_{model_name}_{direction}",
                        stage="validate",
                        description=f"Validate {direction} output for {model_name}",
                        command=[
                            eval_python_executable,
                            _script_path(project_root, "validate_outputs.py"),
                            "--source",
                            str(source_path(direction)),
                            "--output",
                            str(raw_output),
                        ],
                    ),
                    PipelineStep(
                        name=f"bleu_{model_name}_{direction}",
                        stage="bleu",
                        description=f"Compute {direction} BLEU for {model_name}",
                        command=[
                            eval_python_executable,
                            _script_path(project_root, "eval_bleu.py"),
                            "--direction",
                            direction,
                            "--hyp",
                            str(metric_hyp),
                            "--refs",
                            str(ref1),
                            str(ref2),
                            "--output",
                            str(metrics_bleu_dir / f"{model_name}_{direction}.json"),
                            "--model-name",
                            model_name,
                        ],
                    ),
                    PipelineStep(
                        name=f"comet_{model_name}_{direction}",
                        stage="comet",
                        description=f"Compute {direction} COMET for {model_name}",
                        command=[
                            eval_python_executable,
                            _script_path(project_root, "eval_comet.py"),
                            "--src",
                            str(source_path(direction)),
                            "--hyp",
                            str(metric_hyp),
                            "--ref1",
                            str(ref1),
                            "--ref2",
                            str(ref2),
                            "--model",
                            DEFAULT_COMET_MODEL,
                            "--output",
                            str(metrics_comet_dir / f"{model_name}_{direction}.json"),
                            "--model-name",
                            model_name,
                            "--direction",
                            direction,
                        ],
                    ),
                ]
            )

    steps.append(
        PipelineStep(
            name="summarize_metrics",
            stage="summary",
            description="Merge metric JSON files into summary CSVs",
            command=[
                eval_python_executable,
                _script_path(project_root, "summarize_metrics.py"),
                "--metrics-root",
                str(output_root / "metrics"),
                "--output-dir",
                str(summary_dir),
            ],
        )
    )
    return steps


def build_pipeline_plan(
    *,
    mode: str,
    smoke_lines: int,
    project_root: Path = PROJECT_ROOT,
    infer_python_executable: str = "python",
    eval_python_executable: str = "python",
    config_root: Path | None = None,
    output_root: Path | None = None,
    models_root: Path | None = None,
    data_root: Path | None = None,
    models_config_path: Path | None = None,
    prompts_config_path: Path | None = None,
    generation_config_path: Path | None = None,
    selected_models: list[str] | None = None,
    selected_directions: list[str] | None = None,
) -> PipelinePlan:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unknown mode: {mode}")
    if smoke_lines <= 0:
        raise ValueError("smoke_lines must be positive")

    resolved_config_root = config_root or (project_root / "configs")
    resolved_models_config = models_config_path or (
        resolved_config_root / "models.yaml"
    )
    resolved_prompts_config = prompts_config_path or (
        resolved_config_root / "prompts.yaml"
    )
    resolved_generation_config = generation_config_path or (
        resolved_config_root / "generation.yaml"
    )
    resolved_output_root = output_root or (project_root / "outputs" / mode)
    resolved_models_root = models_root or (project_root / "models")
    resolved_data_root = data_root or (project_root / "data" / "test")

    models = _select_models(
        models_config_path=resolved_models_config,
        prompts_config_path=resolved_prompts_config,
        generation_config_path=resolved_generation_config,
        selected_models=selected_models,
    )
    directions = _select_directions(selected_directions)
    steps = _build_steps(
        mode=mode,
        smoke_lines=smoke_lines,
        project_root=project_root,
        infer_python_executable=infer_python_executable,
        eval_python_executable=eval_python_executable,
        config_root=resolved_config_root,
        models_root=resolved_models_root,
        output_root=resolved_output_root,
        data_root=resolved_data_root,
        models=models,
        directions=directions,
    )
    return PipelinePlan(
        mode=mode,
        models=models,
        summary_dir=resolved_output_root / "metrics" / "summary",
        steps=steps,
    )


def run_subprocess_command(command: list[str], cwd: Path) -> None:
    env = dict(os.environ)
    project_cache = PROJECT_ROOT / "root-cache"
    project_hf_home = project_cache / "huggingface"
    env.setdefault("HF_HOME", str(project_hf_home))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(project_hf_home / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(project_hf_home / "hub"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    subprocess.run(command, cwd=cwd, check=True, env=env)


def execute_pipeline_plan(
    plan: PipelinePlan,
    *,
    project_root: Path,
    command_runner: Callable[[list[str], Path], None] = run_subprocess_command,
) -> list[PipelineStepResult]:
    results = []
    for step in plan.steps:
        try:
            if step.command is not None:
                command_runner(step.command, project_root)
                detail = "command completed"
            elif step.callable_runner is not None:
                detail = str(step.callable_runner())
            else:
                raise PipelineExecutionError(step.name, "step has no executable action")
        except PipelineExecutionError:
            raise
        except subprocess.CalledProcessError as error:
            raise PipelineExecutionError(
                step.name,
                f"command failed with exit code {error.returncode}",
            ) from error
        except Exception as error:
            raise PipelineExecutionError(step.name, str(error)) from error

        results.append(
            PipelineStepResult(
                step_name=step.name,
                stage=step.stage,
                detail=detail,
            )
        )
    return results
