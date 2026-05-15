from pathlib import Path
import json
import runpy
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.pipeline import (
    PipelineExecutionError,
    PipelinePlan,
    PipelineStep,
    build_pipeline_plan,
    execute_pipeline_plan,
)


def test_build_pipeline_plan_for_smoke_mode_includes_max_lines_and_summary_step():
    plan = build_pipeline_plan(
        mode="smoke",
        smoke_lines=12,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
    )

    assert plan.mode == "smoke"
    assert plan.models == ["demo_model"]
    assert plan.summary_dir == Path("/repo/outputs/smoke/metrics/summary")
    translate_commands = [
        step.command
        for step in plan.steps
        if step.stage == "translate"
    ]
    assert len(translate_commands) == 4
    assert all("--max-lines" in command for command in translate_commands)
    assert all("12" in command for command in translate_commands)
    assert all("--model-name" in command for command in translate_commands)
    assert all("--direction" in command for command in translate_commands)
    assert all("--model-path" in command for command in translate_commands)
    assert all("--config-root" in command for command in translate_commands)
    assert all(command[0] == "python-infer" for command in translate_commands)
    assert plan.steps[-1].name == "summarize_metrics"
    assert plan.steps[-1].command[-2:] == [
        "--output-dir",
        str(Path("/repo/outputs/smoke/metrics/summary")),
    ]


def test_build_pipeline_plan_for_full_mode_omits_max_lines():
    plan = build_pipeline_plan(
        mode="full",
        smoke_lines=12,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
    )

    translate_commands = [
        step.command
        for step in plan.steps
        if step.stage == "translate"
    ]
    assert translate_commands
    assert all("--max-lines" not in command for command in translate_commands)
    assert all("--model-name" in command for command in translate_commands)


def test_build_pipeline_plan_uses_smoke_specific_validation_sources():
    plan = build_pipeline_plan(
        mode="smoke",
        smoke_lines=7,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
    )

    validation_commands = [
        step.command
        for step in plan.steps
        if step.stage == "validate"
    ]
    assert validation_commands
    assert all(
        Path(command[command.index("--source") + 1]).as_posix().startswith(
            "/repo/outputs/smoke/smoke_inputs/"
        )
        for command in validation_commands
    )


def test_build_pipeline_plan_writes_metric_outputs_for_bleu_and_comet():
    plan = build_pipeline_plan(
        mode="full",
        smoke_lines=10,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
    )

    bleu_commands = [step.command for step in plan.steps if step.stage == "bleu"]
    comet_commands = [step.command for step in plan.steps if step.stage == "comet"]
    assert bleu_commands
    assert comet_commands
    assert all("--output" in command for command in bleu_commands)
    assert all("--model-name" in command for command in bleu_commands)
    assert all("--output" in command for command in comet_commands)
    assert all("--model-name" in command for command in comet_commands)
    assert all("--direction" in command for command in comet_commands)
    assert any(command[-1] == "zh2my" for command in comet_commands)
    assert any(command[-1] == "en2my" for command in comet_commands)
    assert all(command[0] == "python-eval" for command in bleu_commands + comet_commands)


def test_build_pipeline_plan_download_step_uses_model_filter_and_config_root():
    plan = build_pipeline_plan(
        mode="full",
        smoke_lines=10,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        config_root=Path("/custom-configs"),
        selected_models=["demo_model"],
    )

    download_step = plan.steps[0]
    assert download_step.name == "download_models"
    assert "--config-root" in download_step.command
    assert Path(
        download_step.command[download_step.command.index("--config-root") + 1]
    ).as_posix() == Path("/custom-configs").as_posix()
    assert "--models" in download_step.command
    assert download_step.command[-1] == "demo_model"


def test_build_pipeline_plan_segment_step_uses_formal_backend():
    plan = build_pipeline_plan(
        mode="full",
        smoke_lines=10,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
    )

    segment_commands = [step.command for step in plan.steps if step.stage == "segment"]
    assert len(segment_commands) == 2
    assert all("--backend" in command for command in segment_commands)
    assert all(command[command.index("--backend") + 1] == "pyidaungsu" for command in segment_commands)
    assert {Path(command[command.index("--output") + 1]).name for command in segment_commands} == {
        "demo_model_en2my.seg.txt",
        "demo_model_zh2my.seg.txt",
    }


def test_build_pipeline_plan_supports_direction_filter():
    plan = build_pipeline_plan(
        mode="full",
        smoke_lines=10,
        project_root=Path("/repo"),
        infer_python_executable="python-infer",
        eval_python_executable="python-eval",
        selected_models=["demo_model"],
        selected_directions=["my2en", "en2my"],
    )

    translate_commands = [step.command for step in plan.steps if step.stage == "translate"]
    assert [
        command[command.index("--direction") + 1]
        for command in translate_commands
    ] == ["my2en", "en2my"]
    segment_commands = [step.command for step in plan.steps if step.stage == "segment"]
    assert len(segment_commands) == 1
    assert Path(segment_commands[0][segment_commands[0].index("--input") + 1]).name == "demo_model_en2my.txt"


def test_build_pipeline_plan_rejects_unknown_direction():
    with pytest.raises(ValueError, match="unknown directions"):
        build_pipeline_plan(
            mode="full",
            smoke_lines=10,
            project_root=Path("/repo"),
            infer_python_executable="python-infer",
            eval_python_executable="python-eval",
            selected_models=["demo_model"],
            selected_directions=["bad"],
        )


def test_build_pipeline_plan_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        build_pipeline_plan(
            mode="debug",
            smoke_lines=10,
            project_root=Path("/repo"),
            infer_python_executable="python-infer",
            eval_python_executable="python-eval",
        )


def test_execute_pipeline_plan_runs_steps_in_order_with_runner():
    plan = PipelinePlan(
        mode="smoke",
        models=["demo_model"],
        summary_dir=Path("/repo/outputs/metrics/summary"),
        steps=[
            PipelineStep(
                name="download_models",
                stage="download",
                description="Download model snapshots",
                command=["python", "scripts/download_models.py"],
            ),
            PipelineStep(
                name="custom_python_step",
                stage="summary",
                description="Custom callable",
                callable_runner=lambda: "ok",
            ),
        ],
    )
    calls = []

    def fake_command_runner(command, cwd):
        calls.append({"command": command, "cwd": cwd})

    results = execute_pipeline_plan(
        plan,
        project_root=Path("/repo"),
        command_runner=fake_command_runner,
    )

    assert calls == [
        {
            "command": ["python", "scripts/download_models.py"],
            "cwd": Path("/repo"),
        }
    ]
    assert [result.step_name for result in results] == [
        "download_models",
        "custom_python_step",
    ]
    assert results[1].detail == "ok"


def test_execute_pipeline_plan_fails_fast_on_runner_error():
    plan = PipelinePlan(
        mode="full",
        models=["demo_model"],
        summary_dir=Path("/repo/outputs/metrics/summary"),
        steps=[
            PipelineStep(
                name="translate_my2zh",
                stage="translate",
                description="Translate my2zh",
                command=["python", "scripts/translate_batch.py"],
            ),
            PipelineStep(
                name="unreached_step",
                stage="summary",
                description="Should not run",
                command=["python", "scripts/summarize_metrics.py"],
            ),
        ],
    )
    calls = []

    def fake_command_runner(command, cwd):
        calls.append(command)
        raise subprocess.CalledProcessError(returncode=2, cmd=command)

    with pytest.raises(PipelineExecutionError, match="translate_my2zh"):
        execute_pipeline_plan(
            plan,
            project_root=Path("/repo"),
            command_runner=fake_command_runner,
        )

    assert calls == [["python", "scripts/translate_batch.py"]]


def test_run_pipeline_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "--plan-only" in result.stdout
    assert "--execute" in result.stdout


def test_run_pipeline_cli_writes_structured_plan_json(capsys):
    captured = {}
    fake_plan = PipelinePlan(
        mode="smoke",
        models=["demo_model"],
        summary_dir=Path("/repo/outputs/metrics/summary"),
        steps=[
            PipelineStep(
                name="download_models",
                stage="download",
                description="Download model snapshots",
                command=["python", "scripts/download_models.py"],
            )
        ],
    )

    def fake_build_pipeline_plan(**kwargs):
        captured["kwargs"] = kwargs
        return fake_plan

    module_globals = runpy.run_path("scripts/run_pipeline.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--mode",
            "smoke",
            "--smoke-lines",
            "9",
            "--models",
            "demo_model",
            "--directions",
            "my2en",
            "en2my",
            "--infer-python",
            "python-infer",
            "--eval-python",
            "python-eval",
        ],
        build_pipeline_plan_fn=fake_build_pipeline_plan,
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert captured["kwargs"]["mode"] == "smoke"
    assert captured["kwargs"]["smoke_lines"] == 9
    assert captured["kwargs"]["selected_models"] == ["demo_model"]
    assert captured["kwargs"]["selected_directions"] == ["my2en", "en2my"]
    assert captured["kwargs"]["infer_python_executable"] == "python-infer"
    assert captured["kwargs"]["eval_python_executable"] == "python-eval"
    payload = json.loads(stdout)
    assert payload["mode"] == "smoke"
    assert payload["models"] == ["demo_model"]
    assert payload["steps"][0]["name"] == "download_models"


def test_run_pipeline_cli_execute_mode_invokes_executor():
    captured = {}
    fake_plan = PipelinePlan(
        mode="full",
        models=["demo_model"],
        summary_dir=Path("/repo/outputs/metrics/summary"),
        steps=[],
    )

    def fake_build_pipeline_plan(**kwargs):
        captured["build_kwargs"] = kwargs
        return fake_plan

    def fake_execute_pipeline_plan(plan, *, project_root, command_runner):
        captured["execute"] = {
            "plan": plan,
            "project_root": project_root,
            "command_runner": command_runner,
        }
        return []

    module_globals = runpy.run_path("scripts/run_pipeline.py", run_name="__test__")
    exit_code = module_globals["main"](
        ["--mode", "full", "--execute", "--infer-python", "python-infer", "--eval-python", "python-eval"],
        build_pipeline_plan_fn=fake_build_pipeline_plan,
        execute_pipeline_plan_fn=fake_execute_pipeline_plan,
    )

    assert exit_code == 0
    assert captured["build_kwargs"]["mode"] == "full"
    assert captured["execute"]["plan"] is fake_plan
    assert captured["execute"]["project_root"] == Path.cwd()
    assert callable(captured["execute"]["command_runner"])


def test_run_pipeline_cli_returns_non_zero_on_execution_failure(capsys):
    fake_plan = PipelinePlan(
        mode="full",
        models=["demo_model"],
        summary_dir=Path("/repo/outputs/metrics/summary"),
        steps=[],
    )

    def fake_build_pipeline_plan(**kwargs):
        return fake_plan

    def fake_execute_pipeline_plan(plan, *, project_root, command_runner):
        raise PipelineExecutionError(
            step_name="translate_my2zh",
            message="step failed",
        )

    module_globals = runpy.run_path("scripts/run_pipeline.py", run_name="__test__")
    exit_code = module_globals["main"](
        ["--mode", "full", "--execute"],
        build_pipeline_plan_fn=fake_build_pipeline_plan,
        execute_pipeline_plan_fn=fake_execute_pipeline_plan,
    )
    stderr = capsys.readouterr().err

    assert exit_code == 1
    assert "translate_my2zh" in stderr


def test_run_pipeline_uses_repo_root_not_cwd():
    module_globals = runpy.run_path("scripts/run_pipeline.py", run_name="__test__")
    project_root = module_globals["_resolve_project_root"]()

    assert project_root == Path("scripts/run_pipeline.py").resolve().parents[1]
