from pathlib import Path
import json
import runpy
import subprocess
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.comet_eval import (
    build_comet_model,
    run_two_reference_comet,
    score_comet_reference,
    score_comet_reference_with_model,
    summarize_comet_scores,
)


def test_summarize_comet_scores_returns_ref_scores_and_mean():
    result = summarize_comet_scores(0.4, 0.8)

    assert result == {
        "comet_ref1": 0.4,
        "comet_ref2": 0.8,
        "comet_mean": 0.6,
    }


def test_run_two_reference_comet_calls_runner_twice_and_aggregates(
    tmp_path: Path,
):
    hyp = tmp_path / "hyp.txt"
    src = tmp_path / "src.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    hyp.write_text("translated\n", encoding="utf-8")
    src.write_text("source\n", encoding="utf-8")
    ref1.write_text("reference one\n", encoding="utf-8")
    ref2.write_text("reference two\n", encoding="utf-8")
    calls = []

    def fake_runner(src_path, hyp_path, ref_path, model_name):
        calls.append(
            {
                "src_path": src_path,
                "hyp_path": hyp_path,
                "ref_path": ref_path,
                "model_name": model_name,
            }
        )
        return 0.25 if ref_path == ref1 else 0.75

    result = run_two_reference_comet(
        src,
        hyp,
        ref1,
        ref2,
        model_name="Unbabel/wmt22-comet-da",
        score_runner=fake_runner,
    )

    assert calls == [
        {
            "src_path": src,
            "hyp_path": hyp,
            "ref_path": ref1,
            "model_name": "Unbabel/wmt22-comet-da",
        },
        {
            "src_path": src,
            "hyp_path": hyp,
            "ref_path": ref2,
            "model_name": "Unbabel/wmt22-comet-da",
        },
    ]
    assert result == {
        "comet_ref1": 0.25,
        "comet_ref2": 0.75,
        "comet_mean": 0.5,
    }


def test_score_comet_reference_builds_line_aligned_samples(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.txt"
    hyp = tmp_path / "hyp.txt"
    ref = tmp_path / "ref.txt"
    src.write_text("source one\nsource two\n", encoding="utf-8")
    hyp.write_text("mt one\nmt two\n", encoding="utf-8")
    ref.write_text("ref one\nref two\n", encoding="utf-8")
    captured = {}

    class FakeModel:
        def predict(self, data, batch_size, gpus):
            captured["data"] = data
            captured["batch_size"] = batch_size
            captured["gpus"] = gpus
            return ([0.2, 0.4], 0.7)

    fake_comet = types.SimpleNamespace(
        download_model=lambda model_name: f"/fake/{model_name}",
        load_from_checkpoint=lambda model_path: FakeModel(),
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    result = score_comet_reference(src, hyp, ref, "Unbabel/wmt22-comet-da")

    assert captured == {
        "data": [
            {"src": "source one", "mt": "mt one", "ref": "ref one"},
            {"src": "source two", "mt": "mt two", "ref": "ref two"},
        ],
        "batch_size": 1,
        "gpus": 0,
    }
    assert result == 0.7


def test_build_comet_model_loads_local_checkpoint_without_downloading(
    tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    calls = {"download_model": 0, "load_from_checkpoint": []}

    def fake_download_model(model_name):
        calls["download_model"] += 1
        return f"/fake/{model_name}"

    def fake_load_from_checkpoint(model_path):
        calls["load_from_checkpoint"].append(model_path)
        return "model"

    fake_comet = types.SimpleNamespace(
        download_model=fake_download_model,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    assert build_comet_model(str(checkpoint)) == "model"
    assert calls == {
        "download_model": 0,
        "load_from_checkpoint": [str(checkpoint)],
    }


def test_build_comet_model_loads_local_snapshot_checkpoint(
    tmp_path: Path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    checkpoint = snapshot / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    calls = {"download_model": 0, "load_from_checkpoint": []}

    fake_comet = types.SimpleNamespace(
        download_model=lambda model_name: calls.__setitem__("download_model", 1),
        load_from_checkpoint=lambda model_path: calls["load_from_checkpoint"].append(model_path) or "model",
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    assert build_comet_model(str(snapshot)) == "model"
    assert calls == {
        "download_model": 0,
        "load_from_checkpoint": [str(checkpoint)],
    }


def test_score_comet_reference_rejects_misaligned_line_counts(
    tmp_path: Path, monkeypatch
):
    src = tmp_path / "src.txt"
    hyp = tmp_path / "hyp.txt"
    ref = tmp_path / "ref.txt"
    src.write_text("source one\nsource two\n", encoding="utf-8")
    hyp.write_text("mt one\n", encoding="utf-8")
    ref.write_text("ref one\nref two\n", encoding="utf-8")
    fake_comet = types.SimpleNamespace(
        download_model=lambda model_name: f"/fake/{model_name}",
        load_from_checkpoint=lambda model_path: object(),
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    try:
        score_comet_reference(src, hyp, ref, "Unbabel/wmt22-comet-da")
    except ValueError as exc:
        assert str(exc) == "source, hypothesis, and reference files must have aligned line counts"
    else:
        raise AssertionError("expected ValueError for misaligned input files")


def test_score_comet_reference_averages_segment_scores_without_system_score(
    tmp_path: Path, monkeypatch
):
    src = tmp_path / "src.txt"
    hyp = tmp_path / "hyp.txt"
    ref = tmp_path / "ref.txt"
    src.write_text("source one\nsource two\n", encoding="utf-8")
    hyp.write_text("mt one\nmt two\n", encoding="utf-8")
    ref.write_text("ref one\nref two\n", encoding="utf-8")

    class FakeModel:
        def predict(self, data, batch_size, gpus):
            return ([0.2, 0.4], None)

    fake_comet = types.SimpleNamespace(
        download_model=lambda model_name: f"/fake/{model_name}",
        load_from_checkpoint=lambda model_path: FakeModel(),
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    result = score_comet_reference(src, hyp, ref, "Unbabel/wmt22-comet-da")

    assert result == 0.3


def test_score_comet_reference_uses_prediction_system_score(tmp_path: Path):
    src = tmp_path / "src.txt"
    hyp = tmp_path / "hyp.txt"
    ref = tmp_path / "ref.txt"
    src.write_text("source one\nsource two\n", encoding="utf-8")
    hyp.write_text("mt one\nmt two\n", encoding="utf-8")
    ref.write_text("ref one\nref two\n", encoding="utf-8")

    class FakePrediction(dict):
        @property
        def scores(self):
            return self["scores"]

        @property
        def system_score(self):
            return self["system_score"]

    class FakeModel:
        def predict(self, data, batch_size, gpus):
            return FakePrediction(scores=[0.2, 0.4], system_score=0.8)

    result = score_comet_reference_with_model(FakeModel(), src, hyp, ref)

    assert result == 0.8


def test_run_two_reference_comet_default_path_loads_model_once(
    tmp_path: Path, monkeypatch
):
    src = tmp_path / "src.txt"
    hyp = tmp_path / "hyp.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    src.write_text("source one\nsource two\n", encoding="utf-8")
    hyp.write_text("mt one\nmt two\n", encoding="utf-8")
    ref1.write_text("ref one\nref two\n", encoding="utf-8")
    ref2.write_text("ref alt one\nref alt two\n", encoding="utf-8")
    calls = {"download_model": 0, "load_from_checkpoint": 0}

    class FakeModel:
        def __init__(self):
            self.predict_calls = []

        def predict(self, data, batch_size, gpus):
            self.predict_calls.append(data)
            return ([0.1, 0.2], 0.15) if data[0]["ref"] == "ref one" else ([0.3, 0.5], 0.4)

    fake_model = FakeModel()

    def fake_download_model(model_name):
        calls["download_model"] += 1
        return f"/fake/{model_name}"

    def fake_load_from_checkpoint(model_path):
        calls["load_from_checkpoint"] += 1
        return fake_model

    fake_comet = types.SimpleNamespace(
        download_model=fake_download_model,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    result = run_two_reference_comet(src, hyp, ref1, ref2)

    assert calls == {"download_model": 1, "load_from_checkpoint": 1}
    assert fake_model.predict_calls == [
        [
            {"src": "source one", "mt": "mt one", "ref": "ref one"},
            {"src": "source two", "mt": "mt two", "ref": "ref two"},
        ],
        [
            {"src": "source one", "mt": "mt one", "ref": "ref alt one"},
            {"src": "source two", "mt": "mt two", "ref": "ref alt two"},
        ],
    ]
    assert result == {
        "comet_ref1": 0.15,
        "comet_ref2": 0.4,
        "comet_mean": 0.275,
    }


def test_eval_comet_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/eval_comet.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--src" in result.stdout
    assert "--hyp" in result.stdout
    assert "--ref1" in result.stdout
    assert "--ref2" in result.stdout
    assert "--model" in result.stdout
    assert "--output" in result.stdout
    assert "--model-name" in result.stdout


def test_eval_comet_cli_writes_result_json_to_stdout(tmp_path: Path, capsys):
    hyp = tmp_path / "hyp.txt"
    src = tmp_path / "src.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    hyp.write_text("translated\n", encoding="utf-8")
    src.write_text("source\n", encoding="utf-8")
    ref1.write_text("reference one\n", encoding="utf-8")
    ref2.write_text("reference two\n", encoding="utf-8")
    captured = {}

    def fake_run_two_reference_comet(
        src_path,
        hyp_path,
        ref1_path,
        ref2_path,
        model_name,
    ):
        captured["call"] = {
            "src_path": src_path,
            "hyp_path": hyp_path,
            "ref1_path": ref1_path,
            "ref2_path": ref2_path,
            "model_name": model_name,
        }
        return {
            "comet_ref1": 0.1,
            "comet_ref2": 0.3,
            "comet_mean": 0.2,
        }

    module_globals = runpy.run_path("scripts/eval_comet.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--src",
            str(src),
            "--hyp",
            str(hyp),
            "--ref1",
            str(ref1),
            "--ref2",
            str(ref2),
            "--model",
            "Unbabel/wmt22-comet-da",
        ],
        run_two_reference_comet_fn=fake_run_two_reference_comet,
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert captured == {
        "call": {
            "src_path": src,
            "hyp_path": hyp,
            "ref1_path": ref1,
            "ref2_path": ref2,
            "model_name": "Unbabel/wmt22-comet-da",
        }
    }
    assert json.loads(stdout) == {
        "comet_ref1": 0.1,
        "comet_ref2": 0.3,
        "comet_mean": 0.2,
    }


def test_eval_comet_cli_writes_output_file_with_model_metadata(tmp_path: Path):
    hyp = tmp_path / "hyp.txt"
    src = tmp_path / "src.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    out = tmp_path / "comet.json"
    hyp.write_text("translated\n", encoding="utf-8")
    src.write_text("source\n", encoding="utf-8")
    ref1.write_text("reference one\n", encoding="utf-8")
    ref2.write_text("reference two\n", encoding="utf-8")

    def fake_run_two_reference_comet(
        src_path,
        hyp_path,
        ref1_path,
        ref2_path,
        model_name,
    ):
        return {
            "comet_ref1": 0.1,
            "comet_ref2": 0.3,
            "comet_mean": 0.2,
        }

    module_globals = runpy.run_path("scripts/eval_comet.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--src",
            str(src),
            "--hyp",
            str(hyp),
            "--ref1",
            str(ref1),
            "--ref2",
            str(ref2),
            "--model",
            "Unbabel/wmt22-comet-da",
            "--output",
            str(out),
            "--model-name",
            "hy_mt1_5_7b",
            "--direction",
            "my2zh",
        ],
        run_two_reference_comet_fn=fake_run_two_reference_comet,
    )

    assert exit_code == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "model": "hy_mt1_5_7b",
        "direction": "my2zh",
        "comet_ref1": 0.1,
        "comet_ref2": 0.3,
        "comet_mean": 0.2,
    }
