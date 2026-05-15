from pathlib import Path
import csv
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.summary import summarize_metrics


def test_summarize_metrics_merges_bleu_and_comet_and_writes_csvs(tmp_path: Path):
    metrics_root = tmp_path / "metrics"
    bleu_dir = metrics_root / "bleu"
    comet_dir = metrics_root / "comet"
    output_dir = metrics_root / "summary"
    bleu_dir.mkdir(parents=True)
    comet_dir.mkdir(parents=True)

    (bleu_dir / "demo_model_my2zh.json").write_text(
        '{"model":"demo_model","direction":"my2zh","score":27.1,"signature":"sig-a"}',
        encoding="utf-8",
    )
    (bleu_dir / "demo_model_zh2my.json").write_text(
        '{"model":"demo_model","direction":"zh2my","score":19.5,"signature":"sig-b"}',
        encoding="utf-8",
    )
    (bleu_dir / "demo_model_my2en.json").write_text(
        '{"model":"demo_model","direction":"my2en","score":31.2,"signature":"sig-c"}',
        encoding="utf-8",
    )
    (bleu_dir / "demo_model_en2my.json").write_text(
        '{"model":"demo_model","direction":"en2my","score":17.4,"signature":"sig-d"}',
        encoding="utf-8",
    )
    (comet_dir / "demo_model_my2zh.json").write_text(
        '{"model":"demo_model","direction":"my2zh","comet_ref1":0.81,"comet_ref2":0.79,"comet_mean":0.8}',
        encoding="utf-8",
    )
    (comet_dir / "demo_model_my2en.json").write_text(
        '{"model":"demo_model","direction":"my2en","comet_ref1":0.71,"comet_ref2":0.69,"comet_mean":0.7}',
        encoding="utf-8",
    )

    result = summarize_metrics(metrics_root=metrics_root, output_dir=output_dir)

    assert result.main_csv_path == output_dir / "main_results.csv"
    assert result.detail_csv_path == output_dir / "detail_results.csv"
    assert result.main_rows == [
        {
            "model": "demo_model",
            "direction": "en2my",
            "bleu_score": "17.4",
            "comet_mean": "",
        },
        {
            "model": "demo_model",
            "direction": "my2en",
            "bleu_score": "31.2",
            "comet_mean": "0.7",
        },
        {
            "model": "demo_model",
            "direction": "my2zh",
            "bleu_score": "27.1",
            "comet_mean": "0.8",
        },
        {
            "model": "demo_model",
            "direction": "zh2my",
            "bleu_score": "19.5",
            "comet_mean": "",
        },
    ]
    with result.main_csv_path.open("r", encoding="utf-8", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    with result.detail_csv_path.open("r", encoding="utf-8", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))

    assert main_rows == result.main_rows
    assert detail_rows == [
        {
            "model": "demo_model",
            "direction": "en2my",
            "bleu_score": "17.4",
            "bleu_signature": "sig-d",
            "comet_ref1": "",
            "comet_ref2": "",
            "comet_mean": "",
        },
        {
            "model": "demo_model",
            "direction": "my2en",
            "bleu_score": "31.2",
            "bleu_signature": "sig-c",
            "comet_ref1": "0.71",
            "comet_ref2": "0.69",
            "comet_mean": "0.7",
        },
        {
            "model": "demo_model",
            "direction": "my2zh",
            "bleu_score": "27.1",
            "bleu_signature": "sig-a",
            "comet_ref1": "0.81",
            "comet_ref2": "0.79",
            "comet_mean": "0.8",
        },
        {
            "model": "demo_model",
            "direction": "zh2my",
            "bleu_score": "19.5",
            "bleu_signature": "sig-b",
            "comet_ref1": "",
            "comet_ref2": "",
            "comet_mean": "",
        },
    ]


def test_summarize_metrics_uses_filename_fallback_for_missing_model_direction(
    tmp_path: Path,
):
    metrics_root = tmp_path / "metrics"
    bleu_dir = metrics_root / "bleu"
    comet_dir = metrics_root / "comet"
    output_dir = metrics_root / "summary"
    bleu_dir.mkdir(parents=True)
    comet_dir.mkdir(parents=True)

    (bleu_dir / "alt_model_en2my.json").write_text(
        '{"score":18.2,"signature":"sig-c"}',
        encoding="utf-8",
    )
    (comet_dir / "alt_model_en2my.json").write_text(
        '{"comet_ref1":0.4,"comet_ref2":0.6,"comet_mean":0.5}',
        encoding="utf-8",
    )

    result = summarize_metrics(metrics_root=metrics_root, output_dir=output_dir)

    assert result.main_rows == [
        {
            "model": "alt_model",
            "direction": "en2my",
            "bleu_score": "18.2",
            "comet_mean": "0.5",
        }
    ]


def test_summarize_metrics_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/summarize_metrics.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--metrics-root" in result.stdout
    assert "--output-dir" in result.stdout


def test_summarize_metrics_cli_writes_paths_and_invokes_summary(tmp_path: Path, capsys):
    metrics_root = tmp_path / "metrics"
    output_dir = tmp_path / "summary"
    captured = {}

    class FakeResult:
        def __init__(self):
            self.main_csv_path = output_dir / "main_results.csv"
            self.detail_csv_path = output_dir / "detail_results.csv"
            self.main_rows = []
            self.detail_rows = []

    def fake_summarize_metrics(*, metrics_root, output_dir):
        captured["call"] = {
            "metrics_root": metrics_root,
            "output_dir": output_dir,
        }
        return FakeResult()

    module_globals = runpy.run_path("scripts/summarize_metrics.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--metrics-root",
            str(metrics_root),
            "--output-dir",
            str(output_dir),
        ],
        summarize_metrics_fn=fake_summarize_metrics,
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert captured == {
        "call": {
            "metrics_root": metrics_root,
            "output_dir": output_dir,
        }
    }
    assert "main_results.csv" in stdout
    assert "detail_results.csv" in stdout
