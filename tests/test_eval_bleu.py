from pathlib import Path
import json
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.bleu_eval import compute_bleu


class FakeBleuScore:
    def __init__(self, score: float, signature: str):
        self.score = score
        self._signature = signature

    def format(self, signature: bool = False) -> str:
        if signature:
            return self._signature
        return f"BLEU = {self.score}"


class FakeSacrebleu:
    def __init__(self, callback):
        self.corpus_bleu = callback


def test_compute_bleu_uses_zh_tokenizer_for_my2zh(tmp_path: Path, monkeypatch):
    hyp = tmp_path / "hyp.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    hyp.write_text("alpha\n", encoding="utf-8")
    ref1.write_text("alpha\n", encoding="utf-8")
    ref2.write_text("alpha beta\n", encoding="utf-8")
    captured = {}

    def fake_corpus_bleu(hypotheses, references, tokenize):
        captured["hypotheses"] = hypotheses
        captured["references"] = references
        captured["tokenize"] = tokenize
        return FakeBleuScore(100.0, "tok:zh|v:test")

    monkeypatch.setattr(
        "mt_eval.bleu_eval.sacrebleu",
        FakeSacrebleu(fake_corpus_bleu),
    )

    result = compute_bleu("my2zh", hyp, [ref1, ref2])

    assert captured == {
        "hypotheses": ["alpha"],
        "references": [["alpha"], ["alpha beta"]],
        "tokenize": "zh",
    }
    assert result == {
        "direction": "my2zh",
        "tokenizer": "zh",
        "score": 100.0,
        "signature": "tok:zh|v:test",
    }


def test_compute_bleu_uses_none_tokenizer_for_zh2my(tmp_path: Path, monkeypatch):
    hyp = tmp_path / "hyp.seg.txt"
    ref1 = tmp_path / "ref1.seg.txt"
    ref2 = tmp_path / "ref2.seg.txt"
    hyp.write_text("ka kha ga\n", encoding="utf-8")
    ref1.write_text("ka kha ga\n", encoding="utf-8")
    ref2.write_text("ka kha\n", encoding="utf-8")
    captured = {}

    def fake_corpus_bleu(hypotheses, references, tokenize):
        captured["hypotheses"] = hypotheses
        captured["references"] = references
        captured["tokenize"] = tokenize
        return FakeBleuScore(55.5, "tok:none|v:test")

    monkeypatch.setattr(
        "mt_eval.bleu_eval.sacrebleu",
        FakeSacrebleu(fake_corpus_bleu),
    )

    result = compute_bleu("zh2my", hyp, [ref1, ref2])

    assert captured["tokenize"] == "none"
    assert result["direction"] == "zh2my"
    assert result["tokenizer"] == "none"
    assert result["score"] == 55.5
    assert result["signature"] == "tok:none|v:test"


def test_compute_bleu_uses_13a_tokenizer_for_my2en(tmp_path: Path, monkeypatch):
    hyp = tmp_path / "hyp.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    hyp.write_text("hello world\n", encoding="utf-8")
    ref1.write_text("hello world\n", encoding="utf-8")
    ref2.write_text("hello\n", encoding="utf-8")
    captured = {}

    def fake_corpus_bleu(hypotheses, references, tokenize):
        captured["tokenize"] = tokenize
        return FakeBleuScore(66.6, "tok:13a|v:test")

    monkeypatch.setattr(
        "mt_eval.bleu_eval.sacrebleu",
        FakeSacrebleu(fake_corpus_bleu),
    )

    result = compute_bleu("my2en", hyp, [ref1, ref2])

    assert captured["tokenize"] == "13a"
    assert result["direction"] == "my2en"
    assert result["tokenizer"] == "13a"


def test_compute_bleu_uses_none_tokenizer_for_en2my(tmp_path: Path, monkeypatch):
    hyp = tmp_path / "hyp.seg.txt"
    ref1 = tmp_path / "ref1.seg.txt"
    ref2 = tmp_path / "ref2.seg.txt"
    hyp.write_text("ka kha ga\n", encoding="utf-8")
    ref1.write_text("ka kha ga\n", encoding="utf-8")
    ref2.write_text("ka kha\n", encoding="utf-8")
    captured = {}

    def fake_corpus_bleu(hypotheses, references, tokenize):
        captured["tokenize"] = tokenize
        return FakeBleuScore(44.4, "tok:none|v:test")

    monkeypatch.setattr(
        "mt_eval.bleu_eval.sacrebleu",
        FakeSacrebleu(fake_corpus_bleu),
    )

    result = compute_bleu("en2my", hyp, [ref1, ref2])

    assert captured["tokenize"] == "none"
    assert result["direction"] == "en2my"
    assert result["tokenizer"] == "none"


def test_eval_bleu_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/eval_bleu.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--direction" in result.stdout
    assert "--hyp" in result.stdout
    assert "--refs" in result.stdout
    assert "--output" in result.stdout
    assert "--model-name" in result.stdout


def test_eval_bleu_cli_writes_result_json_to_stdout(tmp_path: Path, capsys):
    hyp = tmp_path / "hyp.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    hyp.write_text("alpha\n", encoding="utf-8")
    ref1.write_text("alpha\n", encoding="utf-8")
    ref2.write_text("beta\n", encoding="utf-8")
    captured = {}

    def fake_compute_bleu(direction, hyp_path, ref_paths):
        captured["call"] = {
            "direction": direction,
            "hyp_path": hyp_path,
            "ref_paths": ref_paths,
        }
        return {
            "direction": direction,
            "tokenizer": "zh",
            "score": 12.34,
            "signature": "tok:zh|v:test",
        }

    module_globals = runpy.run_path("scripts/eval_bleu.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--direction",
            "my2zh",
            "--hyp",
            str(hyp),
            "--refs",
            str(ref1),
            str(ref2),
        ],
        compute_bleu_fn=fake_compute_bleu,
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert captured == {
        "call": {
            "direction": "my2zh",
            "hyp_path": hyp,
            "ref_paths": [ref1, ref2],
        }
    }
    assert json.loads(stdout) == {
        "direction": "my2zh",
        "tokenizer": "zh",
        "score": 12.34,
        "signature": "tok:zh|v:test",
    }


def test_eval_bleu_cli_writes_output_file_with_model_metadata(tmp_path: Path):
    hyp = tmp_path / "hyp.txt"
    ref1 = tmp_path / "ref1.txt"
    ref2 = tmp_path / "ref2.txt"
    out = tmp_path / "bleu.json"
    hyp.write_text("alpha\n", encoding="utf-8")
    ref1.write_text("alpha\n", encoding="utf-8")
    ref2.write_text("beta\n", encoding="utf-8")

    def fake_compute_bleu(direction, hyp_path, ref_paths):
        return {
            "direction": direction,
            "tokenizer": "zh",
            "score": 12.34,
            "signature": "tok:zh|v:test",
        }

    module_globals = runpy.run_path("scripts/eval_bleu.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--direction",
            "my2zh",
            "--hyp",
            str(hyp),
            "--refs",
            str(ref1),
            str(ref2),
            "--output",
            str(out),
            "--model-name",
            "hy_mt1_5_7b",
        ],
        compute_bleu_fn=fake_compute_bleu,
    )

    assert exit_code == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "model": "hy_mt1_5_7b",
        "direction": "my2zh",
        "tokenizer": "zh",
        "score": 12.34,
        "signature": "tok:zh|v:test",
    }
