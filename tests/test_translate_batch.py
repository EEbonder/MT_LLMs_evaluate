import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.inference import (
    BackendSettings,
    TransformersTranslationBackend,
    render_prompt,
    translate_lines,
    translate_lines_limited,
)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def translate_batch(self, prompts):
        self.calls.append(list(prompts))
        return [f"OUT::{prompt}" for prompt in prompts]


class NewlineBackend:
    def translate_batch(self, prompts):
        return [f"line1\nline2::{prompt}" for prompt in prompts]


class ShortBackend:
    def translate_batch(self, prompts):
        return ["only-one-result"]


def test_render_prompt_inserts_source_text():
    assert render_prompt("Translate: {source}", "hello") == "Translate: hello"


def test_translate_lines_writes_one_output_per_input(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    backend = FakeBackend()

    translate_lines(
        input_path=src,
        output_path=out,
        backend=backend,
        prompt_template="Translate: {source}",
        batch_size=2,
        resume=False,
    )

    assert out.read_text(encoding="utf-8").splitlines() == [
        "OUT::Translate: a",
        "OUT::Translate: b",
    ]
    assert backend.calls == [["Translate: a", "Translate: b"]]


def test_translate_lines_resumes_from_existing_output(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\nc\n", encoding="utf-8")
    out = tmp_path / "nested" / "out.txt"
    out.parent.mkdir()
    out.write_text("done-a\n", encoding="utf-8")
    backend = FakeBackend()

    translate_lines(
        input_path=src,
        output_path=out,
        backend=backend,
        prompt_template="Translate: {source}",
        batch_size=2,
        resume=True,
    )

    assert out.read_text(encoding="utf-8").splitlines() == [
        "done-a",
        "OUT::Translate: b",
        "OUT::Translate: c",
    ]
    assert backend.calls == [["Translate: b", "Translate: c"]]


def test_translate_lines_limited_caps_input_lines_before_resume(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\nc\nd\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    backend = FakeBackend()

    translate_lines_limited(
        input_path=src,
        output_path=out,
        backend=backend,
        prompt_template="Translate: {source}",
        batch_size=2,
        resume=False,
        max_lines=3,
    )

    assert out.read_text(encoding="utf-8").splitlines() == [
        "OUT::Translate: a",
        "OUT::Translate: b",
        "OUT::Translate: c",
    ]
    assert backend.calls == [["Translate: a", "Translate: b"], ["Translate: c"]]


def test_translate_lines_normalizes_embedded_newlines_to_single_output_line(
    tmp_path: Path,
):
    src = tmp_path / "src.txt"
    src.write_text("a\n", encoding="utf-8")
    out = tmp_path / "out.txt"

    translate_lines(
        input_path=src,
        output_path=out,
        backend=NewlineBackend(),
        prompt_template="Translate: {source}",
        batch_size=1,
        resume=False,
    )

    assert out.read_text(encoding="utf-8").splitlines() == [
        "line1 line2::Translate: a"
    ]


def test_translate_lines_rejects_backend_result_count_mismatch(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\n", encoding="utf-8")
    out = tmp_path / "out.txt"

    with pytest.raises(ValueError, match="exactly one result per prompt"):
        translate_lines(
            input_path=src,
            output_path=out,
            backend=ShortBackend(),
            prompt_template="Translate: {source}",
            batch_size=2,
            resume=False,
        )


def test_translate_lines_rejects_non_positive_batch_size(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("a\n", encoding="utf-8")
    out = tmp_path / "out.txt"

    with pytest.raises(ValueError, match="batch_size must be positive"):
        translate_lines(
            input_path=src,
            output_path=out,
            backend=FakeBackend(),
            prompt_template="Translate: {source}",
            batch_size=0,
            resume=False,
        )


def test_translate_lines_rejects_resume_state_with_more_output_lines_than_source(
    tmp_path: Path,
):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    out.write_text("done-a\ndone-b\ndone-c\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume state is invalid"):
        translate_lines(
            input_path=src,
            output_path=out,
            backend=FakeBackend(),
            prompt_template="Translate: {source}",
            batch_size=1,
            resume=True,
        )


def test_translate_lines_rejects_resume_output_without_trailing_newline(
    tmp_path: Path,
):
    src = tmp_path / "src.txt"
    src.write_text("a\nb\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    out.write_text("done-a", encoding="utf-8")

    with pytest.raises(ValueError, match="resume state is invalid"):
        translate_lines(
            input_path=src,
            output_path=out,
            backend=FakeBackend(),
            prompt_template="Translate: {source}",
            batch_size=1,
            resume=True,
        )


def test_translate_batch_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/translate_batch.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model-name" in result.stdout
    assert "--direction" in result.stdout
    assert "--config-root" in result.stdout


def test_build_backend_uses_model_overrides_and_cli_runtime_overrides():
    module_globals = runpy.run_path("scripts/translate_batch.py", run_name="__test__")
    cfg = module_globals["load_runtime_config"](
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )
    captured = {}

    class BuiltBackend:
        pass

    backend = BuiltBackend()

    def fake_backend_factory(settings, model_loader=None, tokenizer_loader=None):
        captured["settings"] = settings
        return backend

    built = module_globals["build_backend"](
        cfg,
        model_name="qwen3_32b",
        model_path=Path("E:/models/qwen3"),
        quantization="8bit",
        dtype="bf16",
        attn_implementation="flash_attention_2",
        backend_factory=fake_backend_factory,
    )

    assert built is backend
    assert captured["settings"] == BackendSettings(
        model_name="qwen3_32b",
        model_path="E:/models/qwen3",
        hf_id="Qwen/Qwen3-32B",
        dtype="bf16",
        quantization="8bit",
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        device_map="auto",
        tokenizer_mode="chat",
        uses_chat_template=True,
        padding_side="left",
        max_input_length=4096,
        stop_strings=[],
        chat_add_generation_prompt=True,
        tokenizer_kwargs={},
        apply_chat_template_kwargs={"enable_thinking": False},
        load_mode="causal_lm",
        generation_config={
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "repetition_penalty": 1.0,
        },
    )


def test_build_backend_defaults_to_bf16_without_forcing_fallback_quantization():
    module_globals = runpy.run_path("scripts/translate_batch.py", run_name="__test__")
    cfg = module_globals["load_runtime_config"](
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )
    captured = {}

    def fake_backend_factory(settings, model_loader=None, tokenizer_loader=None):
        captured["settings"] = settings
        return object()

    module_globals["build_backend"](
        cfg,
        model_name="qwen3_32b",
        model_path=None,
        quantization=None,
        dtype=None,
        attn_implementation=None,
        backend_factory=fake_backend_factory,
    )

    assert captured["settings"].dtype == "bf16"
    assert captured["settings"].quantization is None


def test_translate_batch_cli_builds_backend_and_calls_translate_lines(
    tmp_path: Path,
):
    src = tmp_path / "src.txt"
    out = tmp_path / "out.txt"
    src.write_text("line\n", encoding="utf-8")
    log_json = tmp_path / "run.json"
    captured = {}

    class BuiltBackend:
        pass

    backend = BuiltBackend()

    def fake_build_backend_fn(*, runtime_config, args):
        captured["backend_args"] = {
            "runtime_config": runtime_config,
            "args": args,
        }
        return backend

    def fake_translate_lines_fn(
        *,
        input_path,
        output_path,
        backend,
        prompt_template,
        batch_size,
        resume,
        max_lines,
    ):
        captured["call"] = {
            "input_path": input_path,
            "output_path": output_path,
            "backend": backend,
            "prompt_template": prompt_template,
            "batch_size": batch_size,
            "resume": resume,
            "max_lines": max_lines,
        }

    module_globals = runpy.run_path("scripts/translate_batch.py", run_name="__test__")
    exit_code = module_globals["main"](
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--model-name",
            "hy_mt1_5_7b",
            "--direction",
            "my2zh",
            "--batch-size",
            "3",
            "--resume",
            "--max-lines",
            "10",
            "--log-json",
            str(log_json),
        ],
        build_backend_fn=fake_build_backend_fn,
        translate_lines_fn=fake_translate_lines_fn,
    )

    assert exit_code == 0
    assert captured["call"]["input_path"] == src
    assert captured["call"]["output_path"] == out
    assert captured["call"]["backend"] is backend
    assert captured["call"]["batch_size"] == 3
    assert captured["call"]["resume"] is True
    assert captured["call"]["max_lines"] == 10
    assert "将以下文本翻译为中文" in captured["call"]["prompt_template"]

    payload = json.loads(log_json.read_text(encoding="utf-8"))
    assert payload["model_name"] == "hy_mt1_5_7b"
    assert payload["direction"] == "my2zh"
    assert payload["batch_size"] == 3
    assert payload["prompt_template_key"] == "hy_mt_official_my2zh"


def test_translate_batch_cli_rejects_unsupported_quantization():
    module_globals = runpy.run_path("scripts/translate_batch.py", run_name="__test__")

    with pytest.raises(ValueError, match="unsupported quantization override"):
        module_globals["main"](
            [
                "--input",
                "in.txt",
                "--output",
                "out.txt",
                "--model-name",
                "hy_mt1_5_7b",
                "--direction",
                "my2zh",
                "--quantization",
                "4bit",
            ]
        )


def test_transformers_backend_drops_token_type_ids_before_generation():
    captured = {}

    class FakeInputIds:
        shape = (1, 3)

    class FakeGenerated:
        def __getitem__(self, key):
            return ["generated-token-ids"]

    class FakeBatchEncoding(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompts, **kwargs):
            return FakeBatchEncoding(
                {
                    "input_ids": FakeInputIds(),
                    "attention_mask": "mask",
                    "token_type_ids": "segments",
                }
            )

        def batch_decode(self, generated_only, skip_special_tokens):
            return ["translated text"]

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeGenerated()

    settings = BackendSettings(
        model_name="fake",
        model_path="/tmp/fake",
        hf_id="fake/repo",
        dtype="bf16",
        quantization=None,
        trust_remote_code=False,
        attn_implementation=None,
        device_map="auto",
        tokenizer_mode="plain",
        uses_chat_template=False,
        padding_side="right",
        max_input_length=2048,
        stop_strings=[],
        chat_add_generation_prompt=True,
        tokenizer_kwargs={},
        apply_chat_template_kwargs={},
        load_mode="causal_lm",
        generation_config={"max_new_tokens": 1},
    )
    backend = TransformersTranslationBackend(
        settings,
        model_loader=lambda *args, **kwargs: FakeModel(),
        tokenizer_loader=lambda *args, **kwargs: FakeTokenizer(),
    )

    assert backend.translate_batch(["prompt"]) == ["translated text"]
    assert "token_type_ids" not in captured["kwargs"]
    assert captured["kwargs"]["attention_mask"] == "mask"


def test_transformers_backend_normalizes_gemma4_extra_special_tokens():
    captured = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    settings = BackendSettings(
        model_name="gemma4_31b_it",
        model_path="/tmp/fake",
        hf_id="fake/repo",
        dtype="bf16",
        quantization=None,
        trust_remote_code=False,
        attn_implementation=None,
        device_map="auto",
        tokenizer_mode="chat",
        uses_chat_template=True,
        padding_side="left",
        max_input_length=4096,
        stop_strings=[],
        chat_add_generation_prompt=True,
        tokenizer_kwargs={},
        apply_chat_template_kwargs={},
        load_mode="causal_lm",
        generation_config={"max_new_tokens": 1},
    )

    def fake_tokenizer_loader(*args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeTokenizer()

    TransformersTranslationBackend(
        settings,
        model_loader=lambda *args, **kwargs: object(),
        tokenizer_loader=fake_tokenizer_loader,
    )

    assert captured["kwargs"]["extra_special_tokens"] == {
        "video_token": "<|video|>"
    }


def test_transformers_backend_uses_model_chat_generation_prompt_setting():
    captured = {}

    class FakeBatchEncoding(dict):
        def to(self, device):
            return self

    class FakeInputIds:
        shape = (1, 3)

    class FakeGenerated:
        def __getitem__(self, key):
            return ["generated-token-ids"]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return FakeBatchEncoding({"input_ids": FakeInputIds()})

        def batch_decode(self, generated_only, skip_special_tokens):
            return ["translated"]

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            return FakeGenerated()

    settings = BackendSettings(
        model_name="hy_mt1_5_7b",
        model_path="/tmp/fake",
        hf_id="fake/repo",
        dtype="bf16",
        quantization=None,
        trust_remote_code=False,
        attn_implementation=None,
        device_map="auto",
        tokenizer_mode="chat",
        uses_chat_template=True,
        padding_side="right",
        max_input_length=2048,
        stop_strings=[],
        chat_add_generation_prompt=False,
        tokenizer_kwargs={},
        apply_chat_template_kwargs={},
        load_mode="causal_lm",
        generation_config={"max_new_tokens": 1},
    )
    backend = TransformersTranslationBackend(
        settings,
        model_loader=lambda *args, **kwargs: FakeModel(),
        tokenizer_loader=lambda *args, **kwargs: FakeTokenizer(),
    )

    assert backend.translate_batch(["prompt"]) == ["translated"]
    assert captured["messages"] == [[{"role": "user", "content": "prompt"}]]
    assert captured["kwargs"]["add_generation_prompt"] is False


def test_transformers_backend_passes_model_tokenizer_kwargs():
    captured = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    settings = BackendSettings(
        model_name="niutrans_lmt_60_8b",
        model_path="/tmp/fake",
        hf_id="fake/repo",
        dtype="bf16",
        quantization=None,
        trust_remote_code=False,
        attn_implementation=None,
        device_map="auto",
        tokenizer_mode="chat",
        uses_chat_template=True,
        padding_side="left",
        max_input_length=4096,
        stop_strings=[],
        chat_add_generation_prompt=True,
        tokenizer_kwargs={"fix_mistral_regex": True},
        apply_chat_template_kwargs={},
        load_mode="causal_lm",
        generation_config={"max_new_tokens": 1},
    )

    def fake_tokenizer_loader(*args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeTokenizer()

    TransformersTranslationBackend(
        settings,
        model_loader=lambda *args, **kwargs: object(),
        tokenizer_loader=fake_tokenizer_loader,
    )

    assert captured["kwargs"]["fix_mistral_regex"] is True
