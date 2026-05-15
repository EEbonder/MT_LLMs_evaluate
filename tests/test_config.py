from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import (
    load_runtime_config,
    resolve_generation_config,
    resolve_prompt_template,
)


def test_load_runtime_config_contains_all_mainline_models():
    cfg = load_runtime_config(
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )
    assert sorted(cfg.models) == [
        "gemma3_27b_it",
        "gemma4_31b_it",
        "gemma4_moe_26b_a4b_it",
        "hy_mt1_5_7b",
        "niutrans_lmt_60_8b",
        "qwen3_32b",
        "qwen3_moe_30b_a3b",
    ]
    assert cfg.models["hy_mt1_5_7b"].hf_id == "tencent/HY-MT1.5-7B"
    assert cfg.models["niutrans_lmt_60_8b"].hf_id == "NiuTrans/LMT-60-8B"
    assert cfg.models["gemma4_31b_it"].hf_id == "google/gemma-4-31B-it"
    assert cfg.models["qwen3_32b"].hf_id == "Qwen/Qwen3-32B"
    assert cfg.models["gemma3_27b_it"].hf_id == "google/gemma-3-27b-it"
    assert cfg.models["qwen3_moe_30b_a3b"].hf_id == "Qwen/Qwen3-30B-A3B"
    assert cfg.models["gemma4_moe_26b_a4b_it"].hf_id == "google/gemma-4-26B-A4B-it"


def test_load_runtime_config_exposes_server_inference_fields():
    cfg = load_runtime_config(
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )

    hy = cfg.models["hy_mt1_5_7b"]
    assert hy.dtype == "bf16"
    assert hy.default_quantization is None
    assert hy.fallback_quantization is None
    assert hy.prompts == {
        "en2my": "hy_mt_official_en2my",
        "my2en": "hy_mt_official_my2en",
        "my2zh": "hy_mt_official_my2zh",
        "zh2my": "hy_mt_official_zh2my",
    }
    assert hy.load_mode == "causal_lm"
    assert hy.trust_remote_code is False
    assert hy.attn_implementation is None
    assert hy.device_map == "auto"
    assert hy.tokenizer_mode == "chat"
    assert hy.uses_chat_template is True
    assert hy.chat_add_generation_prompt is False
    assert hy.padding_side == "left"
    assert hy.max_input_length == 2048
    assert hy.stop_strings == []

    niutrans = cfg.models["niutrans_lmt_60_8b"]
    assert niutrans.tokenizer_kwargs == {"fix_mistral_regex": True}
    assert resolve_generation_config(cfg, "niutrans_lmt_60_8b") == {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "num_beams": 5,
    }

    gemma = cfg.models["gemma4_31b_it"]
    assert gemma.dtype == "bf16"
    assert gemma.default_quantization is None
    assert gemma.fallback_quantization == "8bit"
    assert gemma.prompts == {
        "en2my": "chat_translation_en2my_v1",
        "my2en": "chat_translation_my2en_v1",
        "my2zh": "chat_translation_my2zh_v1",
        "zh2my": "chat_translation_zh2my_v1",
    }
    assert gemma.tokenizer_mode == "chat"
    assert gemma.uses_chat_template is True
    assert gemma.apply_chat_template_kwargs == {"enable_thinking": False}
    assert gemma.stop_strings == ["<end_of_turn>"]

    gemma_moe = cfg.models["gemma4_moe_26b_a4b_it"]
    assert gemma_moe.apply_chat_template_kwargs == {"enable_thinking": False}


def test_resolve_prompt_template_uses_model_direction_mapping():
    cfg = load_runtime_config(
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )

    assert resolve_prompt_template(cfg, "hy_mt1_5_7b", "my2zh").startswith("将以下文本翻译为中文")
    assert "{source}" in resolve_prompt_template(cfg, "qwen3_32b", "zh2my")
    assert "English" in resolve_prompt_template(cfg, "gemma3_27b_it", "my2en")
    assert "Myanmar" in resolve_prompt_template(cfg, "gemma4_moe_26b_a4b_it", "en2my")


def test_resolve_generation_config_merges_model_overrides():
    cfg = load_runtime_config(
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )

    assert resolve_generation_config(cfg, "hy_mt1_5_7b") == {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "do_sample": True,
        "repetition_penalty": 1.05,
    }
    assert resolve_generation_config(cfg, "gemma4_31b_it") == {
        "max_new_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "do_sample": True,
        "repetition_penalty": 1.0,
    }
    assert resolve_generation_config(cfg, "qwen3_32b") == {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "do_sample": True,
        "repetition_penalty": 1.0,
    }
    assert resolve_generation_config(cfg, "gemma3_27b_it") == {
        "max_new_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "do_sample": True,
        "repetition_penalty": 1.0,
    }
    assert resolve_generation_config(cfg, "qwen3_moe_30b_a3b") == {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "do_sample": True,
        "repetition_penalty": 1.0,
    }


def test_load_runtime_config_rejects_empty_models_config(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text("", encoding="ascii")
    prompts_path.write_text("translation:\n  templates:\n    x: test\n", encoding="ascii")
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(ValueError, match="models.yaml must contain a mapping"):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_load_runtime_config_rejects_empty_model_registry(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text("models: {}\n", encoding="ascii")
    prompts_path.write_text("translation:\n  templates:\n    x: test\n", encoding="ascii")
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(ValueError, match="models.yaml 'models' must not be empty"):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_load_runtime_config_rejects_non_mapping_model_entry(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text("models:\n  hy_mt1_5_7b: not-a-mapping\n", encoding="ascii")
    prompts_path.write_text("translation:\n  templates:\n    x: test\n", encoding="ascii")
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(ValueError, match="model 'hy_mt1_5_7b' must contain a mapping"):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_load_runtime_config_requires_model_hf_id(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text(
        "models:\n"
        "  hy_mt1_5_7b:\n"
        "    family: hy_mt\n"
        "    dtype: bf16\n",
        encoding="ascii",
    )
    prompts_path.write_text("translation:\n  templates:\n    x: test\n", encoding="ascii")
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(
        ValueError, match="model 'hy_mt1_5_7b' is missing required key 'hf_id'"
    ):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_load_runtime_config_requires_translation_templates_mapping(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text(
        "models:\n"
        "  hy_mt1_5_7b:\n"
        "    hf_id: tencent/HY-MT1.5-7B\n"
        "    family: hy_mt\n"
        "    dtype: bf16\n"
        "    prompts:\n"
        "      my2zh: x\n"
        "      zh2my: x\n"
        "      my2en: x\n"
        "      en2my: x\n"
        "    load_mode: causal_lm\n"
        "    tokenizer_mode: plain\n"
        "    uses_chat_template: false\n",
        encoding="ascii",
    )
    prompts_path.write_text("translation:\n  other:\n    x: test\n", encoding="ascii")
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(
        ValueError, match="prompts.yaml translation is missing required key 'templates'"
    ):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_load_runtime_config_requires_generation_defaults_mapping(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text(
        "models:\n"
        "  hy_mt1_5_7b:\n"
        "    hf_id: tencent/HY-MT1.5-7B\n"
        "    family: hy_mt\n"
        "    dtype: bf16\n"
        "    prompts:\n"
        "      my2zh: x\n"
        "      zh2my: x\n"
        "      my2en: x\n"
        "      en2my: x\n"
        "    load_mode: causal_lm\n"
        "    tokenizer_mode: plain\n"
        "    uses_chat_template: false\n",
        encoding="ascii",
    )
    prompts_path.write_text("translation:\n  templates:\n    x: test\n", encoding="ascii")
    generation_path.write_text("other:\n  max_new_tokens: 16\n", encoding="ascii")

    with pytest.raises(
        ValueError, match="generation.yaml is missing required key 'defaults'"
    ):
        load_runtime_config(models_path, prompts_path, generation_path)


def test_resolve_prompt_template_rejects_unknown_direction(tmp_path: Path):
    models_path = tmp_path / "models.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    generation_path = tmp_path / "generation.yaml"

    models_path.write_text(
        "models:\n"
        "  test_model:\n"
        "    hf_id: org/model\n"
        "    family: test\n"
        "    dtype: bf16\n"
        "    prompts:\n"
        "      my2zh: p1\n"
        "      zh2my: p2\n"
        "      my2en: p3\n"
        "      en2my: p4\n"
        "    load_mode: causal_lm\n"
        "    tokenizer_mode: plain\n"
        "    uses_chat_template: false\n",
        encoding="ascii",
    )
    prompts_path.write_text(
        "translation:\n"
        "  templates:\n"
        "    p1: source={source}\n"
        "    p2: source={source}\n"
        "    p3: source={source}\n"
        "    p4: source={source}\n",
        encoding="ascii",
    )
    generation_path.write_text("defaults:\n  max_new_tokens: 16\n", encoding="ascii")
    cfg = load_runtime_config(models_path, prompts_path, generation_path)

    with pytest.raises(ValueError, match="unknown translation direction"):
        resolve_prompt_template(cfg, "test_model", "bad")
