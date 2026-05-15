from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_DIRECTIONS = {"my2zh", "zh2my", "my2en", "en2my"}


@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    family: str
    dtype: str
    default_quantization: str | None
    fallback_quantization: str | None
    prompts: dict[str, str]
    load_mode: str
    trust_remote_code: bool
    attn_implementation: str | None
    device_map: str
    tokenizer_mode: str
    uses_chat_template: bool
    padding_side: str
    max_input_length: int | None
    stop_strings: list[str]
    chat_add_generation_prompt: bool
    tokenizer_kwargs: dict[str, Any]
    apply_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    models: dict[str, ModelConfig]
    prompts: dict[str, Any]
    generation: dict[str, Any]


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _require_key(mapping: dict, key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing required key '{key}'")
    return mapping[key]


def _require_non_empty_mapping(mapping: dict, label: str) -> dict:
    if not mapping:
        raise ValueError(f"{label} must not be empty")
    return mapping


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, label)


def _require_list_of_strings(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return list(value)


def _require_int_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _read_yaml(path: Path, label: str) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _require_mapping(data, label)


def _parse_model_config(name: str, raw: dict) -> ModelConfig:
    hf_id = _require_string(_require_key(raw, "hf_id", f"model '{name}'"), f"model '{name}' hf_id")
    family = _require_string(_require_key(raw, "family", f"model '{name}'"), f"model '{name}' family")
    dtype = _require_string(_require_key(raw, "dtype", f"model '{name}'"), f"model '{name}' dtype")
    prompts = _require_mapping(_require_key(raw, "prompts", f"model '{name}'"), f"model '{name}' prompts")
    prompt_mapping = {
        direction: _require_string(
            _require_key(prompts, direction, f"model '{name}' prompts"),
            f"model '{name}' prompt key '{direction}'",
        )
        for direction in sorted(VALID_DIRECTIONS)
    }
    chat_kwargs = raw.get("apply_chat_template_kwargs", {})
    if not isinstance(chat_kwargs, dict):
        raise ValueError(f"model '{name}' apply_chat_template_kwargs must contain a mapping")
    tokenizer_kwargs = raw.get("tokenizer_kwargs", {})
    if not isinstance(tokenizer_kwargs, dict):
        raise ValueError(f"model '{name}' tokenizer_kwargs must contain a mapping")
    return ModelConfig(
        hf_id=hf_id,
        family=family,
        dtype=dtype,
        default_quantization=_require_optional_string(
            raw.get("default_quantization"), f"model '{name}' default_quantization"
        ),
        fallback_quantization=_require_optional_string(
            raw.get("fallback_quantization"), f"model '{name}' fallback_quantization"
        ),
        prompts=prompt_mapping,
        load_mode=_require_string(_require_key(raw, "load_mode", f"model '{name}'"), f"model '{name}' load_mode"),
        trust_remote_code=_require_bool(raw.get("trust_remote_code", False), f"model '{name}' trust_remote_code"),
        attn_implementation=_require_optional_string(
            raw.get("attn_implementation"), f"model '{name}' attn_implementation"
        ),
        device_map=_require_string(raw.get("device_map", "auto"), f"model '{name}' device_map"),
        tokenizer_mode=_require_string(
            _require_key(raw, "tokenizer_mode", f"model '{name}'"),
            f"model '{name}' tokenizer_mode",
        ),
        uses_chat_template=_require_bool(
            _require_key(raw, "uses_chat_template", f"model '{name}'"),
            f"model '{name}' uses_chat_template",
        ),
        padding_side=_require_string(raw.get("padding_side", "right"), f"model '{name}' padding_side"),
        max_input_length=_require_int_or_none(raw.get("max_input_length"), f"model '{name}' max_input_length"),
        stop_strings=_require_list_of_strings(raw.get("stop_strings"), f"model '{name}' stop_strings"),
        chat_add_generation_prompt=_require_bool(
            raw.get("chat_add_generation_prompt", True),
            f"model '{name}' chat_add_generation_prompt",
        ),
        tokenizer_kwargs=dict(tokenizer_kwargs),
        apply_chat_template_kwargs=dict(chat_kwargs),
    )


def load_runtime_config(
    models_path: Path,
    prompts_path: Path,
    generation_path: Path,
) -> RuntimeConfig:
    raw_models = _read_yaml(models_path, models_path.name)
    prompts = _read_yaml(prompts_path, prompts_path.name)
    generation = _read_yaml(generation_path, generation_path.name)

    translation = _require_mapping(
        _require_key(prompts, "translation", prompts_path.name),
        f"{prompts_path.name} translation",
    )
    _require_mapping(
        _require_key(translation, "templates", f"{prompts_path.name} translation"),
        f"{prompts_path.name} translation templates",
    )
    _require_mapping(
        _require_key(generation, "defaults", generation_path.name),
        f"{generation_path.name} 'defaults'",
    )
    _require_mapping(
        generation.get("models", {}),
        f"{generation_path.name} 'models'",
    )

    raw_model_entries = _require_mapping(
        _require_key(raw_models, "models", models_path.name),
        f"{models_path.name} 'models'",
    )
    raw_model_entries = _require_non_empty_mapping(raw_model_entries, f"{models_path.name} 'models'")
    models = {
        name: _parse_model_config(name, _require_mapping(data, f"model '{name}'"))
        for name, data in raw_model_entries.items()
    }
    return RuntimeConfig(models=models, prompts=prompts, generation=generation)


def resolve_prompt_template(
    runtime_config: RuntimeConfig,
    model_name: str,
    direction: str,
) -> str:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"unknown translation direction: {direction}")
    model_config = runtime_config.models[model_name]
    template_key = model_config.prompts[direction]
    templates = runtime_config.prompts["translation"]["templates"]
    if template_key not in templates:
        raise ValueError(f"prompt template '{template_key}' is not defined")
    return _require_string(templates[template_key], f"prompt template '{template_key}'")


def resolve_prompt_template_key(
    runtime_config: RuntimeConfig,
    model_name: str,
    direction: str,
) -> str:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"unknown translation direction: {direction}")
    return runtime_config.models[model_name].prompts[direction]


def resolve_generation_config(runtime_config: RuntimeConfig, model_name: str) -> dict[str, Any]:
    defaults = dict(runtime_config.generation["defaults"])
    model_overrides = dict(runtime_config.generation.get("models", {}).get(model_name, {}))
    defaults.update(model_overrides)
    return defaults
