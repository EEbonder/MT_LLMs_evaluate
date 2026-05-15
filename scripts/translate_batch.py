import argparse
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import (
    VALID_DIRECTIONS,
    load_runtime_config,
    resolve_generation_config,
    resolve_prompt_template,
    resolve_prompt_template_key,
)
from mt_eval.inference import (
    BackendSettings,
    TransformersTranslationBackend,
    translate_lines_limited,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--direction", choices=sorted(VALID_DIRECTIONS), required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--quantization")
    parser.add_argument("--dtype")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--log-json", type=Path)
    return parser


def _resolve_config_root(config_root: Path | None) -> Path:
    if config_root is not None:
        return config_root
    return Path(__file__).resolve().parents[1] / "configs"


def _build_backend_settings(runtime_config, args) -> BackendSettings:
    model_config = runtime_config.models[args.model_name]
    quantization = args.quantization
    if quantization == "4bit":
        raise ValueError("unsupported quantization override: 4bit")
    if quantization is None:
        quantization = model_config.default_quantization
    elif quantization not in {"8bit", "none"}:
        raise ValueError(f"unsupported quantization override: {quantization}")
    if quantization == "none":
        quantization = None

    dtype = args.dtype or model_config.dtype
    attn_implementation = args.attn_implementation or model_config.attn_implementation
    generation_config = resolve_generation_config(runtime_config, args.model_name)
    return BackendSettings(
        model_name=args.model_name,
        model_path=str(args.model_path).replace("\\", "/") if args.model_path else None,
        hf_id=model_config.hf_id,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=attn_implementation,
        device_map=model_config.device_map,
        tokenizer_mode=model_config.tokenizer_mode,
        uses_chat_template=model_config.uses_chat_template,
        padding_side=model_config.padding_side,
        max_input_length=model_config.max_input_length,
        stop_strings=list(model_config.stop_strings),
        chat_add_generation_prompt=model_config.chat_add_generation_prompt,
        tokenizer_kwargs=dict(model_config.tokenizer_kwargs),
        apply_chat_template_kwargs=dict(model_config.apply_chat_template_kwargs),
        load_mode=model_config.load_mode,
        generation_config=generation_config,
    )


def build_backend(
    runtime_config,
    *,
    model_name: str,
    model_path: Path | None,
    quantization: str | None,
    dtype: str | None,
    attn_implementation: str | None,
    backend_factory: Callable[..., object] | None = None,
):
    class Args:
        pass

    args = Args()
    args.model_name = model_name
    args.model_path = model_path
    args.quantization = quantization
    args.dtype = dtype
    args.attn_implementation = attn_implementation
    settings = _build_backend_settings(runtime_config, args)
    factory = TransformersTranslationBackend if backend_factory is None else backend_factory
    return factory(settings=settings, model_loader=None, tokenizer_loader=None)


def build_backend_from_args(*, runtime_config, args, backend_factory=None):
    return build_backend(
        runtime_config,
        model_name=args.model_name,
        model_path=args.model_path,
        quantization=args.quantization,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        backend_factory=backend_factory,
    )


def _write_log_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(
    argv: list[str] | None = None,
    build_backend_fn: Callable[..., object] | None = None,
    translate_lines_fn: Callable[..., None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    config_root = _resolve_config_root(args.config_root)
    runtime_config = load_runtime_config(
        config_root / "models.yaml",
        config_root / "prompts.yaml",
        config_root / "generation.yaml",
    )
    if args.model_name not in runtime_config.models:
        raise ValueError(f"unknown model_name: {args.model_name}")

    prompt_template = resolve_prompt_template(runtime_config, args.model_name, args.direction)
    prompt_template_key = resolve_prompt_template_key(runtime_config, args.model_name, args.direction)
    backend_builder = build_backend_fn or build_backend_from_args
    backend = backend_builder(runtime_config=runtime_config, args=args)
    translate = translate_lines_limited if translate_lines_fn is None else translate_lines_fn
    translate(
        input_path=args.input,
        output_path=args.output,
        backend=backend,
        prompt_template=prompt_template,
        batch_size=args.batch_size,
        resume=args.resume,
        max_lines=args.max_lines,
    )

    if args.log_json is not None:
        settings = _build_backend_settings(runtime_config, args)
        _write_log_json(
            args.log_json,
            {
                "model_name": args.model_name,
                "model_path": str(args.model_path) if args.model_path else None,
                "direction": args.direction,
                "batch_size": args.batch_size,
                "resume": args.resume,
                "max_lines": args.max_lines,
                "prompt_template_key": prompt_template_key,
                "prompt_template": prompt_template,
                "backend": {
                    "dtype": settings.dtype,
                    "quantization": settings.quantization,
                    "attn_implementation": settings.attn_implementation,
                    "device_map": settings.device_map,
                    "tokenizer_mode": settings.tokenizer_mode,
                    "uses_chat_template": settings.uses_chat_template,
                    "padding_side": settings.padding_side,
                    "max_input_length": settings.max_input_length,
                    "chat_add_generation_prompt": settings.chat_add_generation_prompt,
                    "tokenizer_kwargs": settings.tokenizer_kwargs,
                    "apply_chat_template_kwargs": settings.apply_chat_template_kwargs,
                    "generation_config": settings.generation_config,
                },
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
