from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol


class TranslationBackend(Protocol):
    def translate_batch(self, prompts: list[str]) -> list[str]:
        ...


@dataclass(frozen=True)
class BackendSettings:
    model_name: str
    model_path: str | None
    hf_id: str
    dtype: str
    quantization: str | None
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
    load_mode: str
    generation_config: dict[str, Any]


def render_prompt(prompt_template: str, source: str) -> str:
    return prompt_template.format(source=source)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _normalize_output_line(text: str) -> str:
    parts = text.splitlines()
    if not parts:
        return ""
    return " ".join(part.strip() for part in parts).strip()


def translate_lines(
    input_path: Path,
    output_path: Path,
    backend: TranslationBackend,
    prompt_template: str,
    batch_size: int,
    resume: bool,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_lines = _read_lines(input_path)
    completed_lines = 0
    if resume and output_path.exists():
        existing_output = output_path.read_text(encoding="utf-8")
        if existing_output and not existing_output.endswith("\n"):
            raise ValueError(
                "resume state is invalid: output does not end with a trailing newline"
            )
        completed_lines = len(existing_output.splitlines())
        if completed_lines > len(source_lines):
            raise ValueError(
                "resume state is invalid: output has more lines than source"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and completed_lines else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(completed_lines, len(source_lines), batch_size):
            chunk = source_lines[start : start + batch_size]
            prompts = [render_prompt(prompt_template, line) for line in chunk]
            outputs = backend.translate_batch(prompts)
            if len(outputs) != len(prompts):
                raise ValueError(
                    "translate_batch must return exactly one result per prompt"
                )
            for text in outputs:
                handle.write(_normalize_output_line(text) + "\n")


def translate_lines_limited(
    *,
    input_path: Path,
    output_path: Path,
    backend: TranslationBackend,
    prompt_template: str,
    batch_size: int,
    resume: bool,
    max_lines: int | None,
) -> None:
    if max_lines is None:
        translate_lines(
            input_path=input_path,
            output_path=output_path,
            backend=backend,
            prompt_template=prompt_template,
            batch_size=batch_size,
            resume=resume,
        )
        return
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    source_lines = _read_lines(input_path)[:max_lines]
    with TemporaryDirectory() as temp_dir:
        limited_input = Path(temp_dir) / input_path.name
        limited_input.write_text(
            "".join(f"{line}\n" for line in source_lines),
            encoding="utf-8",
        )
        translate_lines(
            input_path=limited_input,
            output_path=output_path,
            backend=backend,
            prompt_template=prompt_template,
            batch_size=batch_size,
            resume=resume,
        )


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _build_model_kwargs(settings: BackendSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": settings.trust_remote_code,
        "device_map": settings.device_map,
    }
    if settings.attn_implementation is not None:
        kwargs["attn_implementation"] = settings.attn_implementation

    if settings.quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "bitsandbytes-backed 8bit loading requires transformers with bitsandbytes support"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif settings.quantization is None:
        kwargs["torch_dtype"] = _torch_dtype(settings.dtype)
    else:
        raise ValueError(f"unsupported quantization: {settings.quantization}")
    return kwargs


class TransformersTranslationBackend:
    def __init__(
        self,
        settings: BackendSettings,
        *,
        model_loader=None,
        tokenizer_loader=None,
    ) -> None:
        self.settings = settings
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()

    def _model_source(self) -> str:
        return self.settings.model_path or self.settings.hf_id

    def _load_tokenizer(self):
        if self._tokenizer_loader is None:
            from transformers import AutoTokenizer

            loader = AutoTokenizer.from_pretrained
        else:
            loader = self._tokenizer_loader

        tokenizer_kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
            "padding_side": self.settings.padding_side,
            **self.settings.tokenizer_kwargs,
        }
        if self.settings.model_name in {"gemma4_31b_it", "gemma4_moe_26b_a4b_it"}:
            tokenizer_kwargs["extra_special_tokens"] = {"video_token": "<|video|>"}

        tokenizer = loader(self._model_source(), **tokenizer_kwargs)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(
            tokenizer, "eos_token_id", None
        ) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        if self._model_loader is None:
            from transformers import AutoModelForCausalLM

            loader = AutoModelForCausalLM.from_pretrained
        else:
            loader = self._model_loader
        return loader(self._model_source(), **_build_model_kwargs(self.settings))

    def _build_model_inputs(self, prompts: list[str]):
        if self.settings.uses_chat_template:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=self.settings.chat_add_generation_prompt,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=self.settings.max_input_length is not None,
                max_length=self.settings.max_input_length,
                return_dict=True,
                **self.settings.apply_chat_template_kwargs,
            )

        return self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.settings.max_input_length is not None,
            max_length=self.settings.max_input_length,
        )

    def translate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        model_inputs = self._build_model_inputs(prompts)
        model_inputs = model_inputs.to(self._model.device)
        model_inputs.pop("token_type_ids", None)
        generated = self._model.generate(**model_inputs, **self.settings.generation_config)
        input_length = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, input_length:]
        decoded = self._tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        outputs = [_strip_stop_strings(text.strip(), self.settings.stop_strings) for text in decoded]
        if len(outputs) != len(prompts):
            raise ValueError("transformers backend returned mismatched output count")
        return outputs


def _strip_stop_strings(text: str, stop_strings: list[str]) -> str:
    for stop_string in stop_strings:
        if stop_string and stop_string in text:
            return text.split(stop_string, 1)[0].rstrip()
    return text
