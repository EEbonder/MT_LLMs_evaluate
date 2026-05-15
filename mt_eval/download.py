from dataclasses import dataclass
import json
from pathlib import Path

from mt_eval.config import RuntimeConfig

try:
    from huggingface_hub import snapshot_download as _huggingface_snapshot_download
except ModuleNotFoundError:
    _huggingface_snapshot_download = None


@dataclass(frozen=True)
class DownloadJob:
    model_name: str
    hf_id: str
    target_dir: Path


def build_download_jobs(
    cfg: RuntimeConfig,
    cache_dir: Path,
    selected_models: list[str] | None = None,
) -> list[DownloadJob]:
    model_names = list(cfg.models.keys()) if selected_models is None else selected_models
    unknown = [name for name in model_names if name not in cfg.models]
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    return [
        DownloadJob(
            model_name=model_name,
            hf_id=cfg.models[model_name].hf_id,
            target_dir=cache_dir / model_name,
        )
        for model_name in model_names
    ]


def snapshot_download(*, repo_id: str, local_dir: Path) -> str:
    if _huggingface_snapshot_download is None:
        raise ModuleNotFoundError("No module named 'huggingface_hub'")
    return _huggingface_snapshot_download(repo_id=repo_id, local_dir=local_dir)


def local_snapshot_is_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False

    has_tokenizer = any(
        (path / filename).is_file()
        for filename in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    if not has_tokenizer:
        return False

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        weight_files = set(payload.get("weight_map", {}).values())
        return bool(weight_files) and all((path / name).is_file() for name in weight_files)

    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def download_job(job: DownloadJob) -> str:
    if local_snapshot_is_ready(job.target_dir):
        print(f"Using local model snapshot: {job.target_dir}")
        return str(job.target_dir)
    return snapshot_download(repo_id=job.hf_id, local_dir=job.target_dir)
