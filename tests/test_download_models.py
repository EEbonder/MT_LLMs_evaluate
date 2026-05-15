from pathlib import Path
import subprocess
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_eval.config import load_runtime_config
from mt_eval.download import DownloadJob, build_download_jobs, download_job


def test_build_download_jobs_uses_all_selected_models_in_config_order(
    tmp_path: Path,
):
    cfg = load_runtime_config(
        Path("configs/models.yaml"),
        Path("configs/prompts.yaml"),
        Path("configs/generation.yaml"),
    )

    jobs = build_download_jobs(cfg, cache_dir=tmp_path / "models")

    assert [job.model_name for job in jobs] == [
        "hy_mt1_5_7b",
        "niutrans_lmt_60_8b",
        "gemma4_31b_it",
        "qwen3_32b",
        "gemma3_27b_it",
        "qwen3_moe_30b_a3b",
        "gemma4_moe_26b_a4b_it",
    ]
    assert [job.hf_id for job in jobs] == [
        "tencent/HY-MT1.5-7B",
        "NiuTrans/LMT-60-8B",
        "google/gemma-4-31B-it",
        "Qwen/Qwen3-32B",
        "google/gemma-3-27b-it",
        "Qwen/Qwen3-30B-A3B",
        "google/gemma-4-26B-A4B-it",
    ]
    assert [job.target_dir for job in jobs] == [
        tmp_path / "models" / "hy_mt1_5_7b",
        tmp_path / "models" / "niutrans_lmt_60_8b",
        tmp_path / "models" / "gemma4_31b_it",
        tmp_path / "models" / "qwen3_32b",
        tmp_path / "models" / "gemma3_27b_it",
        tmp_path / "models" / "qwen3_moe_30b_a3b",
        tmp_path / "models" / "gemma4_moe_26b_a4b_it",
    ]


def test_download_job_calls_snapshot_download(monkeypatch, tmp_path: Path):
    calls = []

    def fake_snapshot_download(*, repo_id: str, local_dir: Path):
        calls.append((repo_id, local_dir))
        return str(local_dir)

    monkeypatch.setattr("mt_eval.download.snapshot_download", fake_snapshot_download)

    job = DownloadJob(
        model_name="hy_mt1_5_7b",
        hf_id="tencent/HY-MT1.5-7B",
        target_dir=tmp_path / "models" / "hy_mt1_5_7b",
    )

    result = download_job(job)

    assert result == str(job.target_dir)
    assert calls == [("tencent/HY-MT1.5-7B", job.target_dir)]


def test_download_job_skips_remote_download_when_local_snapshot_is_ready(
    monkeypatch, tmp_path: Path
):
    def fail_snapshot_download(*, repo_id: str, local_dir: Path):
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr("mt_eval.download.snapshot_download", fail_snapshot_download)

    target_dir = tmp_path / "models" / "hy_mt1_5_7b"
    target_dir.mkdir(parents=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target_dir / "model-00001-of-00001.safetensors").write_text(
        "weights", encoding="utf-8"
    )
    job = DownloadJob(
        model_name="hy_mt1_5_7b",
        hf_id="tencent/HY-MT1.5-7B",
        target_dir=target_dir,
    )

    result = download_job(job)

    assert result == str(target_dir)


def test_download_models_cli_builds_jobs_and_downloads_them(monkeypatch, tmp_path: Path):
    captured = []

    def fake_download_job(job: DownloadJob):
        captured.append((job.model_name, job.hf_id, job.target_dir))

    monkeypatch.setattr("mt_eval.download.download_job", fake_download_job)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/download_models.py",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    runpy.run_path("scripts/download_models.py", run_name="__main__")

    assert captured == [
        (
            "hy_mt1_5_7b",
            "tencent/HY-MT1.5-7B",
            tmp_path / "cache" / "hy_mt1_5_7b",
        ),
        (
            "niutrans_lmt_60_8b",
            "NiuTrans/LMT-60-8B",
            tmp_path / "cache" / "niutrans_lmt_60_8b",
        ),
        (
            "gemma4_31b_it",
            "google/gemma-4-31B-it",
            tmp_path / "cache" / "gemma4_31b_it",
        ),
        (
            "qwen3_32b",
            "Qwen/Qwen3-32B",
            tmp_path / "cache" / "qwen3_32b",
        ),
        (
            "gemma3_27b_it",
            "google/gemma-3-27b-it",
            tmp_path / "cache" / "gemma3_27b_it",
        ),
        (
            "qwen3_moe_30b_a3b",
            "Qwen/Qwen3-30B-A3B",
            tmp_path / "cache" / "qwen3_moe_30b_a3b",
        ),
        (
            "gemma4_moe_26b_a4b_it",
            "google/gemma-4-26B-A4B-it",
            tmp_path / "cache" / "gemma4_moe_26b_a4b_it",
        ),
    ]


def test_download_models_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--cache-dir" in result.stdout


def test_download_models_cli_supports_model_filter(monkeypatch, tmp_path: Path):
    captured = []

    def fake_download_job(job: DownloadJob):
        captured.append(job.model_name)

    monkeypatch.setattr("mt_eval.download.download_job", fake_download_job)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/download_models.py",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--models",
            "hy_mt1_5_7b",
            "qwen3_32b",
        ],
    )

    runpy.run_path("scripts/download_models.py", run_name="__main__")

    assert captured == ["hy_mt1_5_7b", "qwen3_32b"]


def test_download_models_help_lists_config_root():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config-root" in result.stdout


def test_download_models_cli_uses_repo_relative_configs_from_outside_repo(tmp_path: Path):
    script_path = Path("scripts/download_models.py").resolve()
    pythonpath = str(Path.cwd())
    cache_dir = tmp_path / "cache"

    command = (
        "from pathlib import Path; "
        "import importlib; "
        "import sys; "
        f"sys.argv = [{str(script_path)!r}, '--cache-dir', {str(cache_dir)!r}]; "
        "module = importlib.import_module('mt_eval.download'); "
        "module.download_job = lambda job: print(job.model_name); "
        f"code = compile(Path({str(script_path)!r}).read_text(encoding='ascii'), "
        f"{str(script_path)!r}, 'exec'); "
        f"globals_dict = {{'__name__': '__main__', '__file__': {str(script_path)!r}}}; "
        "exec(code, globals_dict)"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": pythonpath},
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "hy_mt1_5_7b",
        "niutrans_lmt_60_8b",
        "gemma4_31b_it",
        "qwen3_32b",
        "gemma3_27b_it",
        "qwen3_moe_30b_a3b",
        "gemma4_moe_26b_a4b_it",
    ]
