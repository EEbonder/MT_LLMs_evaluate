from pathlib import Path
from typing import Callable


def summarize_comet_scores(ref1_score: float, ref2_score: float) -> dict:
    mean_score = round((ref1_score + ref2_score) / 2.0, 10)
    return {
        "comet_ref1": ref1_score,
        "comet_ref2": ref2_score,
        "comet_mean": mean_score,
    }


def _average_scores(scores: list[float]) -> float:
    return round(sum(scores) / len(scores), 10)


def _prediction_score(prediction) -> float:
    system_score = getattr(prediction, "system_score", None)
    if system_score is not None:
        return float(system_score)

    if isinstance(prediction, dict) and prediction.get("system_score") is not None:
        return float(prediction["system_score"])

    if isinstance(prediction, tuple):
        segment_scores = prediction[0]
        if len(prediction) > 1 and prediction[1] is not None:
            return float(prediction[1])
        return _average_scores(segment_scores)

    segment_scores = getattr(prediction, "scores", prediction)
    if isinstance(segment_scores, dict):
        segment_scores = segment_scores["scores"]
    return _average_scores(segment_scores)


def build_comet_model(model_name: str):
    try:
        from comet import download_model, load_from_checkpoint
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("COMET is required to compute COMET scores") from exc

    local_path = Path(model_name)
    if local_path.exists():
        checkpoint_path = local_path
        if local_path.is_dir():
            checkpoint_path = local_path / "checkpoints" / "model.ckpt"
        return load_from_checkpoint(str(checkpoint_path))

    model_path = download_model(model_name)
    return load_from_checkpoint(model_path)


def score_comet_reference_with_model(
    model,
    src_path: Path,
    hyp_path: Path,
    ref_path: Path,
) -> float:
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    hyp_lines = hyp_path.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_path.read_text(encoding="utf-8").splitlines()

    if not (len(src_lines) == len(hyp_lines) == len(ref_lines)):
        raise ValueError(
            "source, hypothesis, and reference files must have aligned line counts"
        )

    data = [
        {"src": src_line, "mt": hyp_line, "ref": ref_line}
        for src_line, hyp_line, ref_line in zip(src_lines, hyp_lines, ref_lines)
    ]
    prediction = model.predict(data, batch_size=1, gpus=0)
    return _prediction_score(prediction)


def score_comet_reference(
    src_path: Path,
    hyp_path: Path,
    ref_path: Path,
    model_name: str,
) -> float:
    model = build_comet_model(model_name)
    return score_comet_reference_with_model(model, src_path, hyp_path, ref_path)


def run_two_reference_comet(
    src_path: Path,
    hyp_path: Path,
    ref1_path: Path,
    ref2_path: Path,
    model_name: str = "Unbabel/wmt22-comet-da",
    score_runner: Callable[[Path, Path, Path, str], float] | None = None,
) -> dict:
    if score_runner is None:
        model = build_comet_model(model_name)
        ref1_score = score_comet_reference_with_model(model, src_path, hyp_path, ref1_path)
        ref2_score = score_comet_reference_with_model(model, src_path, hyp_path, ref2_path)
    else:
        ref1_score = score_runner(src_path, hyp_path, ref1_path, model_name)
        ref2_score = score_runner(src_path, hyp_path, ref2_path, model_name)
    return summarize_comet_scores(ref1_score, ref2_score)
