from pathlib import Path

try:
    import sacrebleu
except ModuleNotFoundError:
    sacrebleu = None


TOKENIZERS_BY_DIRECTION = {
    "my2zh": "zh",
    "zh2my": "none",
    "my2en": "13a",
    "en2my": "none",
}


def compute_bleu(direction: str, hyp_path: Path, ref_paths: list[Path]) -> dict:
    if direction not in TOKENIZERS_BY_DIRECTION:
        raise ValueError(f"unsupported direction: {direction}")
    if not ref_paths:
        raise ValueError("at least one reference path is required")
    if sacrebleu is None:
        raise ModuleNotFoundError("sacrebleu is required to compute BLEU")

    hypotheses = hyp_path.read_text(encoding="utf-8").splitlines()
    references = [
        ref_path.read_text(encoding="utf-8").splitlines() for ref_path in ref_paths
    ]
    tokenizer = TOKENIZERS_BY_DIRECTION[direction]
    score = sacrebleu.corpus_bleu(hypotheses, references, tokenize=tokenizer)
    return {
        "direction": direction,
        "tokenizer": tokenizer,
        "score": score.score,
        "signature": score.format(signature=True),
    }
