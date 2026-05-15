import csv
import json
from dataclasses import dataclass
from pathlib import Path

from mt_eval.config import VALID_DIRECTIONS


MAIN_FIELDS = ["model", "direction", "bleu_score", "comet_mean"]
DETAIL_FIELDS = [
    "model",
    "direction",
    "bleu_score",
    "bleu_signature",
    "comet_ref1",
    "comet_ref2",
    "comet_mean",
]


@dataclass(frozen=True)
class SummaryResult:
    main_csv_path: Path
    detail_csv_path: Path
    main_rows: list[dict[str, str]]
    detail_rows: list[dict[str, str]]


def _read_json_files(metrics_dir: Path) -> list[tuple[Path, dict]]:
    rows = []
    if not metrics_dir.exists():
        return rows
    for path in sorted(metrics_dir.glob("*.json")):
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def _infer_model_and_direction(path: Path, payload: dict) -> tuple[str, str]:
    model = payload.get("model")
    direction = payload.get("direction")
    if model and direction:
        return str(model), str(direction)

    stem = path.stem
    for candidate in sorted(VALID_DIRECTIONS, key=len, reverse=True):
        suffix = f"_{candidate}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], candidate
    raise ValueError(
        f"unable to infer model and direction from metric file: {path.name}"
    )


def _normalize_value(value) -> str:
    if value is None:
        return ""
    return str(value)


def _build_detail_rows(metrics_root: Path) -> list[dict[str, str]]:
    merged = {}

    for path, payload in _read_json_files(metrics_root / "bleu"):
        model, direction = _infer_model_and_direction(path, payload)
        key = (model, direction)
        row = merged.setdefault(
            key,
            {
                "model": model,
                "direction": direction,
                "bleu_score": "",
                "bleu_signature": "",
                "comet_ref1": "",
                "comet_ref2": "",
                "comet_mean": "",
            },
        )
        row["bleu_score"] = _normalize_value(payload.get("score"))
        row["bleu_signature"] = _normalize_value(payload.get("signature"))

    for path, payload in _read_json_files(metrics_root / "comet"):
        model, direction = _infer_model_and_direction(path, payload)
        key = (model, direction)
        row = merged.setdefault(
            key,
            {
                "model": model,
                "direction": direction,
                "bleu_score": "",
                "bleu_signature": "",
                "comet_ref1": "",
                "comet_ref2": "",
                "comet_mean": "",
            },
        )
        row["comet_ref1"] = _normalize_value(payload.get("comet_ref1"))
        row["comet_ref2"] = _normalize_value(payload.get("comet_ref2"))
        row["comet_mean"] = _normalize_value(payload.get("comet_mean"))

    return [
        merged[key]
        for key in sorted(merged.keys(), key=lambda item: (item[0], item[1]))
    ]


def _build_main_rows(detail_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "model": row["model"],
            "direction": row["direction"],
            "bleu_score": row["bleu_score"],
            "comet_mean": row["comet_mean"],
        }
        for row in detail_rows
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(metrics_root: Path, output_dir: Path) -> SummaryResult:
    detail_rows = _build_detail_rows(metrics_root)
    main_rows = _build_main_rows(detail_rows)
    main_csv_path = output_dir / "main_results.csv"
    detail_csv_path = output_dir / "detail_results.csv"
    _write_csv(main_csv_path, MAIN_FIELDS, main_rows)
    _write_csv(detail_csv_path, DETAIL_FIELDS, detail_rows)
    return SummaryResult(
        main_csv_path=main_csv_path,
        detail_csv_path=detail_csv_path,
        main_rows=main_rows,
        detail_rows=detail_rows,
    )
