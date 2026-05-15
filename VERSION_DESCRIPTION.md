# Version Description: Initial Public Evaluation Workflow

Date: 2026-05-15

This version publishes the reproducible code, configuration, tests, and technical documentation for the Myanmar-Chinese/English LLM machine-translation evaluation workflow.

## Included

- Pipeline scripts for smoke/full evaluation.
- Reusable evaluation modules under `mt_eval/`.
- Model, prompt, and generation configuration files.
- Unit tests for configuration, pipeline planning, translation helpers, BLEU, COMET, segmentation, validation, and summary generation.
- Final public summary files:
  - `outputs/full/EVALUATION_SUMMARY.md`
  - `outputs/full/metrics/summary/main_results.csv`
  - `outputs/full/metrics/summary/detail_results.csv`
- Technical reports:
  - `反馈报告.md`
  - `全流程测评项目技术报告.md`

## Excluded

- `data/`: test datasets and references.
- `models/`: local model weights.
- `root-cache/`: HuggingFace and COMET caches.
- Raw translations, BLEU/COMET per-sample artifacts, logs, temporary files, and AutoDL machine state.
- Any local credentials, tokens, API keys, or private SSH keys.

## Evaluation Summary

The final local run completed 7 models across 4 directions (`my2zh`, `zh2my`, `my2en`, `en2my`) using an A800 80GB GPU. All formal runs used HuggingFace Transformers generation with `bf16`; no model required `8bit` or `4bit`.

The repository intentionally does not include datasets or model weights. To reproduce the full evaluation, place the required test data under `data/test/`, place model snapshots under `models/`, create the `mt-infer` and `mt-eval` environments, then run the documented pipeline commands in `README.md`.
