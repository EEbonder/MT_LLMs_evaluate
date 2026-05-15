# Full Evaluation Summary

Run date: 2026-05-15

## Environment

- GPU: NVIDIA A800-SXM4-80GB
- GPU memory: 81920 MiB
- Driver: 580.126.09
- CUDA reported by `nvidia-smi`: 13.0
- Inference Python: `/root/miniconda3/envs/mt-infer/bin/python`
- Evaluation Python: `/root/miniconda3/envs/mt-eval/bin/python`
- Import checks passed:
  - `mt-infer`: `transformers`, `torch`, `accelerate`, `bitsandbytes`
  - `mt-eval`: `pyidaungsu`, `comet`

## Scope

- Models: `hy_mt1_5_7b`, `niutrans_lmt_60_8b`, `gemma3_27b_it`, `gemma4_31b_it`, `gemma4_moe_26b_a4b_it`, `qwen3_32b`, `qwen3_moe_30b_a3b`
- Directions: `my2zh`, `zh2my`, `my2en`, `en2my`
- Precision / quantization: all completed with `bf16`; no model required `8bit` or `4bit`
- Smoke status: passed for all 7 models and all 4 directions
- Full status: passed for all 7 models and all 4 directions
- Backend: HuggingFace Transformers `AutoModelForCausalLM.generate`, not vLLM

## Configuration Notes

- HY-MT follows the official chat-template path with `add_generation_prompt=false`, no default system prompt, `max_new_tokens=2048`, and official sampling parameters.
- NiuTrans follows the official language-tag prompt style with `add_generation_prompt=true`, `num_beams=5`, `do_sample=false`, and tokenizer loading with `fix_mistral_regex=true`.
- Qwen3 and Qwen3-MoE explicitly set `enable_thinking=false` and use the Qwen3 non-thinking sampling recommendation.
- Gemma3, Gemma4, and Gemma4-MoE explicitly set `enable_thinking=false` and use local/official generation config sampling with `max_new_tokens=512`.
- COMET uses `Unbabel/wmt22-comet-da`, averages two references, runs on CPU (`gpus=0`), and is not additionally normalized.

## Output Coverage

- Raw translations: `outputs/full/translations/raw/` contains 28 text files, each with 500 lines
- Myanmar BLEU-ready translations: `outputs/full/translations/bleu_ready/` contains 14 segmented files, each with 500 lines
- BLEU metrics: `outputs/full/metrics/bleu/` contains 28 JSON files
- COMET metrics: `outputs/full/metrics/comet/` contains 28 JSON files
- Summary tables: `outputs/full/metrics/summary/`
- `main_results.csv`: generated, 29 lines including header
- `detail_results.csv`: generated, 29 lines including header
- Logs: `outputs/logs/`

## Length Audit

- `hy_mt1_5_7b` uses `max_new_tokens=2048`; the other six models use `max_new_tokens=512`.
- Myanmar segmented references have about 28-29 average whitespace tokens, P95 about 58-59, and max about 105-107.
- BLEU brevity ratios do not indicate global truncation for Myanmar-target outputs. NiuTrans remains short (`zh2my` ratio 0.689, `en2my` ratio 0.844), which should be treated as model-output/early-stop quality behavior rather than a global max-token setting issue.

## Main Results

| model | direction | BLEU | COMET mean |
| --- | --- | ---: | ---: |
| gemma3_27b_it | en2my | 14.844007 | 0.903101 |
| gemma3_27b_it | my2en | 30.793633 | 0.852896 |
| gemma3_27b_it | my2zh | 35.430966 | 0.860485 |
| gemma3_27b_it | zh2my | 14.395162 | 0.879348 |
| gemma4_31b_it | en2my | 27.865748 | 0.928742 |
| gemma4_31b_it | my2en | 36.129723 | 0.864693 |
| gemma4_31b_it | my2zh | 38.223008 | 0.874781 |
| gemma4_31b_it | zh2my | 26.036836 | 0.910483 |
| gemma4_moe_26b_a4b_it | en2my | 26.342300 | 0.928172 |
| gemma4_moe_26b_a4b_it | my2en | 34.113233 | 0.861647 |
| gemma4_moe_26b_a4b_it | my2zh | 36.547633 | 0.870679 |
| gemma4_moe_26b_a4b_it | zh2my | 24.237266 | 0.908588 |
| hy_mt1_5_7b | en2my | 15.258780 | 0.918957 |
| hy_mt1_5_7b | my2en | 24.929199 | 0.843920 |
| hy_mt1_5_7b | my2zh | 26.465123 | 0.855372 |
| hy_mt1_5_7b | zh2my | 14.279804 | 0.900474 |
| niutrans_lmt_60_8b | en2my | 16.239705 | 0.871732 |
| niutrans_lmt_60_8b | my2en | 34.488905 | 0.849946 |
| niutrans_lmt_60_8b | my2zh | 37.821484 | 0.862032 |
| niutrans_lmt_60_8b | zh2my | 6.757525 | 0.805309 |
| qwen3_32b | en2my | 15.907476 | 0.887297 |
| qwen3_32b | my2en | 30.871733 | 0.852979 |
| qwen3_32b | my2zh | 35.373089 | 0.862547 |
| qwen3_32b | zh2my | 13.523654 | 0.858585 |
| qwen3_moe_30b_a3b | en2my | 12.995779 | 0.841180 |
| qwen3_moe_30b_a3b | my2en | 29.618109 | 0.837401 |
| qwen3_moe_30b_a3b | my2zh | 36.249415 | 0.853371 |
| qwen3_moe_30b_a3b | zh2my | 11.379166 | 0.807895 |

## Notes

- The corrected summary CSVs are `outputs/full/metrics/summary/main_results.csv` and `outputs/full/metrics/summary/detail_results.csv`.
- NiuTrans `zh2my/en2my` remain the main quality outliers after the official flow fix.
- If runtime becomes the bottleneck in future experiments, add a separate vLLM backend and verify equivalence with smoke tests before mixing it into formal scores.
