# Myanmar-Chinese/English MT Evaluation Workflow

This folder is the runnable evaluation bundle for formal Myanmar-Chinese and Myanmar-English MT testing. It is organized as small library modules under `mt_eval/` and thin CLIs under `scripts/`.

## Layout

- `configs/`: model, prompt, and generation config
- `data/test/`: packaged test sets and references
- `scripts/`: entrypoints for download, translation, segmentation, validation, BLEU, COMET, pipeline, and summary
- `mt_eval/`: reusable helpers behind the CLIs
- `outputs/`: generated translations, metrics, and summary tables

## Environments

Use Python 3.11.

### Inference environment

```powershell
conda create -n mt-infer python=3.11 -y
conda activate mt-infer
python -m pip install --upgrade pip
pip install -r requirements-infer.txt
```

### Evaluation environment

```powershell
conda create -n mt-eval python=3.11 -y
conda activate mt-eval
python -m pip install --upgrade pip
pip install -r requirements-eval.txt
```

## Copy to server

Copy the whole `LLMs_evaluate` folder to the A800 server. Keep the bundled `configs/` and `data/test/` paths intact unless you plan to override them with CLI options.

## Pipeline planning

Plan mode prints the exact JSON execution plan without running commands.

```powershell
python scripts/run_pipeline.py --mode smoke --models hy_mt1_5_7b --infer-python <path-to-mt-infer-python> --eval-python <path-to-mt-eval-python> --plan-only
python scripts/run_pipeline.py --mode full --models hy_mt1_5_7b niutrans_lmt_60_8b gemma4_31b_it qwen3_32b gemma3_27b_it qwen3_moe_30b_a3b gemma4_moe_26b_a4b_it --directions my2zh zh2my my2en en2my --infer-python <path-to-mt-infer-python> --eval-python <path-to-mt-eval-python> --plan-only
```

Useful options:

- `--config-root <path>`
- `--data-root <path>`
- `--models-root <path>`
- `--output-root <path>`
- `--directions my2zh zh2my my2en en2my` to restrict or explicitly list evaluation directions
- `--infer-python <python-executable>`
- `--eval-python <python-executable>`
- `--smoke-lines <n>` for smoke mode

## Pipeline execution

Execute mode runs the planned steps in order and stops on the first failure.

```powershell
python scripts/run_pipeline.py --mode smoke --models hy_mt1_5_7b --infer-python <path-to-mt-infer-python> --eval-python <path-to-mt-eval-python> --execute
python scripts/run_pipeline.py --mode full --models hy_mt1_5_7b niutrans_lmt_60_8b gemma4_31b_it qwen3_32b gemma3_27b_it qwen3_moe_30b_a3b gemma4_moe_26b_a4b_it --directions my2zh zh2my my2en en2my --infer-python <path-to-mt-infer-python> --eval-python <path-to-mt-eval-python> --execute
```

The pipeline currently schedules these stages:

1. `download_models.py`
2. `translate_batch.py` for `my2zh`, `zh2my`, `my2en`, and `en2my`
3. `segment_myanmar.py` for Myanmar-target directions: `zh2my` and `en2my`
4. `validate_outputs.py`
5. `eval_bleu.py`
6. `eval_comet.py`
7. `summarize_metrics.py`

Smoke mode includes `--max-lines <n>` on translation commands. Full mode omits it.
Default outputs are isolated by mode:

- `outputs/smoke/...`
- `outputs/full/...`

## Current execution assumptions

`run_pipeline.py` executes the real command path used for formal runs. Smoke mode creates truncated copies of source and reference files under `outputs/smoke/smoke_inputs/` so validation, BLEU, and COMET all run on aligned subsets instead of only truncating translation output.

Model prompt and tokenizer assumptions are kept in `configs/models.yaml`, `configs/prompts.yaml`, and `configs/generation.yaml`.

- HY-MT follows the official chat-template path with `add_generation_prompt=false`, no default system prompt, and the official recommended sampling parameters.
- NiuTrans LMT follows the official language-tag prompt style, `padding_side=left`, `num_beams=5`, and tokenizer loading with `fix_mistral_regex=true`.
- Gemma and Qwen chat models explicitly pass `enable_thinking=false`; formal runs do not use thinking mode.
- Translation inference currently uses HuggingFace Transformers `AutoModelForCausalLM.generate`, not vLLM.

For Myanmar-target BLEU (`zh2my` and `en2my`), supply a formal Myanmar segmenter explicitly when needed:

```powershell
python scripts/segment_myanmar.py --backend command --command "python path/to/segmenter.py" --input outputs/full/translations/raw/hy_mt1_5_7b_zh2my.txt --output outputs/full/translations/bleu_ready/hy_mt1_5_7b_zh2my.seg.txt
python scripts/segment_myanmar.py --backend command --command "python path/to/segmenter.py" --input outputs/full/translations/raw/hy_mt1_5_7b_en2my.txt --output outputs/full/translations/bleu_ready/hy_mt1_5_7b_en2my.seg.txt
```

The default formal backend in the packaged pipeline is `pyidaungsu`, installed through `requirements-eval.txt`. Override it manually with `--backend command` only if you need a different formal tokenizer. The `fallback` backend is only for local smoke tests.

## Metrics summary

After BLEU and COMET JSON files exist under a mode-specific metrics directory such as `outputs/full/metrics/bleu/` and `outputs/full/metrics/comet/`, merge them into CSV tables:

```powershell
python scripts/summarize_metrics.py
python scripts/summarize_metrics.py --metrics-root outputs/full/metrics --output-dir outputs/full/metrics/summary
```

Outputs:

- `outputs/full/metrics/summary/main_results.csv`
- `outputs/full/metrics/summary/detail_results.csv`

`main_results.csv` is the short table for reporting. `detail_results.csv` keeps BLEU signature plus per-reference COMET fields.

## Latest full evaluation

The completed 7-model, 4-direction A800 full run is summarized in `outputs/full/EVALUATION_SUMMARY.md`. On 2026-05-15, all results were refreshed after official prompt/tokenizer/thinking/sampling checks. The summary CSVs contain 28 model-direction rows for `hy_mt1_5_7b`, `niutrans_lmt_60_8b`, `gemma3_27b_it`, `gemma4_31b_it`, `gemma4_moe_26b_a4b_it`, `qwen3_32b`, and `qwen3_moe_30b_a3b`.

All formal runs used HuggingFace Transformers generation with `bf16`; no model required `8bit` or `4bit`. `反馈报告.md` records the official-template audit, COMET setup, and Myanmar-target `max_new_tokens` length check.

Primary result files:

- `outputs/full/translations/raw/`
- `outputs/full/translations/bleu_ready/`
- `outputs/full/metrics/bleu/`
- `outputs/full/metrics/comet/`
- `outputs/full/metrics/summary/main_results.csv`
- `outputs/full/metrics/summary/detail_results.csv`
- `反馈报告.md`

## Focused verification

```powershell
pytest tests/test_run_pipeline.py tests/test_summarize_metrics.py -q
```
