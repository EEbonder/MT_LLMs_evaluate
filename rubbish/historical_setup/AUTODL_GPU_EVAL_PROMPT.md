# AutoDL 有卡阶段提示词

请你作为远程 Linux 运维 + LLM 评测执行代理，在当前目录的 `LLMs_evaluate` 项目中完成“有卡阶段”的正式评测执行。现在假设：

- AutoDL 机器已经开启 GPU
- 无卡阶段已经基本完成
- 项目目录已经就位
- 模型权重已经尽量下载好了
- `mt-infer` 和 `mt-eval` 两个环境已经存在

你的任务是：

1. 先做 GPU 可用性和本地权重检查
2. 先跑 smoke test
3. smoke 稳定后，再跑 full test
4. 最后给我整理输出结果位置和运行总结

## 重要约束

- 尽量使用项目现有脚本，不要重写一套流程
- 优先使用本地已经下载好的权重
- 如果发现某个模型没有落在预期目录，先修正目录，再运行
- 不要随意改动 prompt/config，除非为了处理明确的运行错误
- 当前机器是 A800，默认先尝试 `bf16`
- 如果 `gemma4_31b_it`、`qwen3_32b`、`gemma3_27b_it`、`qwen3_moe_30b_a3b` 或 `gemma4_moe_26b_a4b_it` 在 `bf16` 下 OOM 或明显不稳定，允许切到 `8bit`
- 除非真的没办法，不要用 `4bit`

## 项目约定的本地模型目录

请先确认以下目录都存在并可用：

- `models/hy_mt1_5_7b`
- `models/niutrans_lmt_60_8b`
- `models/gemma4_31b_it`
- `models/qwen3_32b`
- `models/gemma3_27b_it`
- `models/qwen3_moe_30b_a3b`
- `models/gemma4_moe_26b_a4b_it`

## 先做的检查

请先完成：

1. `nvidia-smi`
2. 确认 GPU 型号和显存
3. 找到两个环境的 Python 绝对路径：
   - `mt-infer`
   - `mt-eval`
4. 在 `mt-eval` 中确认：
   - `pyidaungsu` 可导入
   - `comet` 可导入
5. 在 `mt-infer` 中确认：
   - `transformers`
   - `torch`
   - `accelerate`
   - `bitsandbytes`
   都可导入

如果这些基础检查不过，不要急着跑正式测试，先把环境补齐。

## 优先执行策略

### 第一阶段：smoke test

先按**单模型顺序**跑 smoke，不要七个模型一起上来就 full：

1. `hy_mt1_5_7b`
2. `niutrans_lmt_60_8b`
3. `gemma4_31b_it`
4. `qwen3_32b`
5. `gemma3_27b_it`
6. `qwen3_moe_30b_a3b`
7. `gemma4_moe_26b_a4b_it`

优先使用项目的 pipeline：

```bash
python scripts/run_pipeline.py \
  --mode smoke \
  --models <model_name> \
  --infer-python <mt-infer-python> \
  --eval-python <mt-eval-python> \
  --execute
```

### 第二阶段：full test

当 smoke 通过后，再按**单模型顺序**跑 full：

```bash
python scripts/run_pipeline.py \
  --mode full \
  --models <model_name> \
  --infer-python <mt-infer-python> \
  --eval-python <mt-eval-python> \
  --execute
```

不要一上来七个模型一起 full。请按模型逐个完成，这样更容易定位问题、复跑也更方便。

## 大模型显存策略

对于：

- `gemma4_31b_it`
- `qwen3_32b`
- `gemma3_27b_it`
- `qwen3_moe_30b_a3b`
- `gemma4_moe_26b_a4b_it`

请先尝试 pipeline 默认路径，也就是默认 `bf16`。

如果在翻译阶段遇到以下问题之一：

- CUDA OOM
- 模型加载失败且明显与显存相关
- `bf16` 运行明显不稳定

则对该模型切换为 **8bit**，并改用手工分步执行，不要继续死磕 pipeline 默认路径。

## 8bit 手工回退路径

如果某个大模型需要 `8bit`，请只对该模型改走手工命令，其余模型保持原流程。

### 1. my2zh 翻译

```bash
<mt-infer-python> scripts/translate_batch.py \
  --input data/test/my2zh/my.txt \
  --output outputs/full/translations/raw/<model_name>_my2zh.txt \
  --model-name <model_name> \
  --model-path models/<model_name> \
  --direction my2zh \
  --config-root configs \
  --batch-size 10 \
  --resume \
  --quantization 8bit
```

### 2. zh2my 翻译

```bash
<mt-infer-python> scripts/translate_batch.py \
  --input data/test/zh2my/zh.txt \
  --output outputs/full/translations/raw/<model_name>_zh2my.txt \
  --model-name <model_name> \
  --model-path models/<model_name> \
  --direction zh2my \
  --config-root configs \
  --batch-size 10 \
  --resume \
  --quantization 8bit
```

### 3. my2en 翻译

```bash
<mt-infer-python> scripts/translate_batch.py \
  --input data/test/my2en/my.txt \
  --output outputs/full/translations/raw/<model_name>_my2en.txt \
  --model-name <model_name> \
  --model-path models/<model_name> \
  --direction my2en \
  --config-root configs \
  --batch-size 10 \
  --resume \
  --quantization 8bit
```

### 4. en2my 翻译

```bash
<mt-infer-python> scripts/translate_batch.py \
  --input data/test/en2my/en.txt \
  --output outputs/full/translations/raw/<model_name>_en2my.txt \
  --model-name <model_name> \
  --model-path models/<model_name> \
  --direction en2my \
  --config-root configs \
  --batch-size 10 \
  --resume \
  --quantization 8bit
```

### 5. zh2my / en2my 分词

```bash
<mt-eval-python> scripts/segment_myanmar.py \
  --backend pyidaungsu \
  --input outputs/full/translations/raw/<model_name>_zh2my.txt \
  --output outputs/full/translations/bleu_ready/<model_name>_zh2my.seg.txt

<mt-eval-python> scripts/segment_myanmar.py \
  --backend pyidaungsu \
  --input outputs/full/translations/raw/<model_name>_en2my.txt \
  --output outputs/full/translations/bleu_ready/<model_name>_en2my.seg.txt
```

### 6. 输出校验

```bash
<mt-eval-python> scripts/validate_outputs.py \
  --source data/test/my2zh/my.txt \
  --output outputs/full/translations/raw/<model_name>_my2zh.txt

<mt-eval-python> scripts/validate_outputs.py \
  --source data/test/zh2my/zh.txt \
  --output outputs/full/translations/raw/<model_name>_zh2my.txt

<mt-eval-python> scripts/validate_outputs.py \
  --source data/test/my2en/my.txt \
  --output outputs/full/translations/raw/<model_name>_my2en.txt

<mt-eval-python> scripts/validate_outputs.py \
  --source data/test/en2my/en.txt \
  --output outputs/full/translations/raw/<model_name>_en2my.txt
```

### 7. BLEU

```bash
<mt-eval-python> scripts/eval_bleu.py \
  --direction my2zh \
  --hyp outputs/full/translations/raw/<model_name>_my2zh.txt \
  --refs data/test/my2zh/my2zh1.txt data/test/my2zh/my2zh2.txt \
  --output outputs/full/metrics/bleu/<model_name>_my2zh.json \
  --model-name <model_name>

<mt-eval-python> scripts/eval_bleu.py \
  --direction zh2my \
  --hyp outputs/full/translations/bleu_ready/<model_name>_zh2my.seg.txt \
  --refs data/test/zh2my/zh2my1_seg.txt data/test/zh2my/zh2my2_seg.txt \
  --output outputs/full/metrics/bleu/<model_name>_zh2my.json \
  --model-name <model_name>

<mt-eval-python> scripts/eval_bleu.py \
  --direction my2en \
  --hyp outputs/full/translations/raw/<model_name>_my2en.txt \
  --refs data/test/my2en/my2en1.txt data/test/my2en/my2en2.txt \
  --output outputs/full/metrics/bleu/<model_name>_my2en.json \
  --model-name <model_name>

<mt-eval-python> scripts/eval_bleu.py \
  --direction en2my \
  --hyp outputs/full/translations/bleu_ready/<model_name>_en2my.seg.txt \
  --refs data/test/en2my/en2my1_seg.txt data/test/en2my/en2my2_seg.txt \
  --output outputs/full/metrics/bleu/<model_name>_en2my.json \
  --model-name <model_name>
```

### 8. COMET

```bash
<mt-eval-python> scripts/eval_comet.py \
  --src data/test/my2zh/my.txt \
  --hyp outputs/full/translations/raw/<model_name>_my2zh.txt \
  --ref1 data/test/my2zh/my2zh1.txt \
  --ref2 data/test/my2zh/my2zh2.txt \
  --output outputs/full/metrics/comet/<model_name>_my2zh.json \
  --model-name <model_name> \
  --direction my2zh

<mt-eval-python> scripts/eval_comet.py \
  --src data/test/zh2my/zh.txt \
  --hyp outputs/full/translations/bleu_ready/<model_name>_zh2my.seg.txt \
  --ref1 data/test/zh2my/zh2my1_seg.txt \
  --ref2 data/test/zh2my/zh2my2_seg.txt \
  --output outputs/full/metrics/comet/<model_name>_zh2my.json \
  --model-name <model_name> \
  --direction zh2my

<mt-eval-python> scripts/eval_comet.py \
  --src data/test/my2en/my.txt \
  --hyp outputs/full/translations/raw/<model_name>_my2en.txt \
  --ref1 data/test/my2en/my2en1.txt \
  --ref2 data/test/my2en/my2en2.txt \
  --output outputs/full/metrics/comet/<model_name>_my2en.json \
  --model-name <model_name> \
  --direction my2en

<mt-eval-python> scripts/eval_comet.py \
  --src data/test/en2my/en.txt \
  --hyp outputs/full/translations/bleu_ready/<model_name>_en2my.seg.txt \
  --ref1 data/test/en2my/en2my1_seg.txt \
  --ref2 data/test/en2my/en2my2_seg.txt \
  --output outputs/full/metrics/comet/<model_name>_en2my.json \
  --model-name <model_name> \
  --direction en2my
```

### 9. 汇总

```bash
<mt-eval-python> scripts/summarize_metrics.py \
  --metrics-root outputs/full/metrics \
  --output-dir outputs/full/metrics/summary
```

## smoke 结果判定

只有满足下面条件，才算 smoke 通过：

1. 模型能成功加载
2. 四个方向都能产出文本
3. `zh2my` 和 `en2my` 分词成功
4. `validate_outputs.py` 通过
5. BLEU JSON 能写出来
6. COMET JSON 能写出来

如果 smoke 不通过，不要直接跑 full。先定位并修复问题。

## full 执行顺序建议

推荐按这个顺序逐个完成：

1. `hy_mt1_5_7b`
2. `niutrans_lmt_60_8b`
3. `gemma4_31b_it`
4. `qwen3_32b`
5. `gemma3_27b_it`
6. `qwen3_moe_30b_a3b`
7. `gemma4_moe_26b_a4b_it`

每跑完一个模型，就检查一次：

- 输出文件是否齐全
- BLEU/COMET JSON 是否齐全
- `outputs/full/metrics/summary/` 是否已更新

## 运行中的行为要求

- 所有命令尽量保留日志
- 不要因为一个模型失败就跳过不汇报
- 如果某个模型切到 `8bit`，要在最终报告中明确写出
- 不要使用 `4bit`，除非你已经明确证明 `8bit` 也不行，并且必须先向我说明

## 最终输出要求

跑完后给我一份简洁报告，至少包含：

1. GPU 型号、显存、驱动信息
2. 两个环境 Python 路径
3. 七个模型分别用的是：
   - `bf16`
   - 还是 `8bit`
4. 每个模型 smoke 是否通过
5. 每个模型 full 是否通过
6. 结果文件位置：
   - `outputs/full/translations/raw/`
   - `outputs/full/translations/bleu_ready/`
   - `outputs/full/metrics/bleu/`
   - `outputs/full/metrics/comet/`
   - `outputs/full/metrics/summary/`
7. 是否生成了：
   - `main_results.csv`
   - `detail_results.csv`
8. 如果有失败，给出最直接的原因和下一步建议

## 执行风格

- 直接执行
- 优先解决问题，不要长篇空谈
- 对显存、量化、下载源、错误日志保持敏感
- 目标是把正式评测真正跑起来，而不是只跑通 help 命令
