# AutoDL 无卡阶段提示词

请你作为远程 Linux 运维 + LLM 评测执行代理，在当前目录的 `LLMs_evaluate` 项目中完成“无卡阶段”的准备工作。当前机器是 AutoDL 国内网络环境，现在**没有开启 GPU**，所以你现在只做：

1. 配置基础环境
2. 下载模型权重
3. 预下载评测依赖/模型
4. 做不占 GPU 的检查

不要现在启动正式翻译，不要现在跑正式评测。

## 目标

把这个项目准备到这样一个状态：

- 后续只要开启 GPU，就可以直接开始 smoke test 和 full test
- 模型权重已经尽量下载到本地
- `mt-infer` 和 `mt-eval` 两个环境已经建好
- 关键 Python 包已经装好
- COMET 模型如果能提前下载，也提前下载
- 后续评测目标方向为 `my2zh`、`zh2my`、`my2en`、`en2my`；无卡阶段只做 plan/help 检查，不做真实翻译
- 最后给我一份简洁的状态报告

## 工作目录和约束

- 当前工作目录就是项目根目录
- 不要随意改动项目代码和配置文件，除非是为了解决明确的环境兼容问题
- 不要删除已有文件
- 优先复用项目现成脚本
- 任何下载都要优先保证**目录结构与项目约定一致**

## 项目里的目标模型与目标落盘路径

请把最终模型权重分别放到下面这些目录里：

- `models/hy_mt1_5_7b`
- `models/niutrans_lmt_60_8b`
- `models/gemma4_31b_it`
- `models/qwen3_32b`
- `models/gemma3_27b_it`
- `models/qwen3_moe_30b_a3b`
- `models/gemma4_moe_26b_a4b_it`

后续推理脚本会直接按这些目录找本地模型。

## 国内网络下载策略

当前是 AutoDL 国内网络环境，请严格按下面优先级执行：

### 第一优先级：ModelScope / 魔塔社区

如果某个模型在 ModelScope / 魔塔社区存在可用权重，请优先从魔塔下载。

已确认或高度可用的信息：

- `Qwen/Qwen3-32B` 在 ModelScope 有模型页
- 腾讯 `HY-MT` 官方 GitHub 明确提供 ModelScope 分发渠道

对于 `NiuTrans/LMT-60-8B`、Gemma3/Gemma4 系列和 Qwen3 MoE 系列：

- 先在线检查魔塔是否存在可用镜像
- 如果存在，则优先用魔塔下载
- 如果不存在，再走 Hugging Face

### 第二优先级：Hugging Face

如果模型在魔塔没有可用镜像，再走 Hugging Face。

但因为当前是国内网络环境：

- 如果访问 Hugging Face 失败，请优先检查是否已经配置代理或网络加速
- 如果本机已经有代理环境变量或加速能力，请直接利用
- 如果没有代理且 Hugging Face 无法访问，不要盲目无限重试，要明确告诉我卡在什么模型、什么网址、什么报错

### 第三优先级：GitHub

只有在官方明确通过 GitHub 发布权重或 Git LFS 仓库时才走 GitHub。

如果 GitHub 下载需要代理，也按上面的规则处理。

## 建议执行顺序

### 第一步：检查基础环境

请先检查并记录：

- 操作系统
- Python / conda 是否可用
- 磁盘剩余空间
- 当前网络是否能访问：
  - `modelscope.cn`
  - `huggingface.co`
  - `github.com`

### 第二步：创建两个环境

创建并配置：

- `mt-infer`
- `mt-eval`

要求：

- Python 3.11
- `mt-infer` 安装 `requirements-infer.txt`
- `mt-eval` 安装 `requirements-eval.txt`

如果为了从魔塔下载模型需要额外安装 `modelscope`，请在合适的环境里安装即可，优先装在 `mt-infer`

### 第三步：下载模型权重

对七个模型逐个处理。

执行原则：

- 先查魔塔是否有当前模型
- 有则优先魔塔
- 无则 Hugging Face
- 下载完成后，整理到项目要求的本地目录名

如果下载工具默认把模型放到缓存目录，也要把最终可用目录整理成：

- `models/hy_mt1_5_7b`
- `models/niutrans_lmt_60_8b`
- `models/gemma4_31b_it`
- `models/qwen3_32b`
- `models/gemma3_27b_it`
- `models/qwen3_moe_30b_a3b`
- `models/gemma4_moe_26b_a4b_it`

### 第四步：预下载 COMET

进入 `mt-eval` 环境后，尽量把 COMET 模型也提前下载：

- `Unbabel/wmt22-comet-da`

如果能在无卡模式下完成下载，就直接做掉。

### 第五步：做无卡检查

无卡阶段只做这些检查：

- `python scripts/download_models.py --help`
- `python scripts/translate_batch.py --help`
- `python scripts/segment_myanmar.py --help`
- `python scripts/eval_bleu.py --help`
- `python scripts/eval_comet.py --help`
- `python scripts/run_pipeline.py --mode smoke --models hy_mt1_5_7b --infer-python <mt-infer-python> --eval-python <mt-eval-python> --plan-only`
- `python scripts/run_pipeline.py --mode full --models hy_mt1_5_7b niutrans_lmt_60_8b gemma4_31b_it qwen3_32b gemma3_27b_it qwen3_moe_30b_a3b gemma4_moe_26b_a4b_it --directions my2zh zh2my my2en en2my --infer-python <mt-infer-python> --eval-python <mt-eval-python> --plan-only`

不要在无卡阶段做真实翻译。

## 对下载工具的要求

如果你选择魔塔下载，请优先考虑稳定方案，例如：

- `modelscope download`
- 或魔塔官方 Git 仓库方式

如果你选择 Hugging Face：

- 可以用 `huggingface_hub` / `huggingface-cli`
- 如需要代理或加速，请直接使用机器上已存在的可用方案

## 失败处理规则

如果某个模型下载失败，请不要直接放弃。按下面规则处理：

1. 先判断是不是网络问题
2. 再判断是不是权限 / 登录问题
3. 再判断是不是磁盘空间问题
4. 再判断是不是目标站点没有这个模型

只有在确认被阻塞时，才停止并向我汇报。

## 输出要求

全部做完后，请给我一份简洁报告，至少包含：

1. 两个 conda 环境是否创建成功
2. 两个环境里 Python 的绝对路径
3. 七个模型分别从哪里下载的：
   - ModelScope / Hugging Face / GitHub
4. 七个模型是否都已经落到以下路径：
   - `models/hy_mt1_5_7b`
   - `models/niutrans_lmt_60_8b`
   - `models/gemma4_31b_it`
   - `models/qwen3_32b`
   - `models/gemma3_27b_it`
   - `models/qwen3_moe_30b_a3b`
   - `models/gemma4_moe_26b_a4b_it`
5. COMET 模型是否已预下载
6. 哪些步骤成功，哪些步骤失败
7. 如果失败，给出下一步建议

## 执行风格

- 直接执行，不要空谈计划
- 遇到可自动解决的问题就自动解决
- 遇到必须人工决策的问题，再明确告诉我
- 日志不要过度冗长，但关键命令和关键报错要保留
- 为了提高效率你可以自由规划多个子agent帮助你完成项目或者合理的规划并行进行
