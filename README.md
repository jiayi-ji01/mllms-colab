# MLLMs Colab

训练 original/cloned-language GPT 的精简项目，支持 TinyStories 和 BabyLM。
代码直接按
功能放在仓库根目录，不使用额外的 `src/`、`scripts/` 或项目名包装层。

## 结构

```text
.
├── main.py                       统一命令入口
├── train.py                      预训练、验证与 checkpoint
├── model/
│   ├── config.py                 GPT 配置
│   └── model.py                  decoder-only GPT
├── data_lib/
│   ├── cloned.py                 cloned mapping 与 TokenStream
│   ├── prepare_babylm.py         BabyLM 100M 官方数据准备
│   └── prepare_tinystories.py    TinyStories 固定划分
├── tokenizer/
│   ├── tokenizer.py              SentencePiece 加载
│   ├── train_tokenizer.py        BPE tokenizer 训练
│   └── tokenization.py           通用 uint16 token stream 生成
├── blimp/
│   ├── download_blimp.py
│   ├── prepare_blimp.py
│   └── evaluate_blimp.py
├── analysis/
│   ├── prepare_sva_pairs.py       CausalGym controlled SVA pairs
│   ├── evaluate_sva.py            SVA 基线与共同 sanity filter
│   ├── activation_patching.py     layer/token/head patching
│   └── sva_utils.py               SVA 公共加载与评分函数
├── plots/
│   └── reports.py                 训练、SVA、BLiMP 与 patching 报告
├── notebooks/
├── configs/
│   ├── gpt12_babylm_clone_colab.yaml
│   └── gpt12_tinystories_clone_colab.yaml
├── requirements.txt
└── pyproject.toml
```

数据、tokenizer、checkpoint 和实验输出分别写入 `data/`、`artifacts/` 和
配置的 `output_dir`，不会提交到 Git。

## End-to-end Colab notebook

[Open the complete pipeline in Colab](https://colab.research.google.com/github/jiayi-ji01/mllms-colab/blob/main/notebooks/mllms_colab_end_to_end.ipynb)

Notebook 会依次完成环境安装、数据准备或从 Drive 恢复、训练或续训、训练曲线、
BLiMP 评估、original/clone activation patching，以及 PNG/CSV 报告展示。首次
运行前在 Colab 中选择 GPU runtime。

BabyLM 100M 的独立 end-to-end notebook：

[Open the BabyLM pipeline in Colab](https://colab.research.google.com/github/jiayi-ji01/mllms-colab/blob/main/notebooks/mllms_babylm_colab_end_to_end.ipynb)

## Colab 安装

先选择 GPU runtime 并挂载 Drive：

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
git clone YOUR_REPOSITORY_URL /content/mllms-colab
cd /content/mllms-colab
pip install -r requirements.txt
pip install -e . --no-deps
```

Colab 自带与 CUDA 匹配的 PyTorch，因此 `requirements.txt` 不重复安装
PyTorch。

## 数据准备

TinyStories：

```bash
mllms data prepare
mllms tokenizer train
mllms data tokenize --target-train-tokens 100000000
```

tokenize 阶段按实际 SentencePiece token 数停止，不根据文本大小估算。
精确统计写入 `data/processed/token_counts.json`，validation/test 保持固定且不
混入训练集。

BabyLM 100M（在 Colab 中运行，不会下载到本地电脑）：

```bash
mllms data prepare-babylm --output-dir data/babylm/raw
mllms tokenizer train \
  --input data/babylm/raw/train.txt \
  --model-prefix artifacts/babylm_tokenizer/tokenizer \
  --vocab-size 16000 \
  --input-sentence-size 5000000
mllms data tokenize \
  --input-dir data/babylm/raw \
  --output-dir data/babylm/processed \
  --tokenizer artifacts/babylm_tokenizer/tokenizer.model \
  --no-train-token-limit
```

BabyLM 下载使用官方 `cambridge-climb/BabyLM` cleaned 100M-word strict
training corpus，以及独立的官方 dev/test。实际 SentencePiece token 数写入
`data/babylm/processed/token_counts.json`。

## BabyLM 12 层 Colab 训练

```bash
mllms train --config configs/gpt12_babylm_clone_colab.yaml
```

配置为 12 layers、8 heads、`d_model=512`、`d_ff=2048`、context 256，
SentencePiece vocabulary 16000；cloned mapping 后模型 vocabulary 为 32000。
默认 micro batch 4、gradient accumulation 8，每步仍处理 8192 tokens。
训练步数会在 tokenization 后根据实际训练 token 数自动计算为 nominal 2 epochs，
使 original/clone 各自期望获得约一份语料量。输出使用新的 Drive 目录：

```text
/content/drive/MyDrive/mllms-colab/runs/gpt12_babylm_clone_colab/
```

恢复训练：

```bash
mllms train \
  --config configs/gpt12_babylm_clone_colab.yaml \
  --resume /content/drive/MyDrive/mllms-colab/runs/gpt12_babylm_clone_colab/latest.pt
```

## 12 层 Colab 训练

```bash
mllms train --config configs/gpt12_tinystories_clone_colab.yaml
```

配置为 12 layers、4 heads、`d_model=256`、`d_ff=1024`、context 256、
dropout 0.1。基础 SentencePiece vocabulary 为 4096；original/clone 使用独立
ID 空间，因此模型 vocabulary 为 8192。

每个 optimizer step 处理：

```text
8 micro batch × 4 accumulation × 256 context = 8,192 tokens
```

约 200M seen tokens 对应 24,415 optimizer steps。checkpoint 和
`train_log.jsonl` 默认保存在：

```text
/content/drive/MyDrive/mllms-colab/runs/gpt12_tinystories_clone_colab/
```

恢复训练：

```bash
mllms train \
  --config configs/gpt12_tinystories_clone_colab.yaml \
  --resume /content/drive/MyDrive/mllms-colab/runs/gpt12_tinystories_clone_colab/latest.pt
```

如果 CUDA OOM，将配置改为 `micro_batch_size: 4`；要继续保持 effective batch
size 32，同时将 `gradient_accumulation_steps` 改为 8。

## BLiMP

```bash
RUN_DIR=/content/drive/MyDrive/mllms-colab/runs/gpt12_tinystories_clone_colab
CHECKPOINT=$RUN_DIR/best.pt

mllms blimp download
mllms blimp prepare --checkpoint "$CHECKPOINT"
mllms blimp evaluate \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_DIR/blimp"
```

查看全部入口：

```bash
mllms --help
```

BabyLM 模型的严格 one-token verb logit SVA 评估：

```bash
RUN_DIR=/content/drive/MyDrive/mllms-colab/runs/gpt12_babylm_clone_colab
TOKENIZER=artifacts/babylm_tokenizer/tokenizer.model
BLIMP_DATA=data/blimp/processed/babylm_agreement.jsonl

mllms blimp download
mllms blimp prepare \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --output "$BLIMP_DATA"
mllms blimp evaluate \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --data "$BLIMP_DATA" \
  --scoring verb-logit \
  --device cuda \
  --output-dir "$RUN_DIR/blimp_sva_logit"
```

`verb-logit` 只保留共享前缀且两个候选动词均为单个 SentencePiece token 的
BLiMP 样本，因此每个样本严格计算
`logit(correct_verb) - logit(incorrect_verb)`。原有可比评估仍可通过
`--scoring conditional-logprob` 运行，并建议保存到单独的 `blimp_conditional`
目录。

## Controlled SVA 与 activation patching

新的 SVA 流程使用 CausalGym 中三个 subject–verb number agreement task 的
controlled clean/corrupted prompts。默认固定抽取 1000 pairs，clean subject 的
singular/plural 各 500 条；original 和 clone 只做 token-ID 映射，因此文本、顺序、
答案和筛选条件完全相同。CausalGym 个别原始标签同时改变词形和词义，准备脚本会
改用 matched `is/are`、`was/were`、`has/have`，使 logit difference 只比较数一致性。

在 Colab 中运行：

```bash
RUN_DIR=/content/drive/MyDrive/mllms-colab/runs/gpt12_babylm_clone_colab
CHECKPOINT=$RUN_DIR/best.pt
TOKENIZER=artifacts/babylm_tokenizer/tokenizer.model
SVA_DIR=$RUN_DIR/causalgym_sva
PATCH_DIR=$RUN_DIR/causalgym_patching

mllms analyze prepare-sva \
  --tokenizer "$TOKENIZER" \
  --num-pairs 1000 \
  --output data/causalgym/sva_pairs.jsonl

mllms analyze evaluate-sva \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --data data/causalgym/sva_pairs.jsonl \
  --device cuda \
  --output-dir "$SVA_DIR"

mllms analyze activation-patching \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --data "$SVA_DIR/sanity_pairs.jsonl" \
  --language both \
  --max-examples 1000 \
  --device cuda \
  --output-dir "$PATCH_DIR"
```

基线 `sva_summary.json` 同时报告 accuracy 和 mean logit difference。clean prompt
使用 `logit(clean answer) - logit(corrupted answer)`；corrupted prompt 为了保持
“correct − incorrect”的方向，在汇总时反转符号。只有 original 与 clone 都满足
`LD_clean > 0` 且 `LD_corrupted < 0` 的 pair 才写入 `sanity_pairs.jsonl`。如果共同
通过的样本少于 500，脚本会明确警告，此时不应把结果当作正式 500–1000 pair
patching 实验。

Patching 将 clean activation 注入 corrupted run，并保存两种指标：

```text
delta_ld = LD_patched - LD_corrupted
recovery = delta_ld / (LD_clean - LD_corrupted)
```

默认遍历全部 prompt token；residual、attention output、MLP output 保存
`layer × relative_token`，attention-head output 保存
`layer × relative_token × head`。每种语言目录中包含 mean `.npy`、count `.npy`、
完整 per-example `.npz`、`mean_scores.csv`、metadata 和两张 heatmap。若 T4 显存
不足，可把 `--example-batch-size` 从 8 改为 4，或降低
`--intervention-batch-size`；这不会改变实验定义。

## Colab 查看图表

训练结束或中断后生成预训练报告：

```python
from IPython.display import Image, display
import pandas as pd

run_dir = (
    "/content/drive/MyDrive/mllms-colab/"
    "runs/gpt12_tinystories_clone_colab"
)

!mllms plot training --run-dir "{run_dir}"
display(Image(filename=f"{run_dir}/training_report.png"))
display(pd.read_csv(f"{run_dir}/training_summary.csv"))
```

BLiMP 评估完成后：

```python
blimp_dir = f"{run_dir}/blimp"

!mllms plot blimp --results-dir "{blimp_dir}"
display(Image(filename=f"{blimp_dir}/blimp_report.png"))
display(pd.read_csv(f"{blimp_dir}/blimp_summary.csv"))
```

Activation patching 完成后：

```python
from IPython.display import Image, display
import pandas as pd

sva_dir = f"{run_dir}/causalgym_sva"
patching_dir = f"{run_dir}/causalgym_patching"

!mllms plot sva --results-dir "{sva_dir}"
display(Image(filename=f"{sva_dir}/sva_report.png"))
display(pd.read_csv(f"{sva_dir}/sva_summary.csv"))

!mllms plot patching \
  --results-dir "{patching_dir}" \
  --language original \
  --metric recovery
display(Image(filename=f"{patching_dir}/patching_original_recovery.png"))
display(pd.read_csv(
    f"{patching_dir}/patching_top_sites_original_recovery.csv"
))
```

将 `--language` 改为 `clone` 可画 clone，将 `--metric` 改为 `delta_ld` 可画
未归一化指标。各命令同时生成 CSV 表格，方便在 Colab 或本地进一步统计。

## Jupyter Book 实验报告

BabyLM 训练与 SVA 结果整理在 `book/`。本地打开：

```bash
python -m pip install -e '.[book]'
cd book
jupyter book start
```

生成可离线查看的静态 HTML：

```bash
cd book
jupyter book build --html --strict
python -m http.server 8000 --directory _build/html
```
