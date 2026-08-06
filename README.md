# MLLMs Colab

在 TinyStories 上训练 original/cloned-language GPT 的精简项目。代码直接按
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
│   └── activation_patching.py
├── notebooks/
├── configs/
│   └── gpt12_tinystories_clone_colab.yaml
├── requirements.txt
└── pyproject.toml
```

数据、tokenizer、checkpoint 和实验输出分别写入 `data/`、`artifacts/` 和
配置的 `output_dir`，不会提交到 Git。

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

```bash
mllms data prepare
mllms tokenizer train
mllms data tokenize --target-train-tokens 100000000
```

tokenize 阶段按实际 SentencePiece token 数停止，不根据文本大小估算。
精确统计写入 `data/processed/token_counts.json`，validation/test 保持固定且不
混入训练集。

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
mllms blimp download
mllms blimp prepare --checkpoint CHECKPOINT_PATH
mllms blimp evaluate --checkpoint CHECKPOINT_PATH
```

查看全部入口：

```bash
mllms --help
```
