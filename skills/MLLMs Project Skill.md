# MLLMs Project Skill

## Purpose

用于协助完成我的 **MLLMs / Cloned Language / Subject–Verb Agreement / Activation Patching** 项目。

项目核心目标：

> 在一个从零训练的 GPT-2 style decoder-only Transformer 中，研究 original language 和 cloned language 是否使用相似的内部机制完成 Subject–Verb Agreement (SVA)，并通过 activation patching 定位和比较两种语言中的相关 circuit。

---

## Project Context

### Model

- GPT-2 style decoder-only Transformer
- PyTorch
- 从零预训练
- 当前主要实验版本计划：
  - `n_layers = 12`
  - decoder-only
  - causal self-attention
- tokenizer:
  - SentencePiece BPE
- 训练语言：
  - Original English
  - Cloned Language

Cloned Language 的构造参考：

- Dufter & Schütze
- Schäfer et al. (2024)
- *On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task*

---

## Research Questions

主要围绕以下问题开展工作：

1. 模型是否真正学习了 Subject–Verb Agreement？
2. 模型中哪些 layer / attention head / MLP 与 SVA 有关？
3. Original Language 与 Cloned Language 是否使用相同的 SVA circuit？
4. 哪些组件是 shared，哪些组件是 language-specific？
5. 简单 SVA 和带 distractor 的复杂 SVA 是否使用不同机制？

---

## Experimental Pipeline

按照以下顺序进行实验：

### Stage 1 — Pretraining Sanity Check

首先确认模型预训练正常。

检查：

- training loss
- validation loss
- perplexity
- train/validation gap
- loss curve 是否稳定下降
- 是否出现明显 overfitting

不要仅因为 loss 较低就认为模型适合 mechanistic interpretability 实验。

---

### Stage 2 — SVA Sanity Check

在 activation patching 之前，确认模型已经学习 SVA。

优先测试：

- simple agreement
- agreement with intervening noun
- subject relative clauses
- distractor conditions

可以使用：

- BLiMP
- SyntaxGym / CausalGym

主要指标：

```text
accuracy
```

以及：

```text
logit_diff =
logit(correct_verb)
-
logit(incorrect_verb)
```

同时记录：

```text
mean logit difference
accuracy
number of samples
```

Original Language 和 Cloned Language 必须分别评测。

---

## SVA Pair Construction

Activation patching 使用 clean / corrupted sentence pairs。

例如：

```text
Clean:
The boy near the cars runs.

Corrupted:
The boys near the cars runs.
```

或者改变 distractor：

```text
Clean:
The boy near the cars runs.

Corrupted:
The boy near the car runs.
```

构造 pair 时：

- 尽量只改变目标 grammatical feature
- 保持其他 token 和结构一致
- 明确 subject token
- 明确 distractor token
- 明确 verb prediction position
- 保存 correct / incorrect verb token IDs

---

## Activation Patching

主要实验流程：

1. 对 clean sentence forward pass
2. 保存 activations
3. 对 corrupted sentence forward pass
4. 在指定 component 上替换 activation
5. 再次 forward
6. 计算 patched logit difference
7. 与 clean / corrupted baseline 比较

推荐 patch 的位置：

```text
residual stream
attention output
MLP output
individual attention heads
```

优先按照：

```text
layer × token position
```

生成 heatmap。

之后再进一步分析：

```text
layer × attention head
```

---

## Patching Metric

优先使用 normalized patching effect：

```text
patched_score - corrupted_score
--------------------------------
clean_score - corrupted_score
```

其中 score 默认使用：

```text
logit(correct) - logit(incorrect)
```

解释：

```text
0
≈ patch 几乎没有恢复 clean behavior

1
≈ patch 基本恢复 clean behavior
```

必须避免只比较 raw logits。

---

## Circuit Analysis

找到高 patching effect 的组件后：

进一步检查：

- attention heads
- MLP layers
- token positions
- residual stream

重点观察：

```text
subject → verb
distractor → verb
previous tokens → verb
```

分析某个组件是否具有稳定作用，而不是只在少数样本出现。

---

## Cross-Lingual Comparison

Original Language 和 Cloned Language 必须使用：

- 相同 sentence structure
- 对应 clean/corrupted pair
- 相同 evaluation metric
- 相同 patching procedure

之后比较：

```text
layer importance
head importance
token-position importance
patching heatmaps
```

目标是判断：

```text
shared circuit
language-specific circuit
partial overlap
```

不要因为两个 heatmap 看起来相似就直接判断 circuit 相同。

应尽量使用 quantitative comparison，例如：

```text
Pearson correlation
Spearman correlation
top-k overlap
Jaccard similarity
```

---

## Attention Analysis

如果某个 attention head 被认为重要，需要进一步检查：

```text
attention pattern
```

尤其观察：

```text
verb 是否 attend subject
verb 是否 attend distractor
```

Attention weight 本身不能作为因果证据。

必须结合：

```text
activation patching / ablation
```

判断其功能。

---

## Logit Lens

Logit Lens 用于观察：

```text
模型在哪一层开始偏向正确 verb
```

基本过程：

```text
hidden state
→ final layer norm
→ LM head
→ logits
```

然后计算每层：

```text
logit(correct)
-
logit(incorrect)
```

Logit Lens 主要作为辅助分析工具。

不要用 Logit Lens 单独证明 circuit。

---

## Data Requirements

实验不要只依赖单个 sentence。

Activation patching 至少应使用一批 sentence pairs。

建议：

```text
debug:
20–50 pairs

initial experiment:
100–500 pairs

final analysis:
500+ pairs
```

如果不同 sentence 类型差异明显，应分别报告：

```text
simple SVA
distractor SVA
complex SVA
```

不要直接混合后只报告一个平均值。

---

## Visualization

常用图：

### SVA Performance

```text
bar plot
```

比较：

```text
original
clone
```

以及不同 SVA subtasks。

### Activation Patching

```text
heatmap
```

推荐：

```text
x-axis: token position
y-axis: layer
color: normalized patching effect
```

### Head-Level Analysis

```text
x-axis: attention head
y-axis: layer
```

### Cross-Lingual Similarity

可以使用：

```text
scatter plot
correlation matrix
top-k overlap
```

---

## Code Workflow

修改项目代码时遵循：

1. 先阅读当前 repository structure
2. 找到已有：
   - model
   - config
   - dataset
   - training
   - evaluation
3. 优先复用已有代码
4. 不要重复实现已有功能
5. 尽量采用最小修改
6. 保持 pretraining 和 mechanistic interpretability 模块分离

推荐目录：

```text
src/
├── model.py
├── config.py
├── data/
├── training/
├── evaluation/
│   └── sva.py
├── interpretability/
│   ├── hooks.py
│   ├── activation_cache.py
│   ├── patching.py
│   ├── logit_lens.py
│   └── circuit_analysis.py
└── visualization/
```

---

## Debugging Rules

当实验结果异常时，优先检查：

1. tokenizer alignment
2. correct / incorrect verb 是否为预期 token
3. verb prediction position 是否正确
4. clone token mapping 是否正确
5. clean / corrupted pair 是否只有预期差异
6. activation hook 是否挂在正确位置
7. tensor shape
8. patch 后 forward pass 是否真正使用 patched activation
9. metric direction 是否写反
10. original / clone 是否使用完全相同实验配置

---

## Experimental Reproducibility

所有实验保存：

```text
random seed
model checkpoint
model config
tokenizer
dataset split
number of samples
evaluation settings
patching component
patching position
```

输出结果尽量保存为：

```text
.json
.csv
.pt
```

不要只保存图片。

---

## Literature Search Rules

涉及论文事实时优先检查原文。

优先来源：

1. ACL Anthology
2. arXiv
3. OpenReview
4. 官方 GitHub repository

重点参考：

- *On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task*
- cloned language literature
- SVA mechanistic interpretability papers
- activation patching / causal tracing papers

不要凭记忆编造论文实验设置。

---

## Response Style

回答我的问题时：

- 默认使用中文
- 技术术语保留英文
- 尽量简洁
- 优先解释「为什么」
- 不重复已经确定的信息
- 如果是代码问题，先指出问题，再给修改方案
- 如果让我写 Codex prompt，给可以直接复制使用的版本
- 不要自动引入与当前研究问题无关的复杂方法

---

## Decision Principle

整个项目的优先级始终是：

```text
模型是否学会 SVA
        ↓
找到与 SVA 有因果关系的组件
        ↓
定位 circuit
        ↓
比较 original / cloned language
```

不要为了提高模型 benchmark performance 而偏离主要研究问题。

核心目标不是训练最强语言模型，而是获得一个：

```text
SVA behavior 足够稳定
+
可以进行 mechanistic interpretability
+
original / clone 可比较
```

的实验模型。