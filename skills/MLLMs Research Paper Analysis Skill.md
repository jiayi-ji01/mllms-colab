# MLLMs Research Paper Analysis Skill

## Purpose

用于系统检索、阅读和分析与当前 **MLLMs / Cloned Language / Decoder-only Transformer / Subject–Verb Agreement / Mechanistic Interpretability** 项目相关的论文及其官方代码库。

本 Skill 不仅总结论文结果，还需要回答：

> 这篇论文的实验具体是怎么做的？

并进一步判断：

> 哪些方法、模型配置、训练技巧和代码实现可以迁移到当前 MLLMs 项目？

---

# 1. Current Research Goal

当前项目主要研究：

> 在 original language 和 cloned language 上训练 decoder-only Transformer，并研究模型完成 Subject–Verb Agreement (SVA) 时使用的内部机制是否相同。

整体研究路线：

```text
Dataset / Cloned Language
        ↓
Tokenizer
        ↓
Decoder-only Transformer
        ↓
Pretraining
        ↓
SVA Sanity Check
        ↓
Activation Patching
        ↓
Circuit Localization
        ↓
Original vs Clone Comparison
```

因此论文检索不能只关注 Activation Patching。

需要覆盖整个 pipeline。

---

# 2. Main Paper Categories

检索论文时，按照以下类别进行。

## Category A — Decoder-only Language Models

重点寻找：

- GPT
- GPT-2 style models
- small language models
- decoder-only Transformer
- Transformer trained from scratch
- language modeling
- causal language modeling

主要回答：

> 一个适合 mechanistic interpretability 的 decoder-only model 应该如何设计和训练？

---

## Category B — Model Architecture

重点研究不同论文如何设置：

```text
n_layers
n_heads
d_model
d_ff
context length
parameter count
```

以及：

```text
Pre-LN
Post-LN
RMSNorm
LayerNorm
```

```text
GELU
ReLU
SwiGLU
```

```text
learned positional embeddings
sinusoidal embeddings
RoPE
```

```text
standard MHA
GQA
MQA
```

需要判断：

> 哪些改动适合小型 decoder-only Transformer？

而不是简单推荐最新架构。

---

# 3. Model Optimization Literature

特别检索：

> decoder-only Transformer 如何获得更稳定、更高效的训练？

关注：

### Initialization

例如：

```text
Xavier initialization
Kaiming initialization
GPT-style residual scaling
```

检查论文是否对 residual branch 使用特殊初始化。

---

### Normalization

比较：

```text
Pre-LN
vs
Post-LN
```

关注：

- gradient stability
- convergence
- deep Transformer training

---

### Activation Function

比较：

```text
ReLU
GELU
SwiGLU
```

记录：

- performance
- parameter cost
- training stability

---

### Residual Connections

研究：

```text
residual scaling
residual initialization
```

以及深层 Transformer 的稳定性。

---

### Positional Encoding

比较：

```text
learned position embeddings
sinusoidal
RoPE
```

但必须结合当前研究需求判断。

如果修改会明显增加 mechanistic interpretability 难度，需要说明。

---

# 4. Hyperparameter Analysis

论文分析必须提取模型超参数。

至少记录：

```text
n_layers
n_heads
d_model
d_ff
context_length
vocab_size
dropout
bias
parameter_count
```

输出示例：

```text
Model:
Decoder-only Transformer

Layers:
12

Heads:
8

d_model:
512

d_ff:
2048

Context:
512

Parameters:
~XX M
```

如果论文没有报告：

```text
Not explicitly reported.
```

禁止猜测。

---

# 5. Training Hyperparameters

必须提取：

```text
optimizer
learning_rate
batch_size
effective_batch_size
weight_decay
betas
epsilon
gradient clipping
```

以及：

```text
training steps
epochs
tokens seen
```

---

# 6. Learning Rate Strategy

重点检查：

```text
constant LR
linear decay
cosine decay
```

以及：

```text
warmup
```

记录：

```text
peak learning rate
warmup steps
minimum learning rate
decay schedule
```

例如：

```text
Warmup:
2000 steps

Peak LR:
3e-4

Schedule:
Cosine decay
```

然后分析：

> 这种设置为什么适合该模型？

以及：

> 是否适合当前模型规模？

---

# 7. Batch Size

不仅记录：

```text
batch_size
```

还要区分：

```text
micro batch size
gradient accumulation
effective batch size
```

例如：

```text
micro batch = 16
gradient accumulation = 8

effective batch = 128
```

---

# 8. Optimization Techniques

检查代码和论文是否使用：

```text
gradient clipping
gradient accumulation
mixed precision
bf16
fp16
tf32
```

以及：

```text
Adam
AdamW
```

记录：

```text
beta1
beta2
epsilon
weight decay
```

---

# 9. Training Stability

论文涉及以下内容时需要重点分析：

```text
gradient explosion
gradient vanishing
loss spikes
training divergence
```

检查作者使用了什么方法解决，例如：

```text
warmup
Pre-LN
gradient clipping
smaller LR
better initialization
```

---

# 10. Model Scaling

如果论文比较不同模型规模，需要提取：

```text
layers
hidden size
heads
parameters
training tokens
```

比较：

```text
small
medium
large
```

重点回答：

> 模型增加到 12 层是否真的有必要？

以及：

> 模型容量和 SVA / syntactic ability 有什么关系？

---

# 11. Scaling Laws

如果论文涉及：

```text
model size
dataset size
compute
training tokens
```

需要分析：

> 当前模型的数据量与参数量是否匹配？

特别注意：

```text
tokens per parameter
```

但不要机械套用大型 LLM scaling laws 到小模型。

---

# 12. Dataset Analysis

所有涉及 pretraining 的论文都需要提取：

```text
dataset
dataset size
number of tokens
number of documents
language
```

例如：

```text
TinyStories
WikiText
Wikipedia
OpenWebText
Languini
synthetic corpus
```

---

# 13. Dataset Complexity

不仅记录数据规模，还分析：

```text
lexical diversity
sentence complexity
syntactic complexity
long-distance dependencies
```

特别关注：

> 数据集是否包含足够的 grammatical structures，使模型能够学习 SVA？

例如需要比较：

```text
TinyStories
vs
WikiText
```

而不只是比较 token 数量。

---

# 14. Tokenizer Analysis

必须记录：

```text
tokenizer type
vocabulary size
```

例如：

```text
BPE
SentencePiece
Unigram
WordPiece
byte-level BPE
```

进一步分析：

```text
subject tokenization
verb tokenization
```

尤其检查：

> correct verb 和 incorrect verb 是否被分成不同数量的 tokens？

因为这可能影响 SVA evaluation。

---

# 15. Vocabulary Size

记录：

```text
vocab_size
```

并分析它与：

```text
dataset size
model size
embedding parameters
```

之间的关系。

对小模型特别关注：

> vocabulary 是否过大导致大量参数浪费在 embedding？

---

# 16. Cloned Language Literature

重点检索：

```text
cloned language
artificial language
synthetic language
language duplication
cross-lingual representation
```

特别关注：

- Dufter & Schütze
- Schäfer et al.
- multilingual representation literature

---

# 17. Clone Construction

必须明确 cloned language 在哪个阶段生成：

```text
raw text
character
token
token ID
```

例如：

```text
text
↓
tokenizer
↓
original token IDs
↓
clone ID mapping
```

或者：

```text
text transformation
↓
tokenizer
```

不能混淆。

---

# 18. Original / Clone Vocabulary

必须检查：

```text
shared vocabulary?
```

还是：

```text
separate token IDs?
```

以及：

```text
vocab_original
vocab_clone
```

是否完全 disjoint。

---

# 19. Language Sampling

如果训练同时包含 original + clone：

记录：

```text
p_clone
sampling probability
language balance
```

例如：

```text
P(original) = 0.5
P(clone) = 0.5
```

分析：

> imbalance 是否影响 cross-lingual representation？

---

# 20. Subject–Verb Agreement Literature

检索：

```text
subject verb agreement transformer
syntactic agreement language model
SVA GPT
agreement attraction language models
```

关注：

```text
simple SVA
distractor
attractor
long-distance SVA
```

---

# 21. SVA Dataset

重点比较：

```text
BLiMP
SyntaxGym
CausalGym
Marvin & Linzen
custom synthetic SVA
```

需要提取：

```text
number of samples
sentence templates
difficulty
minimal pairs
```

---

# 22. SVA Metric

记录论文使用：

```text
accuracy
surprisal
probability
log probability
logit difference
```

对于 decoder-only Transformer，特别检查：

```text
logit(correct)
-
logit(incorrect)
```

---

# 23. SVA Sanity Check

必须区分：

```text
simple SVA
```

和：

```text
distractor SVA
```

不要只报告 overall accuracy。

输出：

```text
simple:
...

distractor:
...

complex:
...
```

---

# 24. Mechanistic Interpretability Literature

检索：

```text
mechanistic interpretability transformer
activation patching
causal tracing
path patching
circuit discovery
causal intervention transformer
```

重点关注与：

```text
syntax
agreement
multilingual
```

相关的工作。

---

# 25. Activation Patching

论文分析必须回答：

### Clean input

```text
是什么？
```

### Corrupted input

```text
改变了什么？
```

### Patched activation

```text
residual stream?
attention?
MLP?
head?
```

### Position

```text
subject?
distractor?
verb prediction?
all tokens?
```

---

# 26. Layer Sweep

检查是否：

```text
Layer 0
Layer 1
...
Layer N
```

全部进行 patching。

并说明：

通常是：

```text
一次 patch 一个 layer / position
```

而不是：

```text
一次同时 patch 所有层
```

---

# 27. Activation Patching Metric

优先记录：

```text
patched - corrupted
```

或：

```text
patched - corrupted
-------------------
clean - corrupted
```

并说明 paper 实际使用的是哪个。

---

# 28. Attention Head Analysis

如果论文进一步定位：

```text
Layer × Head
```

需要记录：

```text
important heads
```

以及对应 attention patterns。

但必须区分：

```text
attention correlation
```

和：

```text
causal importance
```

---

# 29. MLP Analysis

检查是否 patch：

```text
MLP output
```

或分析：

```text
MLP neurons
```

并记录 attention 与 MLP 是否承担不同功能。

---

# 30. Circuit Identification

需要明确：

> 论文所谓 circuit 究竟是什么？

例如：

```text
important nodes
important heads
edges
computational paths
```

不能把：

```text
important layer
```

直接等同于：

```text
circuit
```

---

# 31. Circuit Validation

查找：

```text
ablation
necessity
sufficiency
faithfulness
completeness
```

判断论文是否真的验证 circuit。

---

# 32. Cross-Lingual Circuit Comparison

重点提取：

```text
original circuit
clone circuit
```

如何比较。

例如：

```text
Pearson correlation
Spearman correlation
top-k overlap
Jaccard similarity
```

---

# 33. Paper Code Repository Search

**只要论文存在官方代码库，就必须同时检查代码。**

优先寻找：

```text
Official GitHub
authors' GitHub
project page
supplementary repository
```

---

# 34. Repository First-Level Analysis

首先阅读：

```text
README.md
requirements.txt
environment.yml
setup.py
pyproject.toml
```

了解：

```text
framework
dependencies
Python version
GPU requirements
```

---

# 35. Repository Structure

输出代码目录，例如：

```text
src/
├── model.py
├── train.py
├── dataset.py
├── eval.py
├── patching.py
└── visualization.py
```

然后说明每部分作用。

---

# 36. Model Code Analysis

重点找到：

```text
Transformer
Block
Attention
MLP
LayerNorm
Embedding
LM Head
```

分析代码实际使用：

```text
Pre-LN or Post-LN
activation function
dropout
bias
position embedding
weight tying
```

---

# 37. Config Analysis

优先寻找：

```text
config.py
config.yaml
train.yaml
*.json
```

提取真实实验参数。

例如：

```text
n_layer
n_head
n_embd
block_size
dropout
```

如果：

> 论文正文和代码配置不同

必须指出差异。

---

# 38. Training Script Analysis

重点检查：

```text
train.py
trainer.py
```

寻找：

```text
optimizer
LR
scheduler
warmup
weight decay
gradient clipping
batch size
gradient accumulation
mixed precision
```

---

# 39. Hidden Training Tricks

特别注意论文正文可能没有详细写出的：

```text
weight initialization
residual scaling
weight tying
gradient clipping
special scheduler
```

如果代码中存在：

必须单独指出。

这类信息对于复现尤其重要。

---

# 40. Dataset Code Analysis

阅读：

```text
dataset.py
dataloader.py
data_utils.py
```

分析：

```text
sampling
sequence packing
padding
truncation
context window
```

以及：

```text
original / clone sampling
```

---

# 41. Tokenizer Code

查看：

```text
tokenizer training
special tokens
BOS
EOS
PAD
UNK
```

以及：

```text
vocabulary extension
clone vocabulary mapping
```

---

# 42. Evaluation Code

查找：

```text
eval.py
evaluate.py
sva.py
```

特别确定：

> accuracy 到底是如何计算的？

例如：

```text
correct_logit > incorrect_logit
```

不要只根据论文描述推测。

---

# 43. Activation Patching Code

搜索：

```text
hook
activation
cache
patch
intervention
```

重点确定：

```text
hook registered where?
```

例如：

```text
block output
attention output
MLP output
residual stream
```

---

# 44. Patching Tensor Shape

阅读代码时检查：

```text
[B, T, D]
```

或：

```text
[B, H, T, D_head]
```

明确 patch 操作发生在哪个 dimension。

---

# 45. Clean Activation Cache

检查代码：

```text
clean forward
↓
cache activation
```

如何实现。

例如：

```text
forward hook
```

---

# 46. Patch Implementation

明确代码是否：

```text
activation[:, position, :]
=
clean_activation[:, position, :]
```

类似这样的逻辑。

必须将：

```text
paper description
```

映射到：

```text
actual code
```

---

# 47. Plotting Code

查看论文 heatmap 是如何生成的：

```text
layer
token
head
```

对应哪个 axis。

还需要检查：

```text
normalization
averaging
aggregation
```

---

# 48. Paper ↔ Code Mapping

所有重要方法尽量输出：

```text
Paper:
"We patch residual activations"

Code:
hooks.py → patch_residual()

Implementation:
Replace residual activation at layer l and token t.
```

---

# 49. Reproducibility Analysis

检查是否提供：

```text
seed
checkpoint
dataset split
config
environment
```

判断：

```text
easy to reproduce
partially reproducible
difficult to reproduce
```

---

# 50. Compare With Current Project

分析每篇论文后增加：

## 可以直接使用

例如：

```text
SVA evaluation metric
activation patching logic
clone construction
```

## 可以参考，但需要修改

例如：

```text
model architecture
dataset size
number of layers
```

## 不建议照搬

例如：

```text
large-model-specific optimization
multi-GPU infrastructure
```

---

# 51. Model Recommendation

如果论文对当前模型设计有帮助，需要最后给出：

```text
Current setting
→
Paper setting
→
Recommended change
```

例如：

```text
Current:
4 layers

Paper:
12 layers

Recommendation:
如果目标是获得更清晰的 layer-wise circuit，
可以扩展到 8–12 layers。
```

---

# 52. Hyperparameter Recommendation

推荐时必须说明依据。

例如：

```text
Learning rate:
当前 3e-4

Paper A:
6e-4

Paper B:
3e-4

Recommendation:
暂时保持 3e-4。
```

不要因为单篇论文使用某参数就直接要求修改。

---

# 53. Cross-Paper Comparison

当检索多篇论文时，生成对比：

| Paper | Layers | Params | Dataset | Tokens | LR | Batch | SVA | Patching |
|---|---:|---:|---|---:|---:|---:|---|---|

然后总结：

```text
共同设置
不同设置
最适合当前项目的设置
```

---

# 54. Evidence Priority

模型参数优先从：

```text
official config
```

确认。

优先级：

```text
official code config
>
paper experimental section
>
appendix
>
README
>
secondary source
```

如果代码与论文冲突：

明确指出。

---

# 55. Do Not Guess

以下信息如果没有找到：

```text
learning rate
batch size
number of samples
model dimensions
patching position
```

直接写：

```text
Not explicitly reported / Not found.
```

不要补全。

---

# 56. Search Strategy

针对当前项目主动组合搜索以下主题。

## Model

```text
small decoder-only transformer training
GPT-2 training from scratch
small language model optimization
decoder-only transformer hyperparameters
```

## Training

```text
transformer training stability
GPT learning rate warmup
decoder-only transformer initialization
small transformer scaling
```

## Syntax

```text
subject verb agreement transformer
syntactic generalization GPT
```

## Mechanistic Interpretability

```text
subject verb agreement circuit transformer
activation patching syntax
causal tracing agreement
```

## Multilingual

```text
cross-lingual circuits transformer
multilingual mechanistic interpretability
cloned language transformer
```

---

# 57. Standard Paper Analysis Output

每篇论文默认按照：

## 1. 论文解决什么问题

简要说明。

## 2. 为什么和当前项目相关

明确关联。

## 3. Model

```text
architecture:
layers:
heads:
d_model:
d_ff:
parameters:
context:
normalization:
activation:
position encoding:
```

## 4. Training

```text
dataset:
tokens:
batch:
optimizer:
learning rate:
warmup:
scheduler:
weight decay:
gradient clipping:
training steps:
```

## 5. Tokenizer

```text
type:
vocab:
special tokens:
```

## 6. Language Setup

```text
languages:
clone construction:
sampling:
```

## 7. SVA

```text
dataset:
conditions:
metric:
performance:
```

## 8. Mechanistic Interpretability

```text
method:
clean/corrupted:
patch target:
layers:
positions:
heads:
metric:
```

## 9. Main Results

只总结与当前项目相关的结果。

## 10. Code Repository

```text
repository:
framework:
important files:
```

## 11. Important Code Implementation

解释关键实现。

## 12. 对当前项目的启发

```text
直接复用：
...

值得尝试：
...

暂时不用：
...
```

---

# 58. Multi-Paper Literature Review

如果用户要求：

> “检索一下相关论文”

不要逐篇独立总结后结束。

最后必须进行综合分析：

```text
Paper A
      \
Paper B → common findings → recommendation
      /
Paper C
```

回答：

> 多篇论文共同说明了什么？

---

# 59. Architecture Comparison

如果多篇论文模型不同，需要比较：

```text
4 layers
8 layers
12 layers
24 layers
```

以及：

```text
d_model
heads
parameters
```

判断：

> 哪个模型规模最适合当前 experimental objective？

而不是：

> 哪个模型 benchmark 最强？

---

# 60. Research Priority

始终围绕当前研究目标排序。

优先级：

```text
1. 能否稳定学会 SVA
2. 是否适合 mechanistic interpretability
3. original / clone 是否公平可比较
4. circuit 是否足够丰富
5. training cost
6. benchmark performance
```

Benchmark performance 不是最高优先级。

---

# 61. Avoid Unnecessary Modernization

不要因为现代 LLM 使用：

```text
RoPE
SwiGLU
RMSNorm
GQA
FlashAttention
```

就自动建议全部加入当前模型。

首先判断：

> 是否真的有助于当前 scientific question？

当前项目更重视：

```text
simple
interpretable
controlled
reproducible
```

而不是：

```text
state-of-the-art architecture
```

---

# 62. Final Goal

这个 Skill 的最终目标不是简单回答：

> “这篇论文讲了什么？”

而是建立：

```text
论文
↓
实验设计
↓
模型配置
↓
训练技巧
↓
代码实现
↓
SVA 方法
↓
MI 方法
↓
当前项目修改方案
```

最终每次论文分析都应该帮助回答：

> **这篇论文中的哪些具体设计，可以帮助我们把当前 MLLMs 项目做得更可靠、更容易训练，同时更适合研究 original language 与 cloned language 的 SVA circuits？**