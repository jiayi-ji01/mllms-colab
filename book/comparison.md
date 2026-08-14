# 对比与质疑

## 与 TinyStories 模型比较

下表使用两次实验都存在的 conditional-logprob 指标。由于数据、模型、tokenizer 和
训练量同时改变，它只能描述结果差异，不能识别因果来源。

| Original 指标 | TinyStories | BabyLM | 变化 |
|---|---:|---:|---:|
| Overall | 57.30% | 64.12% | +6.82 |
| Relational-noun distractor | 33.80% | 35.10% | +1.30 |
| Relative-clause distractor | 38.30% | 34.70% | -3.60 |
| Irregular agreement 1 | 57.10% | 67.50% | +10.40 |
| Irregular agreement 2 | 69.40% | 83.90% | +14.50 |
| Regular agreement 1 | 69.90% | 75.80% | +5.90 |
| Regular agreement 2 | 75.30% | 87.70% | +12.40 |

BabyLM 模型的总体提升主要来自简单和词形相关的子任务。最困难的 relative-clause
distractor 不升反降，因此“整体分数提高”不能证明层级句法能力提高。

## 当前实验可以支持什么

- 训练稳定完成，loss 与 PPL 持续改善。
- Original 和 clone 的总体语言模型能力接近。
- 更大模型与 BabyLM 设置改善了总体 conditional SVA。
- 两种语言共享相似的成功模式和失败模式。

## 当前实验不能支持什么

- 不能声称模型已经掌握强健的层级化 SVA。
- 不能把提升单独归因于 BabyLM，因为同时修改了四个实验变量。
- 不能用一次随机种子估计结果方差。
- 不能把 1,655 个过滤后的 strict 样本称为完整 6,000 样本结果。
- 不能直接用 PPL 与旧模型比较，因为 vocabulary 与 tokenization 不同。

## 仍缺少的证据

1. 独立 test-set loss/PPL；
2. 至少三个随机种子；
3. 不同训练阶段的 SVA 曲线；
4. BabyLM 模型的 activation patching；
5. tokenizer、数据 revision、manifest 和 token counts 的完整归档；
6. 一个只改变单个变量的消融实验。

## 推荐下一步

优先使用两个 distractor 子任务中原始与 clone 都判断正确的 controlled pairs 做
activation patching，分别检查 residual stream、attention output、MLP output 和各
attention head。这样能够回答模型在哪一层开始追踪错误的最近名词，而不是只得到另一个
总体 accuracy。
