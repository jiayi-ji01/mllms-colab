# BabyLM Original + Cloned Language 实验

本书展示 12 层 GPT-2 style decoder-only Transformer 在 BabyLM 100M words 上的
预训练结果，以及 original language 和 cloned language 的 subject–verb agreement
(SVA) 表现。

## 一句话结论

训练过程稳定，original 与 clone 的语言模型表现几乎一致；模型在简单 SVA 上有明显
提升，但在 relational-noun 和 relative-clause distractor 上只有约 31%–36% accuracy，
说明它仍然依赖最近名词，尚未稳定学习层级化的主谓一致关系。

## 关键结果

| 项目 | 结果 |
|---|---:|
| 模型参数 | 54,344,704 |
| 实际训练集 tokens | 135,639,868 |
| 总 seen tokens | 271,286,272 |
| Optimizer steps | 33,116 |
| Original / clone token 比例 | 49.96% / 50.04% |
| Final original validation loss | 3.7998 |
| Final clone validation loss | 3.8036 |
| Final original / clone PPL | 44.69 / 44.86 |
| Strict SVA accuracy | 55.23% / 55.35% |
| Conditional SVA accuracy | 64.12% / 63.83% |

## 如何阅读

- [预训练](training.md)：配置、loss、PPL、学习率和梯度。
- [SVA 评估](sva.md)：严格 verb-logit 与 conditional-logprob 结果。
- [对比与质疑](comparison.md)：与 TinyStories 对比，并说明结果不能证明什么。
- [构建与更新](reproduce.md)：本地或 Colab 中打开这本 Jupyter Book。

## 结果边界

当前下载结果不包含 BabyLM activation patching、独立 test-set PPL、多随机种子实验，
也没有保存 tokenizer 文件。因此本书只报告已经实际生成的结果，不把尚未运行的分析
当作实验发现。
