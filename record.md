# MLLMs 研究进度与实验记录

最后更新：2026-09-16

## 1. 研究问题与老师要求

本项目研究一个 decoder-only Transformer 在完全不共享 subword IDs 的两种等价语言中，是否会形成共享的 subject–verb agreement（SVA）number circuit。

训练语料由 English Wikipedia 和它的 cloned language 组成。两种语言共享语法与语义分布，但 clone token IDs 位于独立的 vocabulary partition。核心问题不是两种语言的 accuracy 是否接近，而是：

> 将 original language 中表示 subject number 的 activation 直接注入 clone language 的 corrupted execution，能否因果改变 clone verb prediction？反方向是否也成立？

老师邮件提出的要求：

- 从零训练一个 small language model，优先使用 decoder-only architecture。
- Original/clone 使用互不重叠的 tokenizer partitions，不依赖 shared subwords。
- 模型必须具备有意义的 next-token prediction 和 SVA 能力，否则 mechanistic analysis 没有解释价值。
- 使用 SVA activation patching 检验两种语言是否使用相同的 number feature。
- 明确定义 distractor tasks 及其作用。
- 可以沿 checkpoints 研究 circuit development，但“无 circuit → language-specific → language-agnostic”只是待检验假说，不是预设结论。
- PoS fine-tuning 可以作为扩展，但不是当前主线。

### SVA constructions

- `simple`：subject 与 verb 之间没有 intervening noun，是容易条件。
- `pp_attractor`：subject 与 verb 之间加入 prepositional phrase noun。
- `object_relative`：加入 object-relative clause noun。
- `subject_relative`：加入 subject-relative clause noun。

后三类用于检验模型是否依据 syntactic subject number，而不是被中间或邻近 noun 的 number 吸引。

## 2. 数据、模型与训练配置

### Wikipedia 数据

- Dataset：English Wikipedia `20231101.en`
- Revision：`e6057dc557255a03c9c3c47ceab0eb44353b1bc5`
- Train：100,001,395 words，139,394 articles
- Validation：1,000,151 words
- Test：1,000,293 words
- Tokenized train：158,841,802 base tokens

Tokenizer 为 16,000-token SentencePiece BPE。Clone language 对所有非-PAD token IDs 加 16,000 offset，因此 model vocabulary 为 32,000，两种语言没有 shared token IDs。

Tokenizer SHA-256：

```text
b656b4abdde9f0fc60adde753725fde16bfe8f263e5347afbb675c31870a880b
```

### 模型与训练

- Architecture：12-layer decoder-only GPT
- Parameters：54,344,704
- Hidden size：512
- Attention heads：8
- FFN size：2,048
- Context length：256
- Original/clone sampling probability：0.5 / 0.5
- Training seed：42
- Completed steps：77,560
- Tokens seen：635,371,520
- Original/clone tokens：317,499,904 / 317,871,616
- Best checkpoint：step 77,500
- Best checkpoint SHA-256：`aa071024c7d778ed3c7cc2aab2c6442d9faf1f9b6b75611301088db389b6ebc1`

Best validation results：

| Language | Loss | Perplexity |
|---|---:|---:|
| Original | 3.6170 | 37.23 |
| Clone | 3.6104 | 36.98 |

Held-out test loss/PPL 尚未正式运行；对应 CLI 与 Slurm stage 已准备好。

## 3. 已完成实验与结果

### 3.1 Controlled SVA v1

固定 test set 包含 3,200 counterfactual pairs，每个 construction 800 pairs。Best checkpoint 结果：

| Metric / task | Original | Clone |
|---|---:|---:|
| Prompt accuracy | 70.81% | 69.06% |
| Pair accuracy | 41.78% | 38.16% |
| Mean LD | 1.3315 | 0.9762 |
| Simple accuracy | 82.94% | 80.50% |
| PP attractor | 69.50% | 67.88% |
| Object relative | 66.63% | 63.63% |
| Subject relative | 64.19% | 64.25% |

Original/clone prompt accuracy gap 为 1.75 percentage points。评估没有 random sampling；在固定模型与固定 test set 上，该差距不能解释为 evaluation sampling noise。由于只有一个 training seed，也不能断言它是跨训练稳定的语言能力差异。

已完成 step 5k–75k、每 5k 一次的 SVA sweep，加上 best checkpoint 共 16 个点。SVA 随训练总体增强但明显波动；这是 behavioral trajectory，不是 circuit trajectory。

### 3.2 现有 within-language activation patching

Best checkpoint 的 v1 joint sanity pairs 为 812；其中 618 pairs 满足 patching 对齐和再次 sanity check。任务分布：simple 327、PP attractor 134、object relative 72、subject relative 85。

现有同语言 patching 是：

```text
original clean → original corrupted
clone clean    → clone corrupted
```

| Component | Original/clone Pearson r | Mean absolute difference |
|---|---:|---:|
| Residual stream | 0.9919 | 0.0089 |
| MLP output | 0.9714 | 0.0077 |
| Attention output | 0.6992 | 0.0032 |
| Attention heads | 0.6456 | 0.0007 |

共同最强候选为 prediction-position `L8H3`，original/clone mean recovery 为 0.1324/0.1348。

仅保留两个答案都为 single-token 的 265 pairs 后，L8H3 仍是共同候选，original/clone recovery 为 0.2447/0.2028。因此它不是完全由 multi-token artifact 制造，但 effect size 与 ranking 明显受 tokenization 影响。

### 3.3 Controlled SVA v2 与跨语言基础设施

已使用 Wikipedia tokenizer 生成并冻结 `data/sva/controlled_v2/`：

- Dev：400 pairs，四类各 100
- Test：3,200 pairs，四类各 800
- Dev/test lexical inventories 互斥
- clean/corrupted answers 均为 single token
- clean/corrupted prompts 只改变 subject number
- token boundaries 与 prompt lengths 对齐
- Dataset generation seed：42

已实现四个 patching directions：

```text
original → original
clone    → clone
original → clone
clone    → original
```

Cross-language recovery 使用 target-language denominator：

```text
recovery = (LD_patched - LD_target_corrupted)
           / (LD_target_clean - LD_target_corrupted)
```

已实现 `clean`、`opposite-number`、`same-number-shuffled` 和 `opposite-number-shuffled` controls，以及 component filtering、prediction-position-only scan、fixed cohort、joint-sanity selection、per-site `patched_ld`/`delta_ld`/`recovery`、provenance hashes、task-stratified paired bootstrap 和 Holm correction。

真实 best checkpoint 的 CPU smoke test 已完成：original→clone、4 fixed pairs、prediction-position heads、clean 与 opposite-number control 均成功写出结果。该运行只验证 end-to-end correctness，不作为研究证据。

## 4. 当前可以与不可以支持的结论

### 可以支持

- 模型已完成稳定预训练，original/clone validation language-model quality 接近。
- 模型在四类 controlled SVA 上均表现出高于 chance 的行为能力。
- SVA behavior 随训练逐步增强，但不是单调过程。
- Original 与 clone 的 within-language causal maps 高度相似。
- L8H3 是值得进行 confirmatory cross-language intervention 的预注册候选。

### 不能支持

- 不能声称已经发现 shared cross-language circuit。
- Heatmap correlation 不能替代 original→clone / clone→original causal intervention。
- 单一 seed 不能估计跨训练方差。
- 不能把 v1 aggregate recovery 当成 tokenizer-independent effect。
- 不能把 success-only sanity subset 的结果推广到所有 SVA examples。
- 在正式 v2 cross-language jobs 完成前，不能判断老师提出的三阶段 circuit-development 假说。

## 5. 已解决失败与工程状态

- Wikipedia data jobs `470120`、`470121` 因 Hugging Face BuilderConfig 问题失败；`470122` 成功完成。
- SVA job `470161` 因单-token假设遇到 `arrives` 的 multi-token encoding 失败；实现 sequence log-probability 后，`470162` 成功。
- Training job `470124` 完成 77,560 steps。
- Best patching job `470177` 完成 v1 within-language full run。
- 当前测试：26/26 passed。
- 当前分支：`wikipedia`；已提交基线 commit 为 `29020fa`，本轮 v2/cross-language 工作尚未提交。

## 6. 实验限制与开放问题

1. 只有 training seed 42；本轮使用 bootstrap/control，多-seed 留作扩展。
2. SentencePiece tokenizer training 跳过了 42,276 / 139,394 个超长 article lines。LM tokenization 仍覆盖完整语料，但 tokenizer learning corpus 有偏差；本轮不重训模型，必须披露。
3. v1 patching 的 618 pairs 中仅 265 pairs 的两个候选答案均为 single token；v2 已消除此混淆。
4. v1 overall patch maps 被 simple samples 主导；v2 必须按 task 分层报告。
5. 大型 checkpoints、token streams 与 per-example outputs 不进入 Git；需要保存 compact summaries 与 manifest。
6. 本地 repository 通过 `/workspace/...` symlinks 访问集群 artifacts，迁移环境时必须重新建立路径或显式传参。
7. Held-out test PPL 尚待运行。

## 7. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-16 | 主问题改为双向 cross-language causal patching | 与老师问题严格对齐；within-language similarity 只作基线 |
| 2026-09-16 | 新建 tokenizer-compatible controlled_v2 | 消除 multi-token answer 对 recovery 的结构性混淆 |
| 2026-09-16 | 使用 raw hidden-state copy | 检验模型是否自然形成 shared hidden space |
| 2026-09-16 | 固定样本 raw ΔLD 为 trajectory 主指标 | 避免 checkpoint success subset 改变造成 composition confound |
| 2026-09-16 | 单 seed + bootstrap + empirical controls | 在2–4周范围内优先回答核心因果问题 |
| 2026-09-16 | L8H3 为预注册 confirmatory site | 来自独立的 v1 within-language结果；其他 sites 仅作 dev discovery |

## 8. 下一步实施清单

- [x] 生成并版本化 controlled_v2。
- [x] 实现四方向 source→target patching。
- [x] 实现 controls、component selection、fixed cohort 与统计函数。
- [x] 真实 checkpoint end-to-end smoke test。
- [ ] 提交 `cluster/sva-v2-array.sbatch`，完成六阶段点与 best 的 v2 SVA。
- [ ] 用 best checkpoint + dev split 运行 full-grid discovery。
- [ ] 用 best checkpoint + test split 运行 L8H3 confirmatory 与 controls。
- [ ] 对 L8H3 和 dev 预先选出的 sites 运行10,000次 paired bootstrap。
- [ ] 提交六 checkpoint prediction-position head trajectory。
- [ ] 运行 best checkpoint held-out test PPL。
- [ ] 归档 compact summaries、figures、checksums 和 Slurm job IDs。
- [ ] 更新 dashboard，使其优先读取 v2 summaries，并在缺少大文件时仍可执行。

### Claim rule

只有在以下条件全部满足时，才报告 shared-circuit 正面证据：

1. Pooled distractor effect 在 original→clone 与 clone→original 两个方向都高于 empirical null；
2. 至少两个 distractor constructions 复现正效应；
3. 报告95% CI、effect size、sample count、coverage 和 Holm-adjusted p-values；
4. 同语言 positive control 正常，opposite-number null 不产生同等 recovery。

否则结论为：未发现足够的跨语言因果共享证据。

## 9. 实验注册表

| Experiment | Config / command | Checkpoint | Job | Status |
|---|---|---|---|---|
| Wikipedia data | `cluster/prepare-data.sbatch` | — | 470122 | completed |
| Wikipedia training | `gpt12_wikipedia_clone.yaml` | last/best | 470124 | completed |
| SVA v1 best | `sva_wikipedia.yaml` | best@77,500 | 470162 | completed |
| SVA v1 sweep | same config | 5k–75k | 470178 | completed |
| Within-language patching v1 | `activation_patching_wikipedia.yaml` | best@77,500 | 470177 | completed |
| Cross-language v2 smoke | confirm config, 4 pairs, CPU | best@77,500 | local | validation only |
| SVA v2 sweep | `cluster/sva-v2-array.sbatch` | 5k,15k,30k,50k,65k,75k,best | TBD | pending |
| Cross-language v2 dev | `activation_patching_wikipedia_v2_dev.yaml` | best@77,500 | TBD | pending |
| Cross-language v2 confirm | `activation_patching_wikipedia_v2_confirm.yaml` | best@77,500 | TBD | pending |
| Cross-language trajectory | `cluster/patch-trajectory-v2-array.sbatch` | 5k,15k,30k,50k,65k,best | TBD | pending |
| Held-out test PPL | `STAGE=test-ppl` | best@77,500 | TBD | pending |

## 附录 A：老师邮件原文

Dear Jayi,

thanks for your honesty.

How about this:

1. you (more or less) reproduce a paper similar to this one: https://arxiv.org/pdf/1912.07840

That is: you pre-train a small language model from scratch on language 1 (e.g., English) and a "copy" of language 1, which uses a separate tokenizer partition.
Similar work is done by Dufter and Schütze 2020 and Schäfer et al. 2024. Both have published their code, and it is relatively easy to get it running.
(I personally would prefer if you took a decoder-only model, since this is what current research is about, but encoder-only models would also work.)
2. You take a grammatical task, say, "subject verb agreement" or something like this (we can discuss this if you want). Now, we do patching from lang1 to the cloned language. Question: Do both languages use the same "feature" to determine number? I.e., if we change the number in the subject in lang1, can this also effect the verb prediction in lang2?

This would - I hope - be a an interesting question using patching, which we covered in our seminar, and it still builds on the two languages idea.

Please let me know what you think. If you are not happy with it, we can find another solution. And if you want to do me a favor, think for yourself. And if it makes you feel more comfortable, write in your mother tongue and let the model translate it to English. Don't let the model write the mail from the start.

Best,

Frederick

---

Hi,

comments in line:

On 8/6/26 11:52, Ji, Jiayi wrote:
Yes, I think the small difference is caused by random sampling. And I did not run any "cross-lingual transfer whatever" thing. I just want to do a sanity check to see whether the baseline had learned subject–verb agreement before performing activation patching.

Okay.
In my understanding, distractor tasks are more difficult because the sentence structures are more complex than simple sentences, and there is an intervening noun between the subject and the verb that may distract the models.

What specifically IS your distractor task and what purpose does it serve?

And now model's accuracy is already around 60%–78% on the simpler SVA subtasks. So I think maybe I can run experiments in some basic data are enough. But model’s generalization ability maybe become weaker.

I understand you are not intersted in model pretraining or finetuing. But I am still new to this field, and building a model from scratch was not very easy for me. So I was thinking about whether fine-tuning or adding some other methods to the baseline could improve the accuracy. But this may move away from our research topic, I will focus on the activation patching experiments first.

Well, I mean, you CAN of course do some fine-tuning and THEN see whether the model uses the same parts for its decision. E.g., fine-tune on PoS tagging data, show that the same parts of the model are relevant. But I think this has been done very often in the past, it's an additional step you need to do first etc.

For further activation patching experiments, I found the paper On the Similarity of Circuits across Languages: A Case Study on the Subject–Verb Agreement Task, which may provide a useful reference for comparing the circuits identified in lang1 and lang2.

Yes, the paper may be good for your purposes. The main thing would be: Does the model STILL develop shared circuits, EVEN WITHOUT shared subwords? You could even trace how the circuits develop across checkpoints, similar to what I did in Riemenschneider and Frank (2025). Then hypothesized phases could be: (1) no circuits exist, (2) brittle, language-specific circuits exist, (3) generalized, language-agnostic circuits exist. This is just a hypothesis and I genuinely don't know whether this will be true.

Our faculty does not provide general personal access to a computing cluster for students, so I trained the model on my laptop with small datasets. If you can provide any help, I would really appreciate it.

So, at the Institute for Computational Linguistics, we have CLuster, which is a server for students and researchers. If you do not have access, you can contact the GT (Gruppe Technik: technik@cl.uni-heidelberg.de) and they'll give you access. There is also the BWUniCluster (https://wiki.bwhpc.de/e/BwUniCluster3.0), but I don't really have experience with this one, since I personally work on another server.

Of course, I don't expect you to pre-train a really powerful model for a seminar project. But at the same time, the model should be able to meaningfully predict next words, that is, it should not be complete crap. Otherwise, your analysis won't be really meaningful.

Let me know if you need any help.

Best,

Frederick
