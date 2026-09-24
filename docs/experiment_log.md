# Public experiment log

Last updated: 2026-09-24

## Research question

Does a decoder-only Transformer develop a shared subject-number mechanism when
trained on English and a cloned language whose non-padding token IDs occupy a
fully separate vocabulary partition?

The causal test copies an activation from a clean source-language prompt into
the corresponding corrupted target-language execution. A positive
clean-minus-number-control effect means the source activation moves the target
verb preference in the source subject-number direction more than the matched
control activation does.

## Fixed experimental setup

- Dataset: English Wikipedia `20231101.en`, revision
  `e6057dc557255a03c9c3c47ceab0eb44353b1bc5`
- Training text: 100,001,395 words; validation/test: approximately 1M words each
- Tokenizer: 16,000-token SentencePiece BPE
- Model vocabulary: 32,000; clone non-padding IDs use a 16,000 offset
- Model: 12 layers, 8 attention heads, hidden size 512, FFN size 2,048
- Parameters: 54,344,704
- Context length: 256
- Original/clone sampling probability: 0.5/0.5
- Training seed: 42
- Completed training: 77,560 steps and 635,371,520 tokens
- Validation-selected best checkpoint: step 77,500

The tokenizer SHA-256 is
`b656b4abdde9f0fc60adde753725fde16bfe8f263e5347afbb675c31870a880b`.
The best-checkpoint SHA-256 is
`aa071024c7d778ed3c7cc2aab2c6442d9faf1f9b6b75611301088db389b6ebc1`.

## Controlled SVA v2

The versioned evaluation suite contains 400 development pairs and 3,200 test
pairs. Development and test lexicons are disjoint. Every answer is one token,
and the clean/corrupted prompt pair changes only the subject-number form while
keeping token boundaries aligned.

The four constructions are `simple`, `pp_attractor`, `object_relative` and
`subject_relative`. The last three form the pooled distractor analysis.

At the best checkpoint:

| Metric | Original | Clone |
|---|---:|---:|
| Prompt accuracy | 74.41% | 73.23% |
| Pair accuracy | 48.88% | 46.47% |
| Held-out test perplexity | 36.910 | 36.851 |

## Patching decisions

- Four directions are evaluated: original→original, clone→clone,
  original→clone and clone→original.
- Cross-language recovery always uses the target-language clean/corrupted
  denominator.
- The trajectory reports raw `ΔLD` clean-minus-control effects because early
  normalized recovery can be unstable near a zero denominator.
- The fixed 1,280-pair cohort is identical at all checkpoints and is not
  reselected according to checkpoint success.
- Controls are `clean`, `opposite-number`, `same-number-shuffled` and
  `opposite-number-shuffled`.
- L8H3 was fixed before the v2 test analysis. Additional development-selected
  heads remain exploratory.
- Uncertainty uses 10,000 task-stratified paired bootstrap iterations, with a
  prompt-cluster bootstrap as a sensitivity analysis.

## Main result

At L8H3 on 888 pooled distractor pairs, the best-checkpoint
clean-minus-opposite-number-shuffled effects are:

| Direction | Mean effect | Prompt-cluster 95% CI |
|---|---:|---:|
| Original→clone | 0.0399 | [0.0311, 0.0489] |
| Clone→original | 0.2342 | [0.2234, 0.2452] |

PP attractor and subject-relative constructions show positive transfer in both
directions. Object-relative transfer is positive for clone→original but remains
unclear for original→clone.

The opposite-number-shuffled pooled trajectory is:

| Step | Original→clone | Clone→original |
|---:|---:|---:|
| 5,000 | -0.0021 | 0.0040 |
| 15,000 | -0.0014 | 0.0464 |
| 30,000 | 0.0294 | 0.1566 |
| 50,000 | 0.0776 | 0.1738 |
| 65,000 | 0.0603 | 0.2186 |
| 77,500 | 0.0399 | 0.2342 |

These observations support a local, asymmetric cross-language causal transfer.
They do not establish a complete shared circuit or a precise circuit-formation
time.

## Limitations

1. The study contains one model and one training seed. Bootstrap intervals do
   not estimate training-seed variance.
2. The cloned language shares grammar, semantic distribution, parameters and
   positional structure with English. Results do not directly generalize to a
   pair of natural languages.
3. The fixed patching cohort covers 40% of the test suite. Prompt clustering
   handles exact repeated prompts but not all lexical/template dependencies.
4. L8H3 patching shows transferable information at one site; no necessity
   ablation or multi-head joint patching has yet been run.
5. The six-checkpoint trajectory is exploratory and uses pointwise intervals.
   It does not provide a multiplicity-adjusted onset test.
6. Tokenizer training skipped unusually long article lines, although language
   model tokenization covered the complete prepared corpus.

## Reproducibility record

The compact evidence, hashes, statistics and figures are versioned in
`reports/cross_language_sva/`. Raw checkpoints and activation arrays are locally
generated artifacts and are excluded from Git. The exact commands and three
reproduction levels are documented in the repository README.
