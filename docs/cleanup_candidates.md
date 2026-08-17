# Cleanup candidates (review before deletion)

No item in this list is deleted automatically. Sizes below reflect the local
worktree at the time of inspection.

## Local generated artifacts

- `output/assets/tinystories_100m/` — 192 MB, old TinyStories data/tokenizer asset.
- `output/assets/babylm_tokenizer/` — 311 MB, old BabyLM tokenizer asset.
- `output/runs/gpt12_tinystories_clone_colab/` — 408 MB, old training/evaluation output.
- `output/gpt12_babylm_clone_colab/` — 1.8 GB, old training/evaluation output.
- `__pycache__/`, `scripts/__pycache__/`, `src/**/__pycache__/`,
  `tests/__pycache__/` — regenerable Python bytecode.
- `src/mllms_cloned_language.egg-info/` — regenerable editable-install metadata.

## Check on the cluster before deletion

- `data/blimp/` — old BLiMP downloads/intermediate files, if present.
- `data/causalgym/` — old CausalGym temporary pairs, if present.
- old `outputs/` run directories not referenced by a paper/report.
- abandoned Slurm logs, plots and interrupted `.partial` files.

## Explicitly retained

- `data/sva/controlled_v1/` and all controlled-SVA generation/evaluation/plotting.
- activation-patching source and configs.
- tokenizer, cloned-language, model and training source.
- TinyStories/BabyLM source and experiment configs as reference implementations.
- `book/` historical report material.
