# Cross-language SVA report evidence

This directory contains the Chinese report, compact evidence tables and the ten
PNG figures used by the report. It intentionally excludes model checkpoints,
tokenizer files, raw activation arrays and duplicate HTML/PDF exports.

The plotting source is
[`../../notebooks/cross_language_dashboard.ipynb`](../../notebooks/cross_language_dashboard.ipynb).
The tracked notebook has no execution outputs and reads only `data/`.

## Redraw the figures

```bash
python -m pip install -e '.[report]'
python -m nbconvert \
  --to notebook \
  --execute notebooks/cross_language_dashboard.ipynb \
  --output /tmp/cross_language_dashboard.executed.ipynb
```

The notebook checks the compact CSV hashes in `data/manifest.json` before
plotting. It does not load checkpoints or raw experiment outputs.

## Rebuild the compact evidence

With locally generated tokenizer, checkpoints and raw outputs in their configured
locations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/build_report.py
```

The builder checks dataset, tokenizer and checkpoint identities, validates the
fixed cohort across all six checkpoints, and writes machine-portable metadata.

## Statistical interpretation

- `mean` is a clean-minus-control raw ΔLD effect, not an accuracy percentage.
- `row_ci_*` is the task-stratified paired bootstrap interval.
- `cluster_ci_*` resamples prompt clusters as a sensitivity analysis.
- Both use 10,000 iterations and seed 42; neither estimates training-seed
  variance.
- Trajectory intervals are exploratory pointwise intervals and do not define a
  circuit-onset checkpoint.
- Historical zero bootstrap-tail counts mean no opposite-tail draw was observed;
  they are not exact p-values of zero.
