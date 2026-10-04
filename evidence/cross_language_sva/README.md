# Compact cross-language SVA evidence

The `data/` directory contains the recorded CSV/JSON summaries behind the
plotting notebook. It includes full-test behavior, the fixed patching cohort,
controls, checkpoint trajectories, exploratory head maps, and provenance.
The notebook checks every file against `data/manifest.json` before drawing
figures. It writes those figures to a local, ignored reports directory.

The manifests preserve hashes and paths from the original analysis run.
Some code paths in `analysis_code_hashes` refer to modules used at that time;
the current public scripts have since been reorganized. Those historical
hashes identify the recorded analysis and should not be replaced with hashes
of the reorganized scripts. The `compact_hashes` entries verify the files in
this directory. Model checkpoints and raw activation arrays are not included.
