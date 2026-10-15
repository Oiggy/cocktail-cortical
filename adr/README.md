# ADR: artifact detection and reconstruction

Supervised, non-neural artifact detection and type-routed reconstruction for
continuous speech-listening EEG. `PROJECT_PLAN.md` is the specification;
`PROGRESS.md` tracks which milestones are done and what is still open.

## Setup

```
conda env create -f environment.yml
conda activate adr
pytest
```

## Data

Nothing in `data/` or `results/` is committed. Datasets live in the sibling
`dataset/` folder next to this repo (same convention as the parent
cocktail-cortical analysis); `configs/paths.yaml` says where each one is.
Override a path on one machine with a git-ignored `configs/paths.local.yaml`.

## Layout

- `configs/` paths, seed, channel lists, later the hyperparameter grids
- `src/adr/` the package (io, preprocessing, envelope, synth, detection,
  reconstruction, baselines, trf, stats)
- `scripts/` numbered entry points, run in order
- `tests/` unit tests for every module
- `splits/` frozen split files (committed, never edited after tuning starts)
