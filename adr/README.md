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

Machine-specific dataset paths go in `configs/paths.local.yaml` (git-ignored),
e.g.

```yaml
datasets:
  dataset1: /path/to/aad_dataset_kuleuven_v2
  physiomotion: /path/to/physiomotion_ds006386
  phantom: /path/to/phantom_ds004784
```

## Running

Every step has a script (batch runs, writes to `results/`) and a notebook
(same functions, results inline). The logic lives in the `adr` package, so
both give identical results.

| Step | Script | Notebook |
|---|---|---|
| Download data | `scripts/01_download_dataset1.py`, `03_download_openneuro.py` | - |
| Data checks (VERIFY items) | `02_verify_dataset1.py`, `04_verify_openneuro.py` | `notebooks/01_data_checks.ipynb` |
| Long-term TRF (milestone 3) | `05_long_term_trf.py` | `notebooks/02_long_term_trf.ipynb` |
| Dataset III synthesis (milestone 4) | `06_synthesize_dataset3.py` | `notebooks/03_dataset3_synthesis.ipynb` |
| Detectors (milestone 5) | `07_extract_features.py`, `08_train_detectors.py [--variant norm]`, `09_phantom_transfer.py` | `notebooks/04_detectors.ipynb` |
| Templates and injection (milestone 6) | `10_build_templates.py` | `notebooks/05_templates_injection.ipynb` |

Scripts: from `adr/`, run e.g. `python scripts/05_long_term_trf.py` with the
`adr` environment active. Later steps read earlier outputs (milestone 4
needs `results/section_a/long_term_trf.npz`; notebook 04 loads the trained
detectors), so run them in order the first time.

Notebooks: open them in VS Code and pick the `adr` conda environment as the
kernel. Slow steps are behind flags at the top of each notebook (e.g.
`RUN_TRAINING = False` loads saved results); `SAVE = True` writes outputs to
`results/`.

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
