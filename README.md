# Cocktail Cortical

Cortical TRF (temporal response function) analysis of the Binaural
Cocktail EEG dataset, built on eelbrain 0.41.2 (the latest version
actually released; eelbrain never had a 0.43 release, on conda-forge
or PyPI - the highest version either has is 0.41.2).

This repo answers a specific question: how well does the cortical (slow,
speech-envelope-tracking) EEG response reflect the foreground story a
listener is attending to, the background story they're ignoring, and how
that changes with the spatial listening condition (clean / diotic /
binaural / dichotic).

It's the cortical counterpart to a subcortical analysis of the same
dataset - same recordings, same experiment design, different frequency
range and predictors (subcortical responses track the raw sound wave;
cortical responses track the slower speech envelope).

## Data

This repo analyzes the Binaural Cocktail EEG dataset: subjects listening
to interleaved male and female speakers under four spatial conditions.

Data lives outside this repo, as a sibling folder next to it, wherever
you've cloned the repo:

```
<parent folder>/
  cocktail-cortical/                 this repo
  dataset/
    cocktail/
      eeg/           original BioSemi recordings, one folder per subject:
                       eeg/s1/s1_cocktail.bdf
                       eeg/s2/s2_cocktail.bdf
                       ...
      bids/          created by data/bids_extraction.py
      stimuli/       the stimulus .wav files (male_1..12, female_1..12,
                     List_1_stim_1..12, List_2_stim_1..12)
      predictors/    created by predictors/gammatone_predictors.py
```

Every script works this location out on its own, from its own file
location (`REPO_ROOT.parent / "dataset" / "cocktail"`), so no path
needs editing and nothing here depends on your username or which
folder you cloned the repo into. If you'd rather put the dataset
somewhere else entirely, edit `DATA_ROOT` in `data/bids_extraction.py`,
`predictors/gammatone_predictors.py`, `predictors/duration.py`,
`diagnostics/explore_gammatone_settings.py`, `diagnostics/check_montage.py`,
and `analysis/experiment.py`.

Your `eeg/sX/` folders may already contain a `-bad_channels.txt` and an
`ica-ica.fif` from earlier work on this data - those were generated
against the old (non-BIDS) folder layout eelbrain used to read from, so
they won't be picked up once the recordings move to `bids/`. They're
harmless to leave in place. See **Preprocessing** below for redoing
that step under the new BIDS layout.

## Setup

```
conda env create -f environment.yml
conda activate cocktail-cortical
```

## Pipeline (run in this order)

1. **`data/bids_extraction.py`** - converts the raw `.bdf` recordings
   into the standard BIDS folder layout under `bids/`.
2. **`predictors/duration.py`** - measures each stimulus's length
   (already saved in `experiment.py`; only needed if the stimuli change).
3. **`predictors/gammatone_predictors.py`** - builds the cortical speech predictors
   (envelope and onset gammatone spectrograms) from the stimulus audio.
4. **`analysis/experiment.py`** - defines the eelbrain pipeline
   (preprocessing, epochs, predictors) that the analysis notebook
   imports, *and* is what you run to mark bad channels and fit ICA for
   every subject: `python analysis/experiment.py`. That loops over
   every subject automatically (see `preprocess_all_subjects()` in the
   file) - `make_bad_channels()` and `make_ica_selection()` each open a
   plot and pause until you close it, so the loop advances to the next
   subject the moment you're done with the current one, nothing to
   edit or re-run by hand. Results are cached to disk per subject, so
   this never redoes a subject that's already done. Importing this
   file elsewhere (`from experiment import e`, what the analysis
   notebook does) never triggers this - it only runs when the file is
   executed directly.
5. **`analysis/cortical_analysis.ipynb`** - the actual analysis: envelope
   model checks, the dichotic ear-of-presentation comparison, the
   binaural-cue comparison, and TRF/peak-time plots. Open it directly
   in Jupyter and run the cells top to bottom.

Exact eelbrain method names in `experiment.py`'s
`preprocess_all_subjects()` can vary slightly by version; see
https://eelbrain.readthedocs.io/en/stable/experiment.html for the
current API. `analysis/cortical_analysis.ipynb` is a real Jupyter
notebook file, committed as-is (clear its cell outputs before
committing changes to it, so diffs stay readable). The scripts in
`diagnostics/` are still
written in [jupytext](https://jupytext.readthedocs.io/) "percent"
format (plain `.py`, paired with a local, not-committed `.ipynb`), so
they can be opened directly as notebooks, or run top to bottom as
plain scripts.

## `diagnostics/` - one-off diagnostic scripts

Not part of the pipeline; kept for reference when something needs
double-checking.

- **`check_montage.py`** - confirms the electrode channel names and the
  custom montage file (`analysis/biosemi64mod.txt`) load correctly.
- **`explore_gammatone_settings.py`** - the scratch work used to pick the
  frequency range and band count used in `predictors/gammatone_predictors.py`.

## A note on BIDS conversion

`mne-bids` keeps BioSemi recordings in their original `.bdf` format
during conversion (confirmed: `data/bids_extraction.py`'s output is
`bids/sub-XX/eeg/sub-XX_task-cocktail_eeg.bdf`), which is what
`analysis/experiment.py`'s `RawSource` reads.
