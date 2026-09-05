# Cocktail Cortical

Cortical TRF (temporal response function) analysis of the Binaural
Cocktail EEG dataset, built on eelbrain 0.43.

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

This repo analyzes the same EEG dataset as the original project it was
adapted from: 12 subjects (numbered 3-14) listening to interleaved male
and female speakers under four spatial conditions. It expects the data
on disk (not committed to git) at:

```
~/Data/BinauralCocktail/
  raw_data/      original BioSemi .bdf files, one per subject (S3.bdf, S4.bdf, ...)
  bids/          created by data/bids_extraction.py
  stimuli/       the 24 stimulus .wav files (male_1..12, female_1..12)
  predictors/    created by predictors/gammatone.py
```

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
3. **`predictors/gammatone.py`** - builds the cortical speech predictors
   (envelope and onset gammatone spectrograms) from the stimulus audio.
4. **`analysis/experiment.py`** - not run directly; it defines the
   eelbrain pipeline (preprocessing, epochs, predictors) that the
   analysis notebook imports.
5. **`analysis/cortical_analysis.py`** - the actual analysis: envelope
   model checks, the dichotic ear-of-presentation comparison, the
   binaural-cue comparison, and TRF/peak-time plots.

`analysis/cortical_analysis.py` and the scripts in `dev/` are written in
[jupytext](https://jupytext.readthedocs.io/) "percent" format, so they
can be opened directly as Jupyter notebooks, or run top to bottom as
plain scripts.

## `dev/` - one-off diagnostic scripts

Not part of the pipeline; kept for reference when something needs
double-checking.

- **`check_montage.py`** - confirms the electrode channel names and the
  custom montage file (`analysis/biosemi64mod.txt`) load correctly.
- **`explore_gammatone_settings.py`** - the scratch work used to pick the
  frequency range and band count used in `predictors/gammatone.py`.

## A note on BIDS conversion

`mne-bids` can't write `.bdf` recordings as-is, so
`data/bids_extraction.py` lets it convert them to BrainVision format
(`.vhdr`/`.eeg`/`.vmrk`), which is what `analysis/experiment.py` expects
to find. After running the extraction once, double check the file
extension actually produced under `bids/sub-XX/eeg/` - if your installed
`mne-bids` version converts to something else, update the `RawSource`
pattern near the top of `analysis/experiment.py` to match.
