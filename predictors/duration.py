# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Measure stimulus durations
#
# analysis/experiment.py needs to know how long each audio stimulus is,
# so it knows how long a segment of EEG to extract for each one
# (SEGMENT_DURATION). This script measures those durations directly from
# the .wav files and prints them in a form you can paste straight into
# experiment.py.
#
# You only need to run this if the stimulus files change - the current
# values are already saved in experiment.py.

# %%
from pathlib import Path

from eelbrain import load


# The dataset lives outside this repo, in a sibling "dataset/cocktail"
# folder next to it. Working this out from this file's own location
# means the script runs the same way no matter whose computer, or which
# folder, the repo is cloned into.
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ isn't defined when a cell is run interactively in a
    # Jupyter kernel; fall back to assuming Jupyter's working directory
    # is this file's own folder.
    REPO_ROOT = Path.cwd().parent
DATA_ROOT = REPO_ROOT.parent / "dataset" / "cocktail"
STIMULUS_DIR = DATA_ROOT / 'stimuli'

# %%
print('SEGMENT_DURATION = {')
for speaker in ['male', 'female']:
    for i in range(1, 13):
        name = f'{speaker}_{i}'
        wav = load.wav(STIMULUS_DIR / f'{name}.wav')
        print(f"    '{name}': {wav.time.tstop:.3f},")
print('}')
