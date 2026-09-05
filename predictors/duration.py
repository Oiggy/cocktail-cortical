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


DATA_ROOT = Path("~/Data/BinauralCocktail").expanduser()
STIMULUS_DIR = DATA_ROOT / 'stimuli'

# %%
print('SEGMENT_DURATION = {')
for speaker in ['male', 'female']:
    for i in range(1, 13):
        name = f'{speaker}_{i}'
        wav = load.wav(STIMULUS_DIR / f'{name}.wav')
        print(f"    '{name}': {wav.time.tstop:.3f},")
print('}')
