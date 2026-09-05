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
# # Explore gammatone spectrogram settings
#
# This is a scratch notebook for visually comparing gammatone spectrogram
# settings (frequency range, number of bands) before committing to the
# ones used in predictors/gammatone.py. It doesn't save anything - it's
# just for looking at the output.

# %%
from pathlib import Path

import numpy
from eelbrain import *


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

# %% [markdown]
# ## Compare frequency ranges
#
# The first 10 seconds of one male and one female stimulus, using an
# 80-15000 Hz range (the full audible range) versus 80-8000 Hz (the
# range predictors/gammatone.py actually uses, since most speech energy
# falls below 8000 Hz).

# %%
spectrograms_full_range = []
for speaker in ['male', 'female']:
    wav = load.wav(STIMULUS_DIR / f'{speaker}_1.wav').sub(time=(0, 10))
    spectrograms_full_range.append(gammatone_bank(wav, 80, 15000, 128, location='left', tstep=0.001))

plot.Array(spectrograms_full_range, columns=1, axh=4, w=20)

# %%
spectrograms_8k = []
for speaker in ['male', 'female']:
    wav = load.wav(STIMULUS_DIR / f'{speaker}_1.wav').sub(time=(0, 10))
    spectrograms_8k.append(gammatone_bank(wav, 80, 8000, 128, location='left', tstep=0.001))

p = plot.Array(spectrograms_8k, columns=1, axh=4, w=20)

# Overlay lines showing where the 8 frequency bands used by the "-8"
# predictors would fall.
band_edges = numpy.linspace(*p.axes[0].get_ylim(), 8, endpoint=False)[1:]
for y in band_edges:
    p.add_hline(y)
