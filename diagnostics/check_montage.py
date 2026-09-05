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
# # Check the electrode montage
#
# This is a one-off diagnostic script, not part of the main pipeline. It
# checks that:
#  1. the BioSemi channel names in a raw recording match what we expect
#  2. the 64-channel names can be mapped from the full 128-channel
#     BioSemi montage
#  3. the custom montage file used by analysis/experiment.py
#     (biosemi64mod.txt) loads and plots correctly
#
# Update RAW_FILE below to point at any one subject's raw recording.

# %%
import mne


RAW_FILE = '~/Data/BinauralCocktail/eeg/sub-03/sub-03_task-cocktail_eeg.vhdr'
MONTAGE_FILE = '../analysis/biosemi64mod.txt'

raw = mne.io.read_raw_brainvision(RAW_FILE)

# %%
print(' '.join(raw.info.ch_names))

# %% [markdown]
# ## Electrode layout notes
#
# The four extra "Ex" channels on the BioSemi cap are placed as follows:
#  - Ex1: above the left eye
#  - Ex2: below the left eye
#  - Ex3: next to the left eye
#  - Ex4: next to the right eye
#  - Ex5: left mastoid
#  - Ex6: right mastoid

# %%
# BioSemi sells caps with either 128 or 64 electrodes. This experiment
# used the 64-electrode layout, so map the full 128-channel names down
# to their 64-channel equivalents before applying a montage.
montage_128 = mne.channels.make_standard_montage('biosemi128')
montage_64 = mne.channels.make_standard_montage('biosemi64')
CH_MAP = dict(zip(montage_128.ch_names, montage_64.ch_names))

_ = raw.rename_channels(CH_MAP)
_ = raw.set_montage('biosemi64', on_missing='ignore')

# %%
montage_64.plot()

# %% [markdown]
# ## The custom montage file
#
# analysis/experiment.py loads electrode positions from this file rather
# than the generic "biosemi64" montage, since it has been adjusted
# slightly for this cap. Loading and plotting it here is a quick way to
# confirm the file is valid before running the full pipeline.

# %%
custom_montage = mne.channels.read_custom_montage(MONTAGE_FILE)
custom_montage.plot(kind='3d')
