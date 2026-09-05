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
# # Cortical TRF analysis
#
# This notebook checks how well cortical EEG responses can be predicted
# from the speech envelope (gammatone predictors), and compares the
# foreground story, the background story, and their mixture across the
# different listening conditions (clean / diotic / binaural / dichotic).
#
# Run this after data/bids_extraction.py, predictors/gammatone.py, and
# with analysis/experiment.py in the same folder (it is imported below).

# %%
from pathlib import Path

from eelbrain import *
from experiment import e

# Where to save plots.
DST = Path('~/Desktop').expanduser()

# %%
e.load_events()

# %%
EPOCHS = ['diotic', 'binaural', 'dichotic']

# Human-readable names and colors for the three predictors we compare
# throughout this notebook: the foreground story, the background story,
# and the mixture of both.
LABELS = {
    'gammatone_on_1': 'Foreground',
    'bg_gammatone_on_1': 'Background',
    'mix_gammatone_on_1': 'Mixture',
}
COLORS = {
    'gammatone_on_1': 'red',
    'bg_gammatone_on_1': 'blue',
    'mix_gammatone_on_1': '.3',
}

# %%
# Settings shared by every TRF estimated in this notebook: which
# preprocessed EEG to use, how many subjects, the response time window
# to fit (-100 to 600 ms relative to sound onset), and the fitting
# method. See https://eelbrain.readthedocs.io/en/stable/experiment.html
# for what each of these controls.
PARAMETERS = {
    'raw': 'ica',
    'group': 'all',
    'samplingrate': 128,
    'data': 'eeg',
    'tstart': -0.100,
    'tstop': 0.600,
    'error': 'l1',
    'basis': 0.050,
    'partitions': -4,
    'selective_stopping': 1,
}

# %% [markdown]
# ## Define a region of interest (ROI)
#
# Cortical speech responses are usually strongest over fronto-central
# electrodes, so later plots average over this set of sensors instead of
# showing every electrode separately.

# %%
data = e.load_trfs(1, 'gammatone-1', epoch='clean', **PARAMETERS, make=True)
eeg = data['det']

ROI = [
    'AF3', 'F1', 'F3', 'F5',
    'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5',
    'AF4', 'AFz',
    'Fz', 'F2', 'F4', 'F6',
    'FC6', 'FC4', 'FC2', 'FCz',
    'Cz', 'C2', 'C4', 'C6',
]

p = plot.SensorMap(eeg, mark=ROI)

# %%
# A cleaner version of the same plot, for use in a figure.
p = plot.SensorMap(eeg, w=1, h=1, labels=False, mark=ROI)
p.save(DST / 'ROI.pdf')

# %% [markdown]
# ## Envelope model
#
# First, a sanity check: does the simple envelope predictor
# ("gammatone-1", the foreground story's overall loudness over time)
# predict the EEG at all, for every subject?

# %%
data = e.load_trfs('all', 'gammatone-1', epoch='clean', **PARAMETERS, make=True)
TOPO_ARGS = dict(vmax=0.005, clip='circle')
p = plot.Topomap('det', data=data, **TOPO_ARGS)
p = plot.Topomap('det', 'subject', rows=1, data=data, **TOPO_ARGS)

# %%
# Same check, now also including the background story as a predictor,
# separately for each listening condition.
for epoch in ['diotic', 'dichotic', 'binaural']:
    data = e.load_trfs('all', 'gammatone-1 + bg~gammatone-1', epoch=epoch, **PARAMETERS, make=True)
    p = plot.Topomap('det', data=data, title=epoch, **TOPO_ARGS)
    p = plot.Topomap('det', 'subject', rows=1, data=data, **TOPO_ARGS)

# %% [markdown]
# ## Dichotic: effect of ear
#
# The auditory pathway crosses sides on its way to the brain, so a sound
# in one ear is represented more strongly on the opposite side of the
# brain. Since in the dichotic condition the background story always
# comes from one ear, we might expect the cortical representation of
# that background to depend on which ear it came from.

# %%
# Compare three predictors (foreground, background, mixture) against a
# combined model that includes all three, to see how much unique
# predictive power each one adds.
FULL = "gammatone-on-1 + bg~gammatone-on-1 + mix~gammatone-on-1"
COMPARISONS = {
    'fg': f"{FULL} @ gammatone-on-1",
    'bg': f"{FULL} @ bg~gammatone-on-1",
    'mix': f"{FULL} @ mix~gammatone-on-1",
}

# %%
e.show_model_test(COMPARISONS, **PARAMETERS, pmin=0.05, metric='det', epoch='dichotic-left', make=True, vmax=.0005)

# %%
e.show_model_test(COMPARISONS, **PARAMETERS, pmin=0.05, metric='det', epoch='dichotic-right', make=True, vmax=.0005)

# %% [markdown]
# Collect the model comparisons for both ears so we can test whether the
# difference is statistically meaningful.

# %%
dss = []
for stream, comparison in COMPARISONS.items():
    for epoch in ['dichotic-left', 'dichotic-right']:
        data, result = e.load_model_test(comparison, **PARAMETERS, pmin=0.05, metric='det', epoch=epoch, return_data=True)
        data = table.difference('det', 'model', 'test', 'baseline', 'subject', data=data)
        data['stream', :] = stream
        dss.append(data)
data = combine(dss)
data['roi_det'] = data['det'].mean(sensor=ROI)

# %%
test.ANOVA('roi_det', 'stream * epoch * subject', sub="epoch != 'dichotic'", data=data)

# %%
p = plot.Barplot('roi_det', 'epoch', sub="stream == 'fg'", match='subject', data=data, h=2, w=2, corr=False, title='FG')
p.set_xtick_rotation(45)

# %% [markdown]
# ## Effect of binaural cues
#
# Now compare all three listening conditions that include a background
# story (diotic, binaural, dichotic) to see whether the strength of the
# background's cortical representation depends on the spatial cues
# available to separate the two speakers.

# %%
FULL = "gammatone-on-1 + bg~gammatone-on-1 + mix~gammatone-on-1"
COMPARISONS = {
    'fg': f"{FULL} @ gammatone-on-1",
    'bg': f"{FULL} @ bg~gammatone-on-1",
    'mix': f"{FULL} @ mix~gammatone-on-1",
}

# %% [markdown]
# Inadvertent attention switches (accidentally attending the wrong
# story) are a possible confound, so check each condition individually
# before comparing them.

# %%
e.show_model_test(COMPARISONS, **PARAMETERS, pmin=0.05, metric='det', epoch='diotic', make=True, vmax=.0005)

# %%
e.show_model_test(COMPARISONS, **PARAMETERS, pmin=0.05, metric='det', epoch='binaural', make=True, vmax=.0005)

# %%
e.show_model_test(COMPARISONS, **PARAMETERS, pmin=0.05, metric='det', epoch='dichotic', make=True, vmax=.0005)

# %% [markdown]
# ### Is the difference between conditions reliable?

# %%
dss = []
for stream, comparison in COMPARISONS.items():
    for epoch in ['diotic', 'dichotic', 'binaural']:
        data, result = e.load_model_test(comparison, **PARAMETERS, pmin=0.05, metric='det', epoch=epoch, return_data=True)
        data = table.difference('det', 'model', 'test', 'baseline', 'subject', data=data)
        data['stream', :] = stream
        dss.append(data)
data = combine(dss)
data['roi_det'] = data['det'].mean(sensor=ROI)
test.ANOVA('roi_det', 'stream * epoch * subject', data=data)

# %%
for stream in COMPARISONS:
    display(test.ANOVA('roi_det', 'epoch * subject', sub=f"stream == '{stream}'", data=data, title=stream))
    p = plot.Barplot('roi_det', 'epoch', sub=f"stream == '{stream}'", match='subject', data=data, h=2, w=2, corr=False, title=stream)
    p.set_xtick_rotation(45)

# %%
# The p-value for the one comparison that came out significant above.
test.pairwise('roi_det', 'epoch', sub="stream == 'bg'", match='subject', data=data, corr=False, title=False)

# %%
# All pairwise comparisons, for reference.
test.pairwise('roi_det', 'stream % epoch', match='subject', data=data, title=False)

# %% [markdown]
# ## TRFs (temporal response functions)
#
# A TRF shows the shape of the brain's response over time to a unit of
# the predictor - essentially "if this sound feature occurred at time
# 0, how does the EEG respond over the following half second".

# %% [markdown]
# ### Single speaker (no competing story)

# %%
data = e.load_trfs(-1, 'gammatone-on-1', epoch='clean', **PARAMETERS, make=True)
trf = data['gammatone_on_1'].mean(sensor=ROI)
PLOT_ARGS = dict(frame='t', h=2.5, w=4, xlim=(-0.050, 0.550), clip=True, top=3.5, bottom=-0.5)
p = plot.UTSStat(trf * 1e3, **PLOT_ARGS)
p.save(DST / 'clean.pdf')

# %% [markdown]
# ### Two speakers (foreground, background, and mixture)

# %%
PLOT_ARGS['top'] = 2.5
all_dss = []
legend = True
for epoch in EPOCHS:
    data = e.load_trfs(-1, FULL, epoch=epoch, **PARAMETERS, make=True)

    dss = []
    for x in data.info['xs']:
        if '_on' not in x:
            # Only keep the onset-response predictors, matching FULL above.
            continue
        ds = data['subject',]
        ds[:, 'x'] = x
        ds[:, 'epoch'] = epoch
        ds['trf'] = data[x].mean(sensor=ROI)
        dss.append(ds)
        all_dss.append(ds)
    roi_data = combine(dss)
    p = plot.UTSStat('trf*1e3', 'x', data=roi_data, title=epoch.capitalize(), **PLOT_ARGS, labels=LABELS, colors=COLORS, legend=legend)
    p.save(DST / f'{epoch}.pdf')
    legend = False

roi_data = combine(all_dss)
# Smooth the TRFs to 512 samples for cleaner peak-time estimates below.
roi_data['trf_sm'] = resample(roi_data['trf'], 512)

# %%
for epoch in EPOCHS:
    p = plot.UTSStat('trf_sm*1e3', 'x', sub=f"epoch == '{epoch}'", data=roi_data, title=epoch.capitalize(), **PLOT_ARGS, labels=LABELS, colors=COLORS, legend=legend)

# %% [markdown]
# ## Peak response time
#
# Find, for each subject/condition/predictor, the time of the largest
# response between 20 and 130 ms - a typical window for the early
# cortical response to speech onsets.

# %%
roi_data['peak'] = roi_data['trf_sm'].sub(time=(0.02, 0.130)).argmax('time')

# %% [markdown]
# ### Peak time by condition

# %%
for epoch in EPOCHS:
    p = plot.Barplot('peak', 'x', match='subject', sub=f"epoch == '{epoch}'", data=roi_data, h=2, w=2.5, labels=LABELS, title=epoch, corr=False)
    display(test.pairwise('peak', 'x', match='subject', sub=f"epoch == '{epoch}'", data=roi_data, corr=False, title=False))
    p.set_xtick_rotation(30)

# %% [markdown]
# ### Peak time by predictor (foreground / background / mixture)

# %%
for x in COLORS:
    p = plot.Barplot('peak', 'epoch', match='subject', sub=f"x == '{x}'", data=roi_data, h=2, w=2.5, title=LABELS[x], corr=False)
    display(test.pairwise('peak', 'epoch', match='subject', sub=f"x == '{x}'", data=roi_data, corr=False, title=False))
    p.set_xtick_rotation(30)
