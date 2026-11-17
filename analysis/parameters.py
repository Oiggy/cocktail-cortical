"""Shared settings for the per-condition analysis notebooks.

Everything that came before the "Clean speech" section of
cortical_analysis.ipynb lives here, so each notebook (clean.ipynb,
diotic.ipynb, diotic_overt_masked_onsets.ipynb, ...) can run on its own
after a single `from parameters import *`.

The cluster computes every TRF fit and model test and keeps it there
(cluster/run_analysis.sbatch, with COCKTAIL_RESULTS=compute). The notebooks
only read them, never computing or saving anything:
- in the cluster's JupyterHub (from any browser; the laptop can stay
  closed), straight from the cluster's disk;
- on the Mac, over SSH into memory, after `bash cluster/connect.sh`.
Either way, ask the cluster for missing results from a notebook with
submit_cluster_run() and follow it with cluster_queue().
"""
import os
from pathlib import Path

# The cluster: where results are computed and kept.
CLUSTER_HOST = 'iggy@narval.alliancecan.ca'
CLUSTER_ACCOUNT = 'def-ibruce'
CLUSTER_REPO = 'projects/def-ibruce/iggy/cocktail-cortical'  # relative to the home folder there
CLUSTER_RESULTS = 'projects/def-ibruce/iggy/dataset/cocktail/bids/derivatives/results'

# Running from the cluster's own copy of the repo (its JupyterHub)?
ON_CLUSTER = Path(__file__).resolve().parent.parent == (Path.home() / CLUSTER_REPO).resolve()

# 'cluster': read results from the cluster's disk (JupyterHub there).
# 'remote': read them from the cluster over SSH (the Mac). 'compute': fit
# and save whatever is missing (the cluster job sets this). Set before
# experiment.py is imported: when only reading, it builds no pipeline.
RESULTS_MODE = os.environ.setdefault('COCKTAIL_RESULTS', 'cluster' if ON_CLUSTER else 'remote')

from eelbrain import *
from experiment import (
    e, topomap_with_colorbar, FIGURES_DIR, set_partitions, configure_analysis,
    ATTEND, IGNORE, MIX, envelope, onset, level_onset,
    split_envelope, split_onset, split_overt_onset, split_masked_onset, split_level_onset,
    compute_unique_contribution, compute_direct_comparison, show_topomap_trio, show_subject_swarm,
    load_trfs_saved, extract_strf, extract_trf, extract_attention_effect, add_anchored_peak,
    print_peak_table, plot_trf_bins, plot_attention_effect, submit_cluster_run, cluster_queue, cluster_log, check_results, ResultsNotReady, ClusterConnectionError, find_roi, show_roi_file, show_roi, check_alignment, cluster_cleanup, tfce_summary,
    dtu_download, dtu_status, dtu_inspect, dtu_run, dtu_rooms, dtu_progress, dtu_setup, dtu_model_test, dtu_trfs,
)
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

# Plots are shown in the notebook only; no image files are written.
SAVE_FIGURES = False
DST = FIGURES_DIR


# Preprocessing (once per subject, on the cluster before fitting: see
# cluster/run_analysis.sbatch). Not run on import. Redoing a subject
# changes its bad channels or ICA, and every TRF for that subject then
# has to be fitted again.

MANUAL_BAD_CHANNELS = False
# A channel is marked bad only if it correlates with its neighbors below
# this. Lower = fewer channels marked bad (was 0.3).
AUTO_BAD_CHANNELS_R = 0.2
REDO_BAD_CHANNELS = True

MANUAL_ICA = False
# A component is removed only if ICLabel is at least this sure it is an
# artifact. Higher = fewer components removed (was 0.75).
AUTO_ICA_CONFIDENCE = 0.9
# Only these artifact types are removed: eye and heart, as in Brodbeck et al. 2020.
AUTO_ICA_REJECT_LABELS = ('eye blink', 'heart beat')


def preprocess():
    "Bad channels and ICA for every subject (skips subjects already done unless told to redo them)"
    e.mark_bad_channels(
        'all',
        manual_bad_channels=MANUAL_BAD_CHANNELS, auto_bad_channels_r=AUTO_BAD_CHANNELS_R,
        redo_bad_channels=REDO_BAD_CHANNELS,
    )
    e.select_ica_artifacts(
        'all',
        manual_ica=MANUAL_ICA, auto_ica_confidence=AUTO_ICA_CONFIDENCE,
        auto_ica_reject_labels=AUTO_ICA_REJECT_LABELS,
    )


# Models. Only `epoch=` changes between diotic/dichotic-left/
# dichotic-right/binaural; the model strings do not depend on condition.
# '' = attended speaker, 'bg~' = ignored speaker, 'mix~' = the mixture.
FULL_CLEAN = f"{envelope(ATTEND)} + {onset(ATTEND)}"
FULL_BASE = f"{envelope(ATTEND)} + {onset(ATTEND)} + {envelope(IGNORE)} + {onset(IGNORE)} + {envelope(MIX)} + {onset(MIX)}"
# Stream-level tests: one stream (its envelope + onsets) removed from FULL_BASE
# at once. Does the brain track that talker (or the mixture) at all?
FULL_BASE_NO_ATTEND_STREAM = f"{envelope(IGNORE)} + {onset(IGNORE)} + {envelope(MIX)} + {onset(MIX)}"
FULL_BASE_NO_IGNORE_STREAM = f"{envelope(ATTEND)} + {onset(ATTEND)} + {envelope(MIX)} + {onset(MIX)}"
FULL_BASE_NO_MIX_STREAM = f"{envelope(ATTEND)} + {onset(ATTEND)} + {envelope(IGNORE)} + {onset(IGNORE)}"
# The overt/masked notebooks (predictors/brodbeck2020_predictors.py): the
# overt/masked split is made element-wise at full frequency resolution and
# only then binned into 8 bands, with each speaker at the level it has in the
# mixture that was played, as Brodbeck et al. 2020 defined it. Which
# spectrogram it starts from:
#   'b20': the paper's (256 bands 20-5,000 Hz, scaled with exponent 0.6)
#   'log': gammatone_predictors.py's (128 bands 80-8,000 Hz, log(x + 1)), the one
#          under which the speakers' onsets predict this EEG (diotic.ipynb)
# Results of the two are kept apart on the cluster; switch and rerun to
# compare. (cluster_cleanup() keeps only the results of the recipe chosen here.)
SPLIT_RECIPE = 'log'
SPLIT_COLUMN = f"gammatone_{SPLIT_RECIPE}"  # start of the overt/masked TRF column names


# The overt/masked notebooks' predictors, in the chosen recipe
def split_env(role=ATTEND):
    return split_envelope(role, recipe=SPLIT_RECIPE)


def split_on(role=ATTEND):
    return split_onset(role, recipe=SPLIT_RECIPE)


def split_tier(tier, role=MIX):
    return split_level_onset(tier, role, recipe=SPLIT_RECIPE)


def overt_onset(role=ATTEND):
    return split_overt_onset(role, recipe=SPLIT_RECIPE)


def masked_onset(role=ATTEND):
    return split_masked_onset(role, recipe=SPLIT_RECIPE)


FULL_INMIX = (
    f"{split_env(ATTEND)} + {split_on(ATTEND)} + {split_env(IGNORE)} + {split_on(IGNORE)} + "
    f"{split_env(MIX)} + {split_on(MIX)}"
)
FULL_MASKED = (
    f"{split_env(ATTEND)} + {overt_onset(ATTEND)} + {masked_onset(ATTEND)} + "
    f"{split_env(IGNORE)} + {overt_onset(IGNORE)} + {masked_onset(IGNORE)} + "
    f"{split_env(MIX)} + {split_on(MIX)}"
)
# Overt and masked onsets of one speaker removed together (their joint
# contribution), and, for reference, the same onsets unsplit removed.
# Overt and masked come from the same onsets and rise and fall together, so
# each can stand in for the other in a one-at-a-time test; removed together,
# they should carry what the unsplit onsets carry.
FULL_MASKED_NO_ATTEND_ONSETS = (
    f"{split_env(ATTEND)} + "
    f"{split_env(IGNORE)} + {overt_onset(IGNORE)} + {masked_onset(IGNORE)} + "
    f"{split_env(MIX)} + {split_on(MIX)}"
)
FULL_MASKED_NO_IGNORE_ONSETS = (
    f"{split_env(ATTEND)} + {overt_onset(ATTEND)} + {masked_onset(ATTEND)} + "
    f"{split_env(IGNORE)} + "
    f"{split_env(MIX)} + {split_on(MIX)}"
)
FULL_INMIX_NO_ATTEND_ONSET = (
    f"{split_env(ATTEND)} + {split_env(IGNORE)} + {split_on(IGNORE)} + {split_env(MIX)} + {split_on(MIX)}")
FULL_INMIX_NO_IGNORE_ONSET = (
    f"{split_env(ATTEND)} + {split_on(ATTEND)} + {split_env(IGNORE)} + {split_env(MIX)} + {split_on(MIX)}")
# Level-aware control (Brodbeck et al. 2020): no ignored-speaker onsets, the
# mixture onsets split into three intensity tiers instead.
LEVEL_AWARE = (
    f"{split_env(ATTEND)} + {split_on(ATTEND)} + "
    f"{split_env(IGNORE)} + "
    f"{split_env(MIX)} + {split_tier('low')} + {split_tier('mid')} + {split_tier('high')}"
)


# Fit metric: 'r' (Pearson correlation) or 'ev' (proportion of variance explained)
METRIC = 'r'

# Model comparisons: permutation tests with threshold-free cluster
# enhancement (TFCE), as in Brodbeck et al. 2020, one-tailed, significant
# at p <= ALPHA. With 13 subjects there are only 2**13 sign flips, so every
# one of them is used (8191), an exact test; SAMPLES caps it otherwise.
PMIN = 'tfce'
ALPHA = 0.05
SAMPLES = 10000

# Where each notebook's model comparisons are tested: 'whole_scalp' (all
# sensors) or 'roi' (the sensors the clean notebook found, read from
# roi.txt on the cluster; see find_roi() in experiment.py). As in Brodbeck
# et al. 2020: clean speech over the whole scalp, then the ROI it defines.
TEST_REGION = {
    'clean': 'whole_scalp',
    'diotic': 'roi',
    'diotic_overt_masked_onsets': 'roi',
    'dichotic': 'roi',
    'dichotic_overt_masked_onsets': 'roi',
    'dichotic_left': 'roi',
    'dichotic_left_overt_masked_onsets': 'roi',
    'dichotic_right': 'roi',
    'dichotic_right_overt_masked_onsets': 'roi',
    'binaural': 'roi',
    'binaural_overt_masked_onsets': 'roi',
}

# ev is a tiny fraction of variance and needs a fixed, narrow colour
# scale; r reads fine with an automatic one.
TOPO_ARGS = dict(clip='circle') if METRIC == 'r' else dict(vmax=0.005, clip='circle')


# TRF fit settings, passed to every fit and model test. tstart/tstop set
# the fit window: with tstop 0.400 the TRFs end at about 390 ms.
PARAMETERS = {
    'raw': 'ica',
    'group': 'all',
    'samplingrate': 64,
    'data': 'eeg',
    'tstart': -0.100,
    'tstop': 0.400,
}

# Cross-validation folds for every TRF fit. Negative = the data are
# concatenated, then split into this many folds.
PARTITIONS = -5
e = set_partitions(PARTITIONS, build=RESULTS_MODE == 'compute')

# The ROI averaged for every ROI summary (STRFs, TRF waveforms, peaks, the
# ROI t-tests) and used for the 'roi' tests is found from clean speech:
# find_roi() in the clean notebook, read from roi.txt by every other
# notebook's check_results() cell. ROI_FALLBACK, the earlier hand-picked
# fronto-central sensors, is used only if clean speech shows no significant
# sensor at all (roi.txt says so when it happens).
ROI = None
ROI_FALLBACK = [
    'AF3', 'F1', 'F3', 'F5',
    'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5',
    'AF4', 'AFz',
    'Fz', 'F2', 'F4', 'F6',
    'FC6', 'FC4', 'FC2', 'FCz',
    'Cz', 'C2', 'C4', 'C6',
]


# Display names and colours
LABELS_ROLE = {'attend': 'Attend', 'ignore': 'Ignore', 'mix': 'Mixture'}
COLORS_ROLE = {'attend': 'red', 'ignore': 'blue', 'mix': '.3'}
LABELS_MASKING = {'overt': 'Overt', 'masked': 'Masked'}
COLORS_MASKING = {'overt': 'darkorange', 'masked': 'purple'}


# Peak search windows, as Brodbeck et al. 2020: 20-200 ms for single-speaker
# and stream-based (mixture/attended/ignored) responses, 20-250 ms for the
# overt/masked onset analysis. Each subject's peak is taken within +-25 ms of
# where the group-average waveform peaks inside the window
# (add_anchored_peak).
ONSET_POSITIVE_PEAK_WINDOW = (0.020, 0.200)
ONSET_NEGATIVE_PEAK_WINDOW = (0.020, 0.200)
ENVELOPE_POSITIVE_PEAK_WINDOW = (0.020, 0.200)
ENVELOPE_NEGATIVE_PEAK_WINDOW = (0.020, 0.200)
MASKED_POSITIVE_PEAK_WINDOW = (0.020, 0.250)
MASKED_NEGATIVE_PEAK_WINDOW = (0.020, 0.250)


# Display settings (the fits are untouched)
TAPER = (0.003, 0.033)         # low until 3 ms, smooth climb to full strength at 33 ms
TAPER_FLOOR = 0.1              # keep 10% of the TRF before that
END_TAPER = 0.050              # fade out over the last 50 ms of the fit window
PLOT_WINDOW = (-0.050, 0.390)  # x range of every STRF and TRF plot (TRFs end at ~390 ms with tstop 0.400)
STRF_INTERPOLATION = 'bicubic' # display-only smoothing of STRF colours (None = raw sample blocks)
BIN_ERROR = 'sem'              # band around each frequency bin's TRF: 'sem' or 'sd' across subjects

configure_analysis(
    parameters=PARAMETERS, roi=ROI, metric=METRIC, pmin=PMIN, alpha=ALPHA, samples=SAMPLES,
    test_regions=TEST_REGION, roi_fallback=ROI_FALLBACK, topo_args=TOPO_ARGS,
    taper=TAPER, taper_floor=TAPER_FLOOR, end_taper=END_TAPER, plot_window=PLOT_WINDOW,
    results_mode=RESULTS_MODE, bin_error=BIN_ERROR, save_figures=SAVE_FIGURES,
    cluster_host=CLUSTER_HOST, cluster_repo=CLUSTER_REPO, cluster_results=CLUSTER_RESULTS,
    cluster_account=CLUSTER_ACCOUNT,
)
