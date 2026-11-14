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
    ATTEND, IGNORE, MIX, envelope, onset, inmix_onset, overt_inmix_onset, masked_inmix_onset, level_onset,
    compute_unique_contribution, compute_direct_comparison, show_topomap_trio, show_subject_swarm,
    load_trfs_saved, extract_strf, extract_trf, extract_attention_effect, add_anchored_peak,
    print_peak_table, plot_trf_bins, submit_cluster_run, cluster_queue, cluster_log, check_results, ResultsNotReady, ClusterConnectionError, find_roi, show_roi_file, show_roi, check_alignment,
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
# Overt and masked onsets: each speaker's onsets split by the mixture,
# with the speaker at the level it has in the mixture (its own .wav file
# is 4x louder, which made part of every onset look masked; see
# predictors/inmix_onsets.py). overt + masked == inmix_onset, the plain
# onset at that level, so FULL_INMIX is the baseline for the split, as in
# Brodbeck et al. 2020. (cortical_analysis.ipynb keeps the older
# file-level overt_onset/masked_onset from experiment.py.)
overt_onset, masked_onset = overt_inmix_onset, masked_inmix_onset
FULL_INMIX = (
    f"{envelope(ATTEND)} + {inmix_onset(ATTEND)} + {envelope(IGNORE)} + {inmix_onset(IGNORE)} + "
    f"{envelope(MIX)} + {onset(MIX)}"
)
FULL_MASKED = (
    f"{envelope(ATTEND)} + {overt_onset(ATTEND)} + {masked_onset(ATTEND)} + "
    f"{envelope(IGNORE)} + {overt_onset(IGNORE)} + {masked_onset(IGNORE)} + "
    f"{envelope(MIX)} + {onset(MIX)}"
)
LEVEL_AWARE = (
    f"{envelope(ATTEND)} + {onset(ATTEND)} + "
    f"{envelope(IGNORE)} + "
    f"{envelope(MIX)} + {level_onset('low')} + {level_onset('mid')} + {level_onset('high')}"
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


# Peak search windows. Each subject's peak is taken within +-25 ms of
# where the group-average waveform peaks inside the window
# (add_anchored_peak).
ONSET_POSITIVE_PEAK_WINDOW = (0.040, 0.200)
ONSET_NEGATIVE_PEAK_WINDOW = (0.040, 0.200)
ENVELOPE_POSITIVE_PEAK_WINDOW = (0.040, 0.200)
ENVELOPE_NEGATIVE_PEAK_WINDOW = (0.040, 0.200)
MASKED_POSITIVE_PEAK_WINDOW = (0.040, 0.200)
MASKED_NEGATIVE_PEAK_WINDOW = (0.040, 0.200)


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
