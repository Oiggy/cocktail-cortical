"""Paths, channel definitions and default parameters shared by every part of
the artifact-cleaning project.

Paths are worked out from this file's own location, the same way the rest
of the repo does it (see analysis/experiment.py), so nothing depends on
whose computer this runs on.
"""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
TASK = 'cocktail'


def set_bids_root(path):
    """Point every path below at a different BIDS dataset. Defaults to the
    sibling dataset folder the rest of the repo uses, or the
    COCKTAIL_BIDS_ROOT environment variable if set."""
    global BIDS_ROOT, DERIV_ROOT, EELBRAIN_CHANNELS_ROOT
    BIDS_ROOT = Path(path)
    # Everything this project writes goes under its own BIDS-derivatives
    # pipeline folder, next to (never inside) eelbrain's.
    DERIV_ROOT = BIDS_ROOT / 'derivatives' / 'artifact_cleaning'
    # eelbrain's mark_bad_channels() (analysis/experiment.py) records bad
    # channels in a BIDS channels.tsv under this folder - reused here so
    # bad channels are only ever marked once for the whole repo.
    EELBRAIN_CHANNELS_ROOT = BIDS_ROOT / 'derivatives' / 'mne'


set_bids_root(os.environ.get('COCKTAIL_BIDS_ROOT', REPO_ROOT.parent / 'dataset' / 'cocktail' / 'bids'))

MONTAGE_FILE = REPO_ROOT / 'analysis' / 'biosemi64mod.txt'

# The 64 scalp electrodes, in BioSemi A1-A32, B1-B32 order, already
# renamed by data/bids_extraction.py.
EEG_CHANNELS = (
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
    'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8',
    'PO4', 'O2',
)
MASTOIDS = ('A1', 'A2')

# Four electrodes around the eyes (data/bids_extraction.py's CH_MAP):
# above and below the left eye, and at the left and right outer canthi.
# Bipolar derivations cancel the (BioSemi CMS/DRL) reference entirely, so
# VEOG/HEOG are reference-free regardless of the EEG reference choice.
EOG_CHANNELS = ('EOG-LS', 'EOG-LI', 'EOG-L', 'EOG-R')
VEOG_PAIR = ('EOG-LS', 'EOG-LI')  # superior minus inferior: blinks, vertical saccades
HEOG_PAIR = ('EOG-L', 'EOG-R')    # left minus right canthus: horizontal saccades

# Channel groups used by labels and (later) features.
TEMPORAL_CHANNELS = ('FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8')  # jaw/temporalis EMG
FRONTAL_CHANNELS = ('Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8')
POSTERIOR_CHANNELS = ('PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz')

# Preprocessing defaults.
SFREQ = 256.  # Hz: keeps EMG power up to ~100 Hz, far below the ~4096 Hz native rate
HIGHPASS = 0.5  # Hz: removes drift without distorting blink shape (~200-400 ms)
REFERENCE = 'mastoids'  # 'mastoids' (matches analysis/experiment.py) or 'average'
LINE_FREQ = None  # None: detect 50 vs 60 Hz from the data; or set explicitly
SEGMENT_PAD = 10.  # s read either side of each speech segment, trimmed after filtering

# Condition names as used everywhere else in the repo, and the plan's
# label for each (the "clean" condition is the single-speaker one).
CONDITIONS = ('clean', 'diotic', 'dichotic', 'binaural')
