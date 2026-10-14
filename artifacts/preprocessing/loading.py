"""Finding subjects and reading the raw BIDS recordings."""
import re

import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from .. import config


def subject_ids(bids_root=None):
    """Every subject in the BIDS dataset, as zero-padded strings ('01', ...)."""
    found = (p.name[4:] for p in (bids_root or config.BIDS_ROOT).glob('sub-*') if p.is_dir())
    return sorted(s for s in found if re.fullmatch(r'\d+', s))


def resolve_subjects(arg, bids_root=None):
    """'all' -> every subject; otherwise a list of ids, zero-padded."""
    if arg in ('all', ['all']):
        return subject_ids(bids_root)
    if isinstance(arg, str):
        arg = [arg]
    return [f'{int(s):02d}' for s in arg]


def read_raw(subject, bids_root=None):
    """The continuous recording, lazily (preload=False).

    At ~4096 Hz the full recording doesn't fit in memory comfortably (it
    crashed the kernel once - see analysis/experiment.py), so nothing is
    read here; preprocess.py reads one speech segment at a time.
    """
    path = BIDSPath(subject=subject, task=config.TASK, datatype='eeg', suffix='eeg', root=bids_root or config.BIDS_ROOT)
    return read_raw_bids(path, verbose=False)


def read_eelbrain_bads(subject, root=None):
    """Bad channels marked through analysis/experiment.py's
    mark_bad_channels(), from its BIDS channels.tsv. Empty if that step
    hasn't been run for this subject yet."""
    bads = set()
    for path in ((root or config.EELBRAIN_CHANNELS_ROOT) / f'sub-{subject}' / 'eeg').glob('*_channels.tsv'):
        table = pd.read_csv(path, sep='\t')
        if 'status' in table:
            bads.update(table.loc[table['status'] == 'bad', 'name'])
    return sorted(bads)
