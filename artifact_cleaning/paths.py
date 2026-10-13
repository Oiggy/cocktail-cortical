"""BIDS-derivatives file names for everything this project writes."""
from . import config


def subject_dir(subject, root=None):
    return (root or config.DERIV_ROOT) / f'sub-{subject}' / 'eeg'


def deriv_path(subject, desc, suffix, ext, root=None, mkdir=True):
    """e.g. deriv_path('01', 'preproc', 'eeg', 'fif') ->
    <DERIV_ROOT>/sub-01/eeg/sub-01_task-cocktail_desc-preproc_eeg.fif"""
    folder = subject_dir(subject, root)
    if mkdir:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / f'sub-{subject}_task-{config.TASK}_desc-{desc}_{suffix}.{ext}'


def group_path(name, root=None, mkdir=True):
    """Group-level (all-subject) outputs, e.g. summary tables."""
    folder = (root or config.DERIV_ROOT) / 'group'
    if mkdir:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / name
