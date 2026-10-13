"""The three batch-1 scripts, run against a small synthetic BIDS dataset laid
out exactly like the real one (channel names, trigger codes, eelbrain's
bad-channel file), with 20 s stories instead of ~4 min ones."""
import json

import mne
import numpy as np
import pandas as pd
import pytest

from artifact_cleaning import config
from artifact_cleaning.labels.store import load_labels, review_dir
from artifact_cleaning.paths import deriv_path, group_path
from artifact_cleaning.preprocessing import segments as segments_module
from artifact_cleaning.scripts import make_labels, make_review_sheets, summarize_reviews

from .synthetic import UV, blink_shape, emg_channels, pink_noise

pytest.importorskip('pybv')  # mne-bids needs it to write BrainVision files
from mne_bids import BIDSPath, write_raw_bids  # noqa: E402

NATIVE = 512.
STORY = 20.  # s per story here


@pytest.fixture
def bids_root(tmp_path, monkeypatch):
    original = config.BIDS_ROOT
    root = tmp_path / 'bids'
    config.set_bids_root(root)
    monkeypatch.setattr(segments_module, 'SEGMENT_DURATION', {k: STORY for k in segments_module.SEGMENT_DURATION})
    _write_subject(root, '01')
    yield root
    config.set_bids_root(original)


def _write_subject(root, subject, seed=0):
    rng = np.random.default_rng(seed)
    onsets = 5. + (STORY + 5.) * np.arange(12)
    duration = onsets[-1] + STORY + 5.
    n = int(duration * NATIVE)
    t = np.arange(n) / NATIVE
    names = list(config.EEG_CHANNELS) + list(config.MASTOIDS) + list(config.EOG_CHANNELS)
    common = 300 * UV * np.sin(2 * np.pi * 2 * t)
    data = np.stack([common + pink_noise(n, rng, 5 * UV) for _ in names])
    shape = blink_shape(NATIVE)
    for blink in np.arange(3., duration - 3., 4.):  # a blink every 4 s
        i = int(blink * NATIVE)
        data[names.index('EOG-LS'), i:i + len(shape)] += 150 * UV * shape
    bursts, _ = emg_channels(sfreq=NATIVE, duration=duration, n_channels=2, n_bursts=10, seed=seed)
    data[names.index('T7')] += bursts[0] - bursts[0].mean()
    data[names.index('T8')] += bursts[1] - bursts[1].mean()

    info = mne.create_info(names, NATIVE, ['eeg'] * 66 + ['eog'] * 4)
    info['line_freq'] = 60.
    raw = mne.io.RawArray(data, info, verbose=False)
    events = np.array([[int(o * NATIVE), 0, 1] for o in onsets] + [[int(2 * NATIVE), 0, 8]])
    events = events[np.argsort(events[:, 0])]
    path = BIDSPath(subject=subject, task=config.TASK, datatype='eeg', suffix='eeg', root=root)
    write_raw_bids(raw, path, events=events, event_id={'code_1': 1, 'code_2': 2, 'code_8': 8},
                   format='BrainVision', allow_preload=True, overwrite=True, verbose=False)

    # eelbrain's mark_bad_channels() output: a BIDS channels.tsv with a status column
    bads_dir = root / 'derivatives' / 'mne' / f'sub-{subject}' / 'eeg'
    bads_dir.mkdir(parents=True)
    pd.DataFrame({'name': names, 'status': ['bad' if ch == 'Cz' else 'good' for ch in names]}).to_csv(
        bads_dir / f'sub-{subject}_task-{config.TASK}_channels.tsv', sep='\t', index=False)


def test_scripts_end_to_end(bids_root):
    make_labels.main(['--subjects', 'all'])

    info = json.loads(deriv_path('01', 'preproc', 'eeg', 'json').read_text())
    assert info['interpolated'] == ['Cz'] and info['n_segments'] == 12
    # no mains noise in this data, so no clear peak: falls back to the BIDS metadata
    assert info['line_freq'] == 60. and info['line_freq_source'] == 'bids'
    assert info['reference'] == 'mastoids:A1+A2'
    raw = mne.io.read_raw_fif(deriv_path('01', 'preproc', 'eeg', 'fif'), verbose=False)
    assert raw.n_times == 12 * int(STORY * config.SFREQ)

    ocular = load_labels('01', 'ocular')
    blinks = ocular[ocular['trial_type'] == 'blink']
    # one blink every 4 s; 5 per 20 s story, give or take one at the edges
    assert 12 * 4 <= len(blinks) <= 12 * 6
    assert set(blinks['condition']) == {'clean', 'binaural', 'dichotic', 'diotic'}
    assert len(load_labels('01', 'emg')) > 0
    summary = pd.read_csv(group_path('label_summary.tsv'), sep='\t')
    assert set(summary['condition']) == {'clean', 'binaural', 'dichotic', 'diotic'}

    make_review_sheets.main(['--subjects', '01', '--n-per-type', '3', '--n-unlabelled', '2'])
    sheet_path = review_dir('ocular', '01') / 'review.csv'
    sheet = pd.read_csv(sheet_path, keep_default_na=False)
    sheet['verdict'] = ['wrong'] + ['ok'] * (len(sheet) - 1)
    sheet.loc[sheet['trial_type'] == 'none', 'verdict'] = 'clean'
    sheet.to_csv(sheet_path, index=False)

    # a sheet with verdicts is never overwritten
    make_review_sheets.main(['--subjects', '01', '--kinds', 'ocular'])
    assert pd.read_csv(sheet_path, keep_default_na=False)['verdict'].tolist() == sheet['verdict'].tolist()

    summarize_reviews.main(['--kinds', 'ocular'])
    assert group_path('review_summary_ocular.tsv').exists()
    assert len(load_labels('01', 'ocular')) == len(ocular) - 1  # the 'wrong' one is dropped
