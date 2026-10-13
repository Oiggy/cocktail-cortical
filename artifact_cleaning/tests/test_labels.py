import mne
import numpy as np
import pandas as pd
import pytest

from artifact_cleaning.labels import emg, ocular
from artifact_cleaning.labels.review import apply_review, make_review_sheet, summarize_reviews
from artifact_cleaning.labels.store import coverage_mask, label_summary

from .synthetic import emg_channels, match, ocular_segment, segments_df

SFREQ = 256.


@pytest.mark.parametrize('sign', [1., -1.])
def test_blinks(sign):
    veog, _, blink_times, saccade_times = ocular_segment(veog_sign=sign, seed=3)
    events, polarity = ocular.detect_blinks(veog, SFREQ)
    assert polarity == sign
    recall, precision = match(blink_times + 0.15, events)  # +0.15: blink peak
    assert recall >= 0.95 and precision >= 0.95
    # windows cover the whole ~300 ms blink, not just its peak
    assert events['duration'].median() > 0.25


def test_saccades():
    _, heog, _, saccade_times = ocular_segment(seed=4)
    events = ocular.detect_saccades(heog, SFREQ, 'HEOG')
    recall, precision = match(saccade_times, events)
    assert recall >= 0.95 and precision >= 0.9
    assert set(np.sign(events['amplitude'])) <= {-1., 1.}


def test_label_segment_keeps_blinks_out_of_vertical_saccades():
    veog, heog, blink_times, _ = ocular_segment(seed=5)
    events, _ = ocular.label_segment(veog, heog, SFREQ)
    vertical = events[(events['trial_type'] == 'saccade') & (events['channel'] == 'VEOG')]
    assert match(blink_times + 0.15, vertical)[0] == 0  # no blink relabelled as a saccade
    assert set(events['trial_type']) == {'blink', 'saccade'}


def _fake_raw(channel_data, names, sfreq=SFREQ):
    return mne.io.RawArray(np.asarray(channel_data), mne.create_info(list(names), sfreq, 'eeg'), verbose=False)


def test_label_recording_offsets_by_segment():
    """Event times are shifted into recording time, segment by segment."""
    parts = [ocular_segment(duration=60., seed=s) for s in (6, 7)]
    veog = np.concatenate([p[0] for p in parts])
    heog = np.concatenate([p[1] for p in parts])
    segs = segments_df([60., 60.], SFREQ, ['clean', 'diotic'])
    events, meta = ocular.label_recording(_fake_raw([veog, heog], ['VEOG', 'HEOG']), segs)
    blinks = events[events['trial_type'] == 'blink']
    truth = np.concatenate([parts[0][2], 60. + parts[1][2]]) + 0.15
    assert match(truth, blinks)[0] >= 0.95
    second = blinks[blinks['onset'] >= 60]
    assert (second['segment'] == 2).all() and (second['condition'] == 'diotic').all()
    assert np.allclose(second['orig_onset'] - second['onset'], 200. - 60.)
    assert set(meta['veog_polarity']) == {1, 2}


def test_emg_weak_labels():
    data, onsets = emg_channels(seed=8)
    names = ['FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8']
    segs = segments_df([120.], SFREQ)
    events = emg.label_recording(_fake_raw(data, names), segs, emg.EMGParams(line_freq=None))
    recall, precision = match(onsets + 0.5, events)  # +0.5: burst centre
    assert recall >= 0.9 and precision >= 0.8
    assert (events['source'] == 'weak').all() and events['channel'].isin(names).all()


def test_summary_and_mask():
    segs = segments_df([60., 60.], SFREQ, ['clean', 'diotic'])
    events = pd.DataFrame({'onset': [1., 70.], 'duration': [0.5, 1.], 'trial_type': ['blink', 'emg'],
                           'condition': ['clean', 'diotic']})
    assert coverage_mask(events, segs['stop'].max(), SFREQ).sum() == int(1.5 * SFREQ)
    summary = label_summary(events, segs, SFREQ).set_index('condition')
    assert summary.loc['clean', 'blink_per_min'] == 1.0
    assert summary.loc['diotic', 'emg_pct_time'] == pytest.approx(100 / 60, abs=0.01)


def test_review_round_trip(tmp_path):
    veog, heog, *_ = ocular_segment(duration=60., seed=9)
    segs = segments_df([60.], SFREQ)
    fp = np.zeros((3, len(veog)))
    raw = _fake_raw(np.vstack([veog, heog, fp]), ['VEOG', 'HEOG', 'Fp1', 'Fp2', 'Fpz'])
    events, _ = ocular.label_recording(raw, segs)
    sheet = make_review_sheet(raw, events, segs, '01', tmp_path, n_per_type=3, n_unlabelled=4)
    assert (sheet['trial_type'] == 'none').sum() == 4
    assert all((tmp_path / png).exists() for png in sheet['png'])

    labelled = sheet[sheet['trial_type'] != 'none'].index
    sheet.loc[labelled[0], 'verdict'] = 'wrong'
    sheet.loc[labelled[1:], 'verdict'] = 'ok'
    sheet.loc[labelled[1], 'corrected_duration'] = '0.123'
    sheet.loc[sheet['trial_type'] == 'none', 'verdict'] = ['clean', 'clean', 'clean', 'missed:blink']
    # verdicts are filled in a spreadsheet and come back through CSV
    sheet.to_csv(tmp_path / 'filled.csv', index=False)
    sheet = pd.read_csv(tmp_path / 'filled.csv', keep_default_na=False)

    summary = summarize_reviews([sheet]).set_index('trial_type')
    assert summary.loc['none', 'miss_rate'] == 0.25
    assert summary['wrong'].drop('none').sum() == 1

    applied = apply_review(events, sheet, '01')
    assert len(applied) == len(events) - 1
    assert sheet.loc[labelled[0], 'id'] not in set(applied['id'])
    corrected = applied[applied['id'] == sheet.loc[labelled[1], 'id']]
    assert corrected['duration'].iloc[0] == pytest.approx(0.123)
