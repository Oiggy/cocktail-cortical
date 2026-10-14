"""Automatic blink and saccade labels from the bipolar VEOG/HEOG channels.

Blinks: large, brief VEOG deflections. VEOG is band-passed 1-10 Hz (the
same band MNE's find_eog_events uses), its sign set so blinks point up
(from the skew of the distribution - blinks make a heavy one-sided tail),
and peaks above k x a robust (median absolute deviation) noise estimate
become blinks. Each blink window runs where the deflection stays above
10% of its peak, plus a margin.

Saccades: fast EOG steps. The EOG is low-passed at 30 Hz (to keep muscle
noise out of the derivative) and differentiated; runs where |velocity|
exceeds lambda x a median-based velocity SD (Engbert & Kliegl 2003) are
candidates, kept only if the EOG level actually shifts by at least
min_step across them. HEOG gives horizontal saccades; VEOG gives vertical
saccades outside blink windows (blinks are fast VEOG deflections too).

All thresholds are data-driven per segment except the minimum saccade
step, so they adapt to each subject's electrode placement and noise.
Reported amplitudes: blinks, the peak of the 1-10 Hz band-passed VEOG
(somewhat below the raw peak); saccades, the signed EOG level shift.
These are labels for training and evaluating detectors - spot-check a
subset by hand (labels/review.py) before trusting them.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt
from scipy.stats import skew

EVENT_COLUMNS = ['onset', 'duration', 'trial_type', 'channel', 'amplitude', 'peak_velocity', 'source']


@dataclass
class OcularParams:
    blink_band: tuple = (1., 10.)       # Hz
    blink_k: float = 5.                 # peak threshold, in robust SDs
    blink_min_amplitude: float = 40e-6  # V: floor under the data-driven threshold
    blink_min_interval: float = 0.2     # s between blink peaks
    blink_width: tuple = (0.04, 0.8)    # s, width at half height: rejects spikes and slow drifts
    blink_extent: float = 0.9           # window ends where deflection falls to (1 - this) of peak
    blink_margin: float = 0.05          # s added either side
    saccade_lowpass: float = 30.        # Hz
    saccade_lambda: float = 6.          # velocity threshold, in median-based SDs
    saccade_min_step: float = 20e-6     # V: minimum EOG level shift across the saccade
    saccade_min_duration: float = 0.008 # s above velocity threshold
    saccade_step_window: float = 0.05   # s either side used to measure the step
    saccade_merge: float = 0.03         # s: runs closer than this are one saccade
    saccade_margin: tuple = (0.05, 0.15)  # s before/after: covers the spike potential and the step's settling


def robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _bandpass(x, sfreq, low, high, order=4):
    sos = butter(order, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, x)


def _lowpass(x, sfreq, high, order=4):
    sos = butter(order, high, btype='lowpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, x)


def _runs(mask):
    """(start, stop) index pairs of each run of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return edges.reshape(-1, 2)


def _merge(windows, gap=0.):
    """Merge (start, stop) windows that overlap or are within `gap`."""
    merged = []
    for start, stop in sorted(windows):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [tuple(w) for w in merged]


def detect_blinks(veog, sfreq, params=OcularParams()):
    """Blink windows in one segment's VEOG (V). Returns (events, polarity)."""
    x = _bandpass(veog, sfreq, *params.blink_band)
    polarity = 1. if skew(x) >= 0 else -1.
    x = x * polarity
    threshold = max(params.blink_k * robust_sd(x), params.blink_min_amplitude)
    peaks, props = find_peaks(x, height=threshold, distance=max(1, int(params.blink_min_interval * sfreq)))
    rows = []
    if len(peaks):
        half = peak_widths(x, peaks, rel_height=0.5)[0] / sfreq
        keep = (half >= params.blink_width[0]) & (half <= params.blink_width[1])
        peaks, heights = peaks[keep], props['peak_heights'][keep]
        if len(peaks):
            _, _, left, right = peak_widths(x, peaks, rel_height=params.blink_extent)
            n = len(x)
            for peak, height, lo, hi in zip(peaks, heights, left, right):
                start = max(0., lo / sfreq - params.blink_margin)
                stop = min(n / sfreq, hi / sfreq + params.blink_margin)
                rows.append((start, stop, height, peak / sfreq))
    events = []
    for start, stop in _merge([(r[0], r[1]) for r in rows]):
        inside = [r for r in rows if start <= r[3] <= stop]
        events.append({
            'onset': start, 'duration': stop - start, 'trial_type': 'blink', 'channel': 'VEOG',
            'amplitude': max(r[2] for r in inside), 'peak_velocity': np.nan, 'source': 'auto',
        })
    return pd.DataFrame(events, columns=EVENT_COLUMNS), polarity


def detect_saccades(eog, sfreq, channel, params=OcularParams(), exclude=()):
    """Saccade windows in one segment's HEOG or VEOG (V). `exclude`:
    (start, stop) windows in s where saccades are not looked for
    (blinks, for VEOG)."""
    x = _lowpass(eog, sfreq, params.saccade_lowpass)
    velocity = np.gradient(x) * sfreq
    sd = np.sqrt(max(np.median(velocity ** 2) - np.median(velocity) ** 2, 1e-30))
    above = np.abs(velocity) > params.saccade_lambda * sd
    for start, stop in exclude:
        above[int(start * sfreq):int(np.ceil(stop * sfreq))] = False

    min_len = max(1, int(round(params.saccade_min_duration * sfreq)))
    runs = [(a, b) for a, b in _runs(above) if b - a >= min_len]
    runs = _merge(runs, gap=params.saccade_merge * sfreq)
    step_n = max(1, int(params.saccade_step_window * sfreq))
    n = len(x)
    events = []
    for a, b in runs:
        before = x[max(0, a - step_n):a]
        after = x[b:min(n, b + step_n)]
        if not len(before) or not len(after):
            continue
        step = after.mean() - before.mean()
        if abs(step) < params.saccade_min_step:
            continue
        start = max(0., a / sfreq - params.saccade_margin[0])
        stop = min(n / sfreq, b / sfreq + params.saccade_margin[1])
        events.append({
            'onset': start, 'duration': stop - start, 'trial_type': 'saccade', 'channel': channel,
            'amplitude': step, 'peak_velocity': float(np.abs(velocity[a:b]).max()), 'source': 'auto',
        })
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def label_segment(veog, heog, sfreq, params=OcularParams()):
    """Blinks and saccades in one segment; onsets in s from segment start."""
    blinks, polarity = detect_blinks(veog, sfreq, params)
    blink_windows = list(zip(blinks['onset'], blinks['onset'] + blinks['duration']))
    horizontal = detect_saccades(heog, sfreq, 'HEOG', params)
    vertical = detect_saccades(veog, sfreq, 'VEOG', params, exclude=blink_windows)
    frames = [f for f in (blinks, horizontal, vertical) if len(f)]
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
    return events.sort_values('onset', ignore_index=True), {'veog_polarity': polarity}


def label_recording(raw, segments_df, params=OcularParams()):
    """Ocular events for a preprocessed recording (preprocess.py output),
    segment by segment. Onsets are in the preprocessed recording's time;
    each event also carries its segment, condition and original-recording
    onset."""
    sfreq = raw.info['sfreq']
    veog_all, heog_all = raw.get_data(picks=['VEOG', 'HEOG'])
    frames, polarity = [], {}
    for seg in segments_df.itertuples():
        events, meta = label_segment(veog_all[seg.start:seg.stop], heog_all[seg.start:seg.stop], sfreq, params)
        polarity[int(seg.index)] = meta['veog_polarity']
        if not len(events):
            continue
        events['orig_onset'] = seg.orig_onset + events['onset']
        events['onset'] = seg.onset + events['onset']
        events['segment'] = seg.index
        events['condition'] = seg.condition
        frames.append(events)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
    return events, {'veog_polarity': polarity}
