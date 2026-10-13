"""Weak muscle (EMG) labels from high-frequency power on temporal channels.

Jaw clenching, swallowing and head/neck tension put broadband power above
~20 Hz on the temporal electrodes (closest to the temporalis muscles),
where cortical EEG has very little. Each sliding window's log power in
EMG_BAND (Welch, line-noise harmonics excluded) is z-scored per channel
against that subject's own median/MAD over the whole recording; windows
whose highest temporal-channel z exceeds `z_threshold` are flagged, and
overlapping flagged windows merged into events.

These are *weak* labels (source='weak'): a high-frequency power threshold
also catches loose electrodes and saccadic spike potentials on the
frontotemporal sites. They must be verified by hand (labels/review.py)
before being used as training targets - the review sheets mark which ones
survived.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch

from .. import config
from .ocular import EVENT_COLUMNS, _merge


@dataclass
class EMGParams:
    band: tuple = (20., 100.)   # Hz; capped at 0.45 x sampling rate
    window: float = 0.5         # s
    step: float = 0.25          # s
    z_threshold: float = 4.
    line_freq: float = 60.      # harmonics +/- line_width excluded from the band
    line_width: float = 2.
    channels: tuple = config.TEMPORAL_CHANNELS


def window_band_power(data, sfreq, params=EMGParams()):
    """Log band power per channel and window: (n_channels, n_windows), and
    each window's start sample."""
    n_win = int(round(params.window * sfreq))
    n_step = int(round(params.step * sfreq))
    starts = np.arange(0, data.shape[1] - n_win + 1, n_step)
    if not len(starts):
        return np.empty((data.shape[0], 0)), starts
    windows = np.stack([data[:, s:s + n_win] for s in starts], axis=1)  # (ch, win, time)
    freqs, psd = welch(windows, fs=sfreq, nperseg=min(n_win, int(sfreq / 4)), axis=-1)
    high = min(params.band[1], 0.45 * sfreq)
    in_band = (freqs >= params.band[0]) & (freqs <= high)
    if params.line_freq:
        for harmonic in np.arange(params.line_freq, freqs[-1] + 1, params.line_freq):
            in_band &= np.abs(freqs - harmonic) > params.line_width
    power = psd[..., in_band].mean(axis=-1)
    return np.log10(power + 1e-30), starts


def label_recording(raw, segments_df, params=EMGParams()):
    """EMG candidate events for a preprocessed recording.

    The baseline (median/MAD) is pooled over all of the subject's
    segments. Constant tension therefore raises the baseline itself and
    is not flagged - only bursts above the subject's own typical level
    are. Review sheets are where that blind spot gets caught."""
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks=list(params.channels))
    per_segment = []
    for seg in segments_df.itertuples():
        power, starts = window_band_power(data[:, seg.start:seg.stop], sfreq, params)
        per_segment.append((seg, power, starts))
    pooled = np.concatenate([p for _, p, _ in per_segment], axis=1)
    median = np.median(pooled, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(pooled - median), axis=1, keepdims=True)

    frames = []
    for seg, power, starts in per_segment:
        z = (power - median) / np.maximum(mad, 1e-12)
        score = z.max(axis=0)
        flagged = np.flatnonzero(score > params.z_threshold)
        windows = [(starts[i] / sfreq, starts[i] / sfreq + params.window) for i in flagged]
        rows = []
        for start, stop in _merge(windows):
            inside = [i for i in flagged if start <= starts[i] / sfreq < stop]
            best = max(inside, key=lambda i: score[i])
            rows.append({
                'onset': seg.onset + start, 'duration': stop - start, 'trial_type': 'emg',
                'channel': params.channels[int(z[:, best].argmax())],
                'amplitude': float(score[best]), 'peak_velocity': np.nan, 'source': 'weak',
                'orig_onset': seg.orig_onset + start, 'segment': seg.index, 'condition': seg.condition,
            })
        if rows:
            frames.append(pd.DataFrame(rows))
    columns = EVENT_COLUMNS + ['orig_onset', 'segment', 'condition']
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
