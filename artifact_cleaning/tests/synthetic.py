"""Synthetic EEG/EOG with injected artifacts at known times, for tests.

Shapes and amplitudes are textbook-typical (blinks ~100-200 uV and
~200-400 ms on VEOG; saccades as ~30 ms EOG steps of tens of uV; EMG as
band-limited 20-100 Hz bursts) - enough to check the detectors find what
was injected, not a model of any particular subject.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

UV = 1e-6


def pink_noise(n, rng, rms=10 * UV):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n)
    return x / x.std() * rms


def blink_shape(sfreq, duration=0.3):
    """Smooth unimodal bump, peak 1, lasting `duration`."""
    t = np.arange(int(duration * sfreq)) / sfreq
    return np.sin(np.pi * t / duration) ** 2


def step_shape(sfreq, n, onset, rise=0.03):
    """0 -> 1 sigmoid step centred at sample `onset`."""
    t = (np.arange(n) - onset) / sfreq
    return expit(t / (rise / 8))


def highpass(x, sfreq, cutoff=0.5):
    sos = butter(2, cutoff, btype='highpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, x)


def event_times(rng, duration, n, min_gap, edge=2.):
    """n sorted random times in [edge, duration - edge], at least min_gap apart.

    Spreads the spare time (span minus the n-1 mandatory gaps) randomly,
    rather than rejection sampling, which stalls near the packing limit."""
    slack = (duration - 2 * edge) - (n - 1) * min_gap
    if slack < 0:
        raise ValueError(f'{n} events {min_gap} s apart do not fit in {duration} s')
    return edge + np.sort(rng.uniform(0, slack, n)) + np.arange(n) * min_gap


def ocular_segment(sfreq=256., duration=120., n_blinks=None, n_saccades=None, blink_amp=150 * UV,
                   saccade_amp=60 * UV, veog_sign=1., seed=0):
    """(veog, heog, blink_onsets_s, saccade_onsets_s) for one segment.
    Default event counts: one blink and one saccade per 5 s."""
    rng = np.random.default_rng(seed)
    n_blinks = int(duration / 5) if n_blinks is None else n_blinks
    n_saccades = int(duration / 5) if n_saccades is None else n_saccades
    n = int(duration * sfreq)
    times = event_times(rng, duration, n_blinks + n_saccades, min_gap=1.5)
    order = rng.permutation(len(times))
    blink_times = np.sort(times[order[:n_blinks]])
    saccade_times = np.sort(times[order[n_blinks:]])

    veog = pink_noise(n, rng)
    shape = blink_shape(sfreq)
    for t in blink_times:
        start = int(t * sfreq) - len(shape) // 2
        veog[start:start + len(shape)] += veog_sign * blink_amp * rng.uniform(0.8, 1.2) * shape

    level = np.zeros(n)
    for t in saccade_times:
        level += rng.choice([-1, 1]) * saccade_amp * rng.uniform(0.8, 1.5) * step_shape(sfreq, n, int(t * sfreq))
    heog = pink_noise(n, rng) + highpass(level, sfreq)
    return veog, heog, blink_times, saccade_times


def emg_channels(sfreq=256., duration=120., n_channels=6, n_bursts=8, burst_duration=1., amp=8., seed=0):
    """(data, burst_onsets_s): EEG-like pink noise plus 20-100 Hz bursts,
    `amp` times the background's high-frequency amplitude, on a random
    subset of channels each time."""
    rng = np.random.default_rng(seed)
    n = int(duration * sfreq)
    data = np.stack([pink_noise(n, rng) for _ in range(n_channels)])
    sos = butter(4, [20, 100], btype='bandpass', fs=sfreq, output='sos')
    background_hf = sosfiltfilt(sos, data, axis=1).std()
    onsets = event_times(rng, duration, n_bursts, min_gap=4.)
    m = int(burst_duration * sfreq)
    envelope = np.hanning(m)
    for t in onsets:
        start = int(t * sfreq)
        channels = rng.choice(n_channels, size=rng.integers(1, n_channels + 1), replace=False)
        for ch in channels:
            burst = sosfiltfilt(sos, rng.standard_normal(m))
            burst = burst / burst.std() * amp * background_hf
            data[ch, start:start + m] += burst * envelope
    return data, onsets


def segments_df(durations, sfreq, conditions=None):
    rows, start = [], 0
    for i, duration in enumerate(durations, start=1):
        n = int(round(duration * sfreq))
        rows.append({'index': i, 'condition': (conditions or ['clean'] * len(durations))[i - 1], 'side': '',
                     'fg': f'male_{i}', 'bg': '', 'orig_onset': 100. * i, 'onset': start / sfreq,
                     'duration': n / sfreq, 'start': start, 'stop': start + n, 'truncated': False})
        start += n
    return pd.DataFrame(rows)


def match(true_times, events, tolerance=0.):
    """Recall: fraction of true times inside some event window (+/- tolerance).
    Precision: fraction of event windows containing some true time."""
    starts = events['onset'].to_numpy() - tolerance
    stops = (events['onset'] + events['duration']).to_numpy() + tolerance
    hit = [np.any((starts <= t) & (t <= stops)) for t in true_times]
    used = [np.any((start <= true_times) & (true_times <= stop)) for start, stop in zip(starts, stops)]
    recall = float(np.mean(hit)) if len(hit) else np.nan
    precision = float(np.mean(used)) if len(used) else np.nan
    return recall, precision
