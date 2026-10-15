"""Window features for the artifact detectors (PROJECT_PLAN.md §7).

Time-domain and FFT/Welch features only (no wavelets, no ICA). Per channel:
log variance, log peak-to-peak, kurtosis, log line length, Hjorth mobility
and complexity, log Welch band powers, and correlation with the frontal
reference (mean of Fp1, Fp2). Global: log frontal / posterior RMS ratio.
Hjorth activity is the variance, so it is not repeated.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.stats

from adr.config import load_yaml

EPS = 1e-20


def detection_config() -> dict:
    return load_yaml("detection")


def bands_for(fs: float, cfg: dict | None = None) -> dict[str, list[float]]:
    cfg = cfg or detection_config()
    bands = dict(cfg["bands"])
    if fs >= 250:
        bands.update(cfg["bands_250"])
    nyquist = fs / 2
    if any(hi >= nyquist for _, hi in bands.values()):
        raise ValueError(f"A band reaches Nyquist ({nyquist} Hz)")
    return bands


def window_starts(n_samples: int, fs: float, window_s: float, step_s: float) -> np.ndarray:
    win, step = int(round(window_s * fs)), int(round(step_s * fs))
    return np.arange(0, n_samples - win + 1, step)


def windows(data: np.ndarray, fs: float, window_s: float, step_s: float) -> np.ndarray:
    """n_windows x window_samples x channels view."""
    win = int(round(window_s * fs))
    starts = window_starts(len(data), fs, window_s, step_s)
    view = np.lib.stride_tricks.sliding_window_view(data, win, axis=0)  # n x ch x win
    return np.moveaxis(view[starts], 1, 2)


def feature_names(ch_names, fs: float, cfg: dict | None = None) -> list[str]:
    cfg = cfg or detection_config()
    per_channel = ["log_var", "log_ptp", "kurtosis", "log_line_length", "hjorth_mobility",
                   "hjorth_complexity", *[f"log_power_{b}" for b in bands_for(fs, cfg)], "corr_frontal"]
    return [f"{f}:{ch}" for f in per_channel for ch in ch_names] + ["log_frontal_posterior_ratio"]


def compute(data: np.ndarray, fs: float, ch_names, cfg: dict | None = None) -> np.ndarray:
    """Features for every window of `data` (samples x channels, volts).

    Returns n_windows x n_features (float32), columns as in `feature_names`.
    """
    cfg = cfg or detection_config()
    ch_names = list(ch_names)
    w = windows(data, fs, cfg["window_s"], cfg["step_s"])          # n x t x ch
    w = w - w.mean(axis=1, keepdims=True)
    var = w.var(axis=1)
    d1 = np.diff(w, axis=1)
    d2 = np.diff(d1, axis=1)
    var_d1, var_d2 = d1.var(axis=1), d2.var(axis=1)
    mobility = np.sqrt(var_d1 / (var + EPS))
    complexity = np.sqrt(var_d2 / (var_d1 + EPS)) / (mobility + EPS)

    freqs, psd = scipy.signal.periodogram(w, fs=fs, window="hann", axis=1)  # n x f x ch
    powers = [np.log10(psd[:, (freqs >= lo) & (freqs < hi)].mean(axis=1) + EPS)
              for lo, hi in bands_for(fs, cfg).values()]

    ref_idx = [ch_names.index(c) for c in cfg["frontal_reference"]]
    ref = w[:, :, ref_idx].mean(axis=2, keepdims=True)
    ref_c = ref / (np.sqrt((ref ** 2).sum(axis=1, keepdims=True)) + EPS)
    w_c = w / (np.sqrt((w ** 2).sum(axis=1, keepdims=True)) + EPS)
    corr = (w_c * ref_c).sum(axis=1)

    rms = np.sqrt(var)
    front = rms[:, [ch_names.index(c) for c in cfg["frontal"]]].mean(axis=1)
    back = rms[:, [ch_names.index(c) for c in cfg["posterior"]]].mean(axis=1)

    blocks = [np.log10(var + EPS), np.log10(np.ptp(w, axis=1) + EPS),
              scipy.stats.kurtosis(w, axis=1), np.log10(np.abs(d1).mean(axis=1) + EPS),
              mobility, complexity, *powers, corr]
    out = np.concatenate([*blocks, np.log10((front + EPS) / (back + EPS))[:, None]], axis=1)
    return out.astype(np.float32)
