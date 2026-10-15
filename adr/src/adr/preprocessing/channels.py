"""Channel alignment shared by every dataset (PROJECT_PLAN.md §6).

Arrays are samples x channels throughout, matching the Dataset I loader.
"""
from __future__ import annotations

from fractions import Fraction

import mne
import numpy as np
import scipy.signal

from adr.config import load_project


def detector_channels() -> list[str]:
    """The frozen 57-channel detector set (configs/project.yaml)."""
    return list(load_project()["detector_channels"])


def label_electrodes() -> list[str]:
    """The 19 electrodes of PhysioMotion's annotated bipolar montage."""
    return list(load_project()["label_electrodes"])


def pick(data: np.ndarray, ch_names, wanted) -> np.ndarray:
    """Columns of `data` for the channels in `wanted`, in that order."""
    index = {name: i for i, name in enumerate(ch_names)}
    missing = [ch for ch in wanted if ch not in index]
    if missing:
        raise KeyError(f"Channels not present: {missing}")
    return data[:, [index[ch] for ch in wanted]]


def average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels at each sample."""
    return data - data.mean(axis=1, keepdims=True)


def interpolate_bads(raw: mne.io.BaseRaw, bads) -> mne.io.BaseRaw:
    """Spherical-spline interpolation of `bads` (raw needs a montage)."""
    raw = raw.copy()
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    return raw


def resample(data: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """Polyphase resampling along samples (anti-aliased).

    Ratios used here: 1000 -> 128 (16/125) and 1000 -> 250 (1/4) for
    PhysioMotion; 512 -> 128 (1/4) and 512 -> 250 (125/256) for the phantom.
    """
    if sfreq_in == sfreq_out:
        return data
    ratio = Fraction(sfreq_out / sfreq_in).limit_denominator(1000)
    if not np.isclose(float(ratio), sfreq_out / sfreq_in):
        raise ValueError(f"No exact rational ratio for {sfreq_in} -> {sfreq_out} Hz")
    return scipy.signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=0)
