"""Detector preprocessing, identical for every dataset (PROJECT_PLAN.md §6).

resample to the detector rate -> (optional: inject simulated pops) ->
0.5 Hz high-pass (zero-phase) -> the 57 detector channels -> average
reference over those channels. In Section A the pipeline injects artifacts
before this step too, so training and use see the same processing.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from adr.preprocessing.channels import average_reference, detector_channels, pick, resample


def to_rate(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """De-mean (raw PhysioMotion is DC-dominated) and resample."""
    return resample(data - data.mean(axis=0), fs_in, fs_out)


def highpass(data: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    sos = scipy.signal.butter(4, cutoff, "highpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, data, axis=0)


def finish(data: np.ndarray, fs: float, ch_names, cutoff: float = 0.5) -> np.ndarray:
    """High-pass, pick the detector channels, average-reference them."""
    return average_reference(pick(highpass(data, fs, cutoff), ch_names, detector_channels()))
