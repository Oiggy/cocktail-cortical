"""Das et al. (2020) EEG preprocessing for the Section A TRF analysis (PROJECT_PLAN.md §9.1).

Follows the authors' `preprocess_data.m` (shipped with Dataset I v2.0):
re-reference to Cz, band-pass 1-9 Hz with an equiripple FIR applied
forward-backward at 128 Hz, then decimate by 4 to 32 Hz without further
filtering, drop Cz (63 channels) and z-score each channel. The plan lists
band-pass, downsample, re-reference; re-referencing and filtering are both
linear, so the order does not change the result.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.signal

FS_IN = 128.0
FS_OUT = 32.0
CZ_INDEX = 47  # 0-based column of Cz in the BioSemi 64 order (verified)

# fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2', ...) in preprocess_data.m:
# edges at highpass -/+ 0.45 Hz and lowpass -/+ 0.45 Hz.
HIGHPASS, LOWPASS, HALF_TRANSITION = 1.0, 9.0, 0.45
STOP1_DB, PASS_RIPPLE_DB, STOP2_DB = 20.0, 0.5, 15.0


def _meets_spec(taps: np.ndarray, fs: float) -> bool:
    freqs, response = scipy.signal.freqz(taps, worN=8192, fs=fs)
    gain = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
    stop1 = freqs <= HIGHPASS - HALF_TRANSITION
    passband = (freqs >= HIGHPASS + HALF_TRANSITION) & (freqs <= LOWPASS - HALF_TRANSITION)
    stop2 = freqs >= LOWPASS + HALF_TRANSITION
    return (
        gain[stop1].max() <= -STOP1_DB
        and gain[stop2].max() <= -STOP2_DB
        and gain[passband].max() - gain[passband].min() <= PASS_RIPPLE_DB
    )


@lru_cache(maxsize=None)
def bandpass_taps(fs: float = FS_IN) -> np.ndarray:
    """Minimum-length equiripple 1-9 Hz FIR meeting the MATLAB design spec.

    MATLAB's `design(..., 'equiripple')` picks the minimum order; here the
    Parks-McClellan design is grown until the same attenuation and ripple
    specs are met.
    """
    bands = [0, HIGHPASS - HALF_TRANSITION, HIGHPASS + HALF_TRANSITION,
             LOWPASS - HALF_TRANSITION, LOWPASS + HALF_TRANSITION, fs / 2]
    ripple = (10 ** (PASS_RIPPLE_DB / 20) - 1) / (10 ** (PASS_RIPPLE_DB / 20) + 1)
    weights = [1 / 10 ** (-STOP1_DB / 20), 1 / ripple, 1 / 10 ** (-STOP2_DB / 20)]
    for numtaps in range(51, 2001, 2):
        taps = scipy.signal.remez(numtaps, bands, [0, 1, 0], weight=weights, fs=fs, maxiter=100)
        if _meets_spec(taps, fs):
            return taps
    raise RuntimeError("No equiripple design met the spec")


def bandpass(data: np.ndarray, fs: float = FS_IN) -> np.ndarray:
    """1-9 Hz zero-phase band-pass along axis 0 (filtfilt, as in MATLAB)."""
    return scipy.signal.filtfilt(bandpass_taps(fs), [1.0], data, axis=0)


def decimate(data: np.ndarray, fs_in: float = FS_IN, fs_out: float = FS_OUT) -> np.ndarray:
    """Keep every k-th sample (MATLAB `downsample`; the band-pass is the anti-alias)."""
    factor = fs_in / fs_out
    if factor != int(factor):
        raise ValueError(f"Non-integer decimation {fs_in} -> {fs_out}")
    return data[:: int(factor)]


def zscore(data: np.ndarray) -> np.ndarray:
    return (data - data.mean(axis=0)) / data.std(axis=0)


def preprocess_eeg(eeg: np.ndarray, cz_index: int = CZ_INDEX) -> np.ndarray:
    """128 Hz, 64-channel trial -> 32 Hz, 63-channel Cz-referenced EEG (not z-scored).

    z-scoring is left to the caller so it can be done over whatever data a
    model is fitted on (PROJECT_PLAN.md §9.1 step 6).
    """
    referenced = eeg - eeg[:, [cz_index]]
    filtered = decimate(bandpass(referenced))
    return np.delete(filtered, cz_index, axis=1)
