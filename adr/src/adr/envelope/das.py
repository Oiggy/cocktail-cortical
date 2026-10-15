"""Speech envelope of Das et al. (2020) / Biesmans et al. (2016) (PROJECT_PLAN.md §9.1).

Follows the authors' `preprocess_data.m`: resample the dry stimulus to
8 kHz, filter with 15 real gammatone FIR filters (AMToolbox
`gammatonefir`, centre frequencies `erbspacebw(150, 4000, 1.5)`, bandwidth
multiplier 1.5), take |.|^0.6 per subband, resample to 128 Hz, band-pass
1-9 Hz (same equiripple filter as the EEG), decimate to 32 Hz, and sum the
subbands (subband weights are all 1). The plan lists "downsample to 32 Hz,
then band-pass"; the authors' code band-passes at 128 Hz first, which is
what is implemented here.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal

from adr.preprocessing.das import FS_IN, FS_OUT, bandpass, decimate

AUDIO_FS = 8000.0
POWER = 0.6
GAMMATONE_ORDER = 4
BW_MULTIPLIER = 1.5
F_LOW, F_HIGH, SPACING, HIT_ME = 150.0, 4000.0, 1.5, 1000.0


def freq_to_erb(f):
    """ERB-rate scale (AMToolbox `freqtoaud(f, 'erb')`)."""
    return 9.2645 * np.sign(f) * np.log(1 + np.abs(f) * 0.00437)


def erb_to_freq(e):
    return (1 / 0.00437) * np.sign(e) * (np.exp(np.abs(e) / 9.2645) - 1)


def erb_bandwidth(f):
    """Equivalent rectangular bandwidth in Hz (AMToolbox `audfiltbw`)."""
    return 24.7 + f / 9.265


def centre_frequencies() -> np.ndarray:
    """`erbspacebw(150, 4000, 1.5)`: 1.5-ERB spacing with one filter at 1 kHz."""
    e_low, e_high, e_hit = freq_to_erb(F_LOW), freq_to_erb(F_HIGH), freq_to_erb(HIT_ME)
    k = np.arange(np.ceil((e_low - e_hit) / SPACING), np.floor((e_high - e_hit) / SPACING) + 1)
    return erb_to_freq(e_hit + k * SPACING)


def gammatone_bank(fs: float = AUDIO_FS) -> list[np.ndarray]:
    """Real 4th-order gammatone FIR impulse responses, unit gain at each centre.

    g(t) = t^(n-1) exp(-2 pi b t) cos(2 pi fc t), b = 1.5 * ERB(fc), truncated
    once the envelope has decayed by 100 dB.
    """
    filters = []
    for fc in centre_frequencies():
        b = BW_MULTIPLIER * erb_bandwidth(fc)
        # envelope t^3 exp(-2 pi b t) peaks at t = 3 / (2 pi b); run well past it
        duration = 30 / (2 * np.pi * b)
        t = np.arange(int(np.ceil(duration * fs))) / fs
        g = t ** (GAMMATONE_ORDER - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
        gain = np.abs(np.sum(g * np.exp(-2j * np.pi * fc * t)))
        filters.append(g / gain)
    return filters


def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    ratio = Fraction(fs_out / fs_in).limit_denominator(10000)
    return scipy.signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def subband_envelopes(audio: np.ndarray, fs: float) -> np.ndarray:
    """Power-law subband envelopes at 32 Hz (samples x 15), before summing."""
    audio = _resample(audio.astype(np.float64), fs, AUDIO_FS)
    bands = np.column_stack([scipy.signal.oaconvolve(audio, g)[: len(audio)] for g in gammatone_bank()])
    env = np.abs(bands) ** POWER
    env = _resample(env, AUDIO_FS, FS_IN)
    return decimate(bandpass(env, FS_IN), FS_IN, FS_OUT)


def envelope(audio: np.ndarray, fs: float) -> np.ndarray:
    """Broadband envelope at 32 Hz: sum of the 15 subband envelopes."""
    return subband_envelopes(audio, fs).sum(axis=1)


def wav_envelope(path: Path, cache_dir: Path | None = None) -> np.ndarray:
    """Envelope of a wav file, cached as .npy next to `cache_dir` if given."""
    if cache_dir is not None:
        cached = Path(cache_dir) / f"{Path(path).stem}_das-envelope.npy"
        if cached.exists():
            return np.load(cached)
    fs, audio = scipy.io.wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    env = envelope(audio, fs)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(cached, env)
    return env
