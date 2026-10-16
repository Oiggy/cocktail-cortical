"""Automatic-marking MWF baseline, RELAX-style (Bailey et al. 2023; PROJECT_PLAN.md §11, optional).

Artifacts are marked without a trained detector:
- blinks: 1 s windows whose frontal (Fp1/Fp2/AF7/AF8 mean) peak-to-peak
  exceeds Q3 + k * IQR over the recording's windows;
- muscle: windows where any channel's log-log PSD slope over 7-60 Hz is
  flatter than -0.59 (RELAX's default; RELAX uses 7-75 Hz, capped here
  below the 64 Hz Nyquist of Section A);
- extreme outliers: windows where any channel's peak-to-peak exceeds
  Q3 + 5 IQR (all channels pooled).
Then MWFs are applied sequentially (muscle, then blink), each fitted on its
own marks against the unmarked data, and - as in RELAX - applied to the
whole recording rather than only the marked samples.

`heuristic_flags` is also the "heuristic flags" condition of the
detection-quality ablation (§12).
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.signal

from adr.config import load_yaml
from adr.reconstruction import mwf, router

log = logging.getLogger("adr.mwf_auto")


def _iqr_threshold(x: np.ndarray, k: float) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return q3 + k * (q3 - q1)


def muscle_slopes(eeg: np.ndarray, fs: float, cfg: dict | None = None) -> np.ndarray:
    """Per window, the flattest channel's log-log PSD slope in the muscle band (RELAX statistic)."""
    cfg = cfg or load_yaml("baselines")["mwf_auto"]
    win, step = int(cfg["window_s"] * fs), int(cfg["step_s"] * fs)
    starts = np.arange(0, len(eeg) - win + 1, step)
    lo, hi = cfg["muscle_slope_band_hz"]
    hi = min(hi, fs / 2 - 1)
    freqs, psd = scipy.signal.welch(np.stack([eeg[s:s + win] for s in starts]), fs=fs, nperseg=win, axis=1)
    band = (freqs >= lo) & (freqs <= hi)
    logf = np.log10(freqs[band])
    logp = np.log10(psd[:, band] + 1e-30)
    slope = ((logf - logf.mean())[None, :, None] * (logp - logp.mean(axis=1, keepdims=True))).sum(1) / \
        ((logf - logf.mean()) ** 2).sum()
    return slope.max(axis=1)


def calibrate_muscle_threshold(subjects=None, target_rate: float = 0.05, snr_db: float = -10) -> tuple[float, dict]:
    """Slope threshold marking `target_rate` of artifact-free Section A windows (Set A only).

    Uses each Set A noise subject's synthetic recording without injected
    artifacts (attend-left, `snr_db`); subjects are weighted equally (the
    threshold is the median over subjects of each subject's (1 - target_rate)
    quantile). Returns the threshold and per-subject details.
    """
    from adr.synth import dataset3 as d3

    subjects = subjects or d3.load_noise_split()["set_a"]
    base, lags, _ = d3.load_base_trfs()
    per_subject = {}
    for subject in subjects:
        rec = d3.synthesize(d3.noise_source(subject), "L", snr_db, base, lags)
        per_subject[subject] = muscle_slopes(rec.eeg, 128.0)
    quantiles = {s: float(np.quantile(v, 1 - target_rate)) for s, v in per_subject.items()}
    threshold = float(np.median(list(quantiles.values())))
    details = {"quantile_per_subject": quantiles,
               "marked_at_threshold": {s: float(np.mean(v > threshold)) for s, v in per_subject.items()},
               "marked_at_relax_default": {s: float(np.mean(v > -0.59)) for s, v in per_subject.items()}}
    return threshold, details


def heuristic_flags(eeg: np.ndarray, fs: float, ch_names, cfg: dict | None = None) -> tuple[dict, np.ndarray]:
    """Per-window flags {'ocular', 'muscle', 'extreme'} and window starts (s)."""
    cfg = cfg or load_yaml("baselines")["mwf_auto"]
    ch_names = list(ch_names)
    win, step = int(cfg["window_s"] * fs), int(cfg["step_s"] * fs)
    starts = np.arange(0, len(eeg) - win + 1, step)
    frontal = eeg[:, [ch_names.index(c) for c in cfg["blink_channels"]]].mean(axis=1)
    sos = scipy.signal.butter(4, [1, 25], "bandpass", fs=fs, output="sos")
    frontal = scipy.signal.sosfiltfilt(sos, frontal)
    fp_ptp = np.array([np.ptp(frontal[s:s + win]) for s in starts])
    ptp = np.array([np.ptp(eeg[s:s + win], axis=0) for s in starts])           # windows x channels
    flags = {
        "ocular": fp_ptp > _iqr_threshold(fp_ptp, cfg["blink_iqr_k"]),
        "muscle": muscle_slopes(eeg, fs, cfg) > cfg["muscle_slope_threshold"],
        "extreme": (ptp > _iqr_threshold(ptp.ravel(), cfg["extreme_iqr_k"])).any(axis=1),
    }
    return flags, starts / fs


def run(eeg: np.ndarray, fs: float, ch_names, cfg: dict | None = None, mwf_cfg: dict | None = None):
    """RELAX-style cleaning. Returns (cleaned eeg, info dict with flags and fractions)."""
    cfg = cfg or load_yaml("baselines")["mwf_auto"]
    mwf_cfg = mwf_cfg or load_yaml("reconstruction")["mwf"]
    flags, starts = heuristic_flags(eeg, fs, ch_names, cfg)
    masks = {k: router.mask_from_windows(v, starts, cfg["window_s"], len(eeg), fs) for k, v in flags.items()}
    out = eeg.copy()
    info = {f"fraction_{k}": float(m.mean()) for k, m in masks.items()}
    marked_any = np.any(list(masks.values()), axis=0)
    for kind in cfg["order"]:
        key = "ocular" if kind == "blink" else kind
        mask = masks[key]
        if mask.sum() < mwf_cfg["min_samples"] or (~marked_any).sum() < mwf_cfg["min_samples"]:
            continue
        model = mwf.fit(out, mask, ~marked_any, delay=mwf_cfg["delay"], rank=mwf_cfg["rank"], mu=mwf_cfg["mu"])
        out = model.clean(out)                 # whole recording, as in RELAX
        info[f"{kind}_mwf_rank"] = model.rank
    info["flags"], info["starts"] = flags, starts
    log.info("auto-MWF: marked muscle %.1f %%, blink %.1f %%, extreme %.1f %%", 100 * info["fraction_muscle"],
             100 * info["fraction_ocular"], 100 * info["fraction_extreme"])
    return out, info
