"""Routed reconstruction (PROJECT_PLAN.md §8): the detector's class decides the method.

    ocular -> MWF fitted on the ocular flags, then frontal-reference ridge
    motion -> MWF fitted on the motion flags (spatial interpolation: ablation)
    muscle, pop -> learned spatial interpolation of the contaminated channels
                   (windows with too many contaminated channels fall back to MWF)

Only flagged samples change. An over-cleaning cap limits the fraction of the
recording that may be modified (the highest-scoring windows are kept), and
corrected segments are blended in with raised-cosine ramps (`taper_s`; 0
disables it - an ablation). `single_mwf` is the "one MWF for every
artifact type" comparison (routing ablation; also the oracle-flag MWF
baseline when given the true flags). All settings come from
configs/reconstruction.yaml and are tuned on Section A Set A only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adr.config import load_yaml
from adr.reconstruction import mwf, ridge_frontal
from adr.reconstruction import spatial_interp as si

CLASSES = ("ocular", "muscle", "motion", "pop")


def mask_from_windows(flags: np.ndarray, starts_s: np.ndarray, window_s: float, n_samples: int, fs: float) -> np.ndarray:
    """Per-sample mask: True inside any flagged window."""
    mask = np.zeros(n_samples, bool)
    for s0 in starts_s[np.asarray(flags, bool)]:
        mask[int(round(s0 * fs)):int(round((s0 + window_s) * fs))] = True
    return mask


def cap_windows(flags: dict, scores: dict | None, starts_s, window_s, n_samples, fs, max_fraction) -> dict:
    """Keep the highest-scoring flagged windows until the modified fraction reaches the cap."""
    union = np.any([np.asarray(flags[c], bool) for c in flags], axis=0)
    if mask_from_windows(union, starts_s, window_s, n_samples, fs).mean() <= max_fraction or scores is None:
        return flags
    best = np.max([np.where(flags[c], scores[c], -np.inf) for c in flags], axis=0)
    keep = np.zeros(len(starts_s), bool)
    covered = np.zeros(n_samples, bool)
    for i in np.argsort(best)[::-1]:
        if not np.isfinite(best[i]):
            break
        span = slice(int(round(starts_s[i] * fs)), int(round((starts_s[i] + window_s) * fs)))
        trial = covered.copy()
        trial[span] = True
        if trial.mean() > max_fraction:
            break
        covered, keep[i] = trial, True
    return {c: np.asarray(flags[c], bool) & keep for c in flags}


def taper_weights(mask: np.ndarray, taper: int) -> np.ndarray:
    """1 inside modified segments with raised-cosine ramps at their edges, 0 outside."""
    w = mask.astype(float)
    if taper <= 0:
        return w
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    for a, b in zip(edges[::2], edges[1::2]):
        ramp = min(taper, (b - a) // 2)
        if ramp > 0:
            up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp)
            w[a:a + ramp] = up
            w[b - ramp:b] = up[::-1]
    return w


@dataclass
class Result:
    eeg: np.ndarray
    modified: np.ndarray                       # per-sample mask of changed samples
    info: dict = field(default_factory=dict)


def _fit_mwf(eeg, artifact_mask, clean_mask, cfg):
    if artifact_mask.sum() < cfg["min_samples"] or clean_mask.sum() < cfg["min_samples"]:
        return None
    return mwf.fit(eeg, artifact_mask, clean_mask, delay=cfg["delay"], rank=cfg["rank"], mu=cfg["mu"])


def route(eeg: np.ndarray, fs: float, ch_names, flags: dict, starts_s: np.ndarray, window_s: float,
          scores: dict | None = None, cfg: dict | None = None) -> Result:
    """Detector-routed reconstruction of one recording (samples x channels)."""
    cfg = cfg or load_yaml("reconstruction")
    ch_names = list(ch_names)
    n = len(eeg)
    flags = {c: np.asarray(flags.get(c, np.zeros(len(starts_s), bool)), bool) for c in CLASSES}
    flags = cap_windows(flags, scores, starts_s, window_s, n, fs, cfg["max_modified_fraction"])
    masks = {c: mask_from_windows(flags[c], starts_s, window_s, n, fs) for c in CLASSES}
    union = np.any(list(masks.values()), axis=0)
    clean = ~union
    out = eeg.copy()
    info = {"fraction_modified": float(union.mean()), **{f"fraction_{c}": float(masks[c].mean()) for c in CLASSES}}

    # ocular: MWF (+ frontal ridge)
    if masks["ocular"].any():
        method = cfg["ocular_method"]
        if "mwf" in method:
            model = _fit_mwf(out, masks["ocular"], clean, cfg["mwf"])
            if model is not None:
                out[masks["ocular"]] = model.clean(out)[masks["ocular"]]
                info["ocular_mwf_rank"] = model.rank
        if "ridge" in method:
            rf = cfg["ridge_frontal"]
            out, _ = ridge_frontal.correct(out, masks["ocular"], ch_names, rf["reference"], rf["alpha"])

    # motion: MWF (or interpolation, ablation)
    interp_classes = ["muscle", "pop"] + (["motion"] if cfg["motion_method"] == "interp" else [])
    if cfg["motion_method"] == "mwf" and masks["motion"].any():
        model = _fit_mwf(out, masks["motion"], clean, cfg["mwf"])
        if model is not None:
            out[masks["motion"]] = model.clean(out)[masks["motion"]]
            info["motion_mwf_rank"] = model.rank

    # muscle / pop (/ motion): learned spatial interpolation of contaminated channels
    sc = cfg["spatial_interp"]
    win = int(round(window_s * fs))
    all_starts = (np.asarray(starts_s) * fs).round().astype(int)
    clean_windows = np.array([clean[s:s + win].all() for s in all_starts])
    if any(masks[c].any() for c in interp_classes) and clean_windows.sum() >= 10:
        interp = si.fit(eeg, clean, ch_names, sc["n_neighbours"], sc["alpha"])
        fallback_model, n_fallback, n_interp, n_untouched = None, 0, 0, 0
        for c in interp_classes:
            band = tuple(sc["band_muscle"]) if c == "muscle" else ("step" if c == "pop" else None)
            if not flags[c].any():
                continue
            scores_c = si.window_scores(eeg, fs, all_starts, win, band)
            bad = si.bad_channels(scores_c, scores_c[clean_windows], sc["bad_z"])
            for i in np.flatnonzero(flags[c]):
                span = slice(all_starts[i], all_starts[i] + win)
                bad_idx = list(np.flatnonzero(bad[i]))
                if not bad_idx:
                    n_untouched += 1
                    continue
                if len(bad_idx) > sc["max_bad_fraction"] * len(ch_names):
                    if fallback_model is None:
                        fallback_model = _fit_mwf(eeg, masks[c], clean, cfg["mwf"])
                    if fallback_model is not None:
                        out[span] = fallback_model.clean(out[span])
                    n_fallback += 1
                    continue
                out[span, bad_idx] = interp.predict(out[span], bad_idx)
                n_interp += 1
        info.update({"interp_windows": n_interp, "interp_fallback_windows": n_fallback,
                     "interp_windows_without_bad_channels": n_untouched})

    weights = taper_weights(union, int(round(cfg["taper_s"] * fs)))[:, None]
    final = eeg + weights * (out - eeg)
    return Result(eeg=final, modified=union, info=info)


def single_mwf(eeg: np.ndarray, artifact_mask: np.ndarray, cfg: dict | None = None,
               taper_s: float | None = None, fs: float = 128.0) -> Result:
    """One MWF for every artifact type, correcting flagged samples only."""
    cfg = cfg or load_yaml("reconstruction")
    mask = np.asarray(artifact_mask, bool)
    out = eeg.copy()
    model = _fit_mwf(eeg, mask, ~mask, cfg["mwf"])
    if model is not None:
        out[mask] = model.clean(eeg)[mask]
    taper = cfg["taper_s"] if taper_s is None else taper_s
    final = eeg + taper_weights(mask, int(round(taper * fs)))[:, None] * (out - eeg)
    return Result(eeg=final, modified=mask, info={"fraction_modified": float(mask.mean()),
                                                  "mwf_rank": model.rank if model is not None else 0})
