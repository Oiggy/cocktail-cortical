"""Artifact templates for injection (PROJECT_PLAN.md §9.3 step 6; milestone 6).

Every template is a short multichannel artifact field on the BioSemi 64
layout (samples x 64, volts, average-referenced, 128 Hz), tagged with its
class (ocular / muscle / motion), source and pool.

Primary source - the phantom (ocular, muscle, motion). The scalp EEG of each
artifact recording minus its brain component leaves artifact + sensor
noise. The brain component is predicted from the broadcast brain sources
with a lagged regression fitted on the brain-only recording (the plan's
"regress them out", since the waveforms are provided). Templates are cut
from each pool's 100 s block (Set A 0-100 s, Set B 100-200 s; the
200-300 s block is reserved for the §9.5 check):
- ocular: Eyes recording, chunks of 1-3 s (blinks are ~1/s, plus saccades);
- muscle: Facial recording, cut at chewing-burst onsets (from the facial
  source waveform), plus Neck recording chunks (continuous EMG);
- motion: Walking recording chunks of 2-6 s.
The field is spline-mapped 128 -> 64 channels, average-referenced across
64 and resampled to 128 Hz. Broadband (no 20 Hz split).

Secondary source - PhysioMotion (ablation): single-class annotated events of
the template participants, isolated with the plan's filters (muscle:
high-pass 20 Hz, also kept broadband; ocular / motion: low-pass), 57 shared
channels mapped directly and the other 7 BioSemi positions spline-
interpolated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mne
import numpy as np
import scipy.signal
import yaml

from adr.config import ADR_ROOT, load_paths, load_yaml
from adr.io import phantom
from adr.io import physiomotion as pm
from adr.preprocessing import spline
from adr.preprocessing.channels import average_reference, interpolate_bads, resample

log = logging.getLogger("adr.templates")
CLASSES = ("ocular", "muscle", "motion")


@dataclass
class Template:
    kind: str            # ocular / muscle / motion
    data: np.ndarray     # samples x 64, volts, average-referenced, at `fs`
    fs: float
    source: str          # e.g. "phantom:Facial" or "physiomotion:sub-2_run-03:chew"
    pool: str            # set_a_templates / set_b_templates
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs


def _lag(x: np.ndarray, lags) -> np.ndarray:
    """samples x (sources * lags) with zero padding (positive lag = source leads)."""
    n = len(x)
    cols = []
    for lag in lags:
        shifted = np.zeros_like(x)
        if lag >= 0:
            shifted[lag:] = x[: n - lag]
        else:
            shifted[:lag] = x[-lag:]
        cols.append(shifted)
    return np.concatenate(cols, axis=1)


def _highpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    sos = scipy.signal.butter(4, cutoff, "highpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, x, axis=0)


def phantom_brain_model(cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Lagged regression weights (brain sources -> 128 scalp channels), from the brain-only recording."""
    cfg = cfg or load_yaml("injection")
    trial = phantom.load_trial("Brain")
    y = _highpass(average_reference(trial.eeg), trial.sfreq, cfg["phantom_highpass_hz"])
    X = _lag(_highpass(trial.brain_sources, trial.sfreq, cfg["phantom_highpass_hz"]), cfg["phantom_brain_lags"])
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ weights
    stats = {"brain_only_r2_median": float(np.median(1 - residual.var(0) / y.var(0))),
             "noise_floor_rms_uv": float(np.sqrt((residual ** 2).mean(0)).mean() * 1e6)}
    return weights, stats


def phantom_artifact_field(condition: str, weights: np.ndarray, cfg: dict | None = None):
    """Artifact field of one phantom recording: (samples x 64 at 128 Hz, trial, stats)."""
    cfg = cfg or load_yaml("injection")
    trial = phantom.load_trial(condition)
    y = _highpass(average_reference(trial.eeg), trial.sfreq, cfg["phantom_highpass_hz"])
    X = _lag(_highpass(trial.brain_sources, trial.sfreq, cfg["phantom_highpass_hz"]), cfg["phantom_brain_lags"])
    field128 = y - X @ weights
    data64, _ = spline.to_biosemi64(field128, spline.positions(trial.info))
    data64 = resample(average_reference(data64), trial.sfreq, cfg["fs"])
    stats = {"field_rms_uv": float(np.sqrt((field128 ** 2).mean(0)).mean() * 1e6)}
    return data64, trial, stats


def _chunks(rng, n_samples: int, fs: float, duration_range, n: int) -> list[slice]:
    out = []
    for _ in range(n):
        length = int(rng.uniform(*duration_range) * fs)
        start = int(rng.integers(0, max(1, n_samples - length)))
        out.append(slice(start, start + length))
    return out


def _burst_onsets(active: np.ndarray) -> np.ndarray:
    padded = np.concatenate([[False], active, [False]]).astype(int)
    return np.flatnonzero(np.diff(padded) == 1)


def phantom_templates(pool: str, n_per_kind: int = 60, seed: int = 0) -> tuple[list[Template], dict]:
    """Template bank from one phantom pool (set_a_templates or set_b_templates)."""
    if pool == "phantom_check":
        raise ValueError("The phantom_check block is reserved for the §9.5 check")
    cfg = load_yaml("injection")
    rng = np.random.default_rng(seed)
    fs = float(cfg["fs"])
    weights, stats = phantom_brain_model(cfg)
    templates = []
    spec = {"ocular": [("Eyes", "chunks")], "muscle": [("Facial", "bursts"), ("Neck", "chunks")],
            "motion": [("Walking", "chunks")]}
    for kind, sources in spec.items():
        for condition, how in sources:
            data, trial, field_stats = phantom_artifact_field(condition, weights, cfg)
            stats[f"{condition}_field_rms_uv"] = field_stats["field_rms_uv"]
            span = phantom.pool_slice(pool, fs, len(data))
            block = data[span]
            n = n_per_kind // len(sources)
            if how == "bursts":
                activity = phantom.source_activity(trial.ground_truth, trial.sfreq, 0.25, 0.25)
                active = np.repeat(activity["facial"].to_numpy(), int(0.25 * fs))[: len(data)][span]
                onsets = _burst_onsets(active)
                rng.shuffle(onsets)
                cuts = [slice(o, o + int(rng.uniform(*cfg["durations"]["muscle"]) * fs)) for o in onsets[:n]]
            else:
                cuts = _chunks(rng, len(block), fs, cfg["durations"][kind], n)
            for cut in cuts:
                piece = block[cut]
                if len(piece) < int(0.25 * fs):
                    continue
                templates.append(Template(kind, average_reference(piece - piece.mean(0)), fs,
                                          f"phantom:{condition}", pool, {"start_s": (span.start + cut.start) / fs}))
    log.info("phantom %s: %s templates; brain-only R2 %.2f, noise floor %.1f uV; field RMS %s",
             pool, {k: sum(t.kind == k for t in templates) for k in CLASSES}, stats["brain_only_r2_median"],
             stats["noise_floor_rms_uv"], {k: round(v, 1) for k, v in stats.items() if k.endswith("field_rms_uv")})
    return templates, stats


def physiomotion_participants(pool: str) -> list[int]:
    with open(ADR_ROOT / "splits" / "physiomotion_participants.yaml") as f:
        return yaml.safe_load(f)["groups"][pool]


def _pm_to_biosemi64(data: np.ndarray, ch_names, info: mne.Info) -> np.ndarray:
    """PhysioMotion channels -> BioSemi 64: shared names copied, the rest spline-interpolated."""
    names64, pos64 = spline.biosemi64_positions()
    index = {c: i for i, c in enumerate(ch_names)}
    out = np.zeros((len(data), 64))
    shared = [i for i, c in enumerate(names64) if c in index]
    missing = [i for i, c in enumerate(names64) if c not in index]
    out[:, shared] = data[:, [index[names64[i]] for i in shared]]
    if missing:
        pos_from = spline.positions(info, [info["ch_names"].index(c) for c in ch_names])
        M = spline.matrix(pos_from, pos64[missing])
        out[:, missing] = data @ M.T
    return out


def physiomotion_templates(pool: str, broadband_muscle: bool = False, max_per_kind: int = 200,
                           seed: int = 0) -> list[Template]:
    """Single-class annotated events of the PhysioMotion template participants."""
    cfg = load_yaml("injection")
    pcfg = cfg["physiomotion"]
    fs = float(cfg["fs"])
    rng = np.random.default_rng(seed)
    templates = []
    for subject in physiomotion_participants(pool):
        for run in pm.RUNS:
            raw = pm.load_raw(subject, run)
            raw = interpolate_bads(raw, pm.bad_channels(raw))
            names = raw.ch_names
            x = raw.get_data().T
            x = resample(x - x.mean(0), raw.info["sfreq"], fs)
            ann = pm.load_annotations(subject, run)
            filtered = {
                "muscle": x if broadband_muscle else _highpass(x, fs, pcfg["muscle_highpass_hz"]),
                "ocular": scipy.signal.sosfiltfilt(scipy.signal.butter(4, pcfg["ocular_lowpass_hz"], "lowpass", fs=fs, output="sos"),
                                                   _highpass(x, fs, 0.5), axis=0),
                "motion": scipy.signal.sosfiltfilt(scipy.signal.butter(4, pcfg["motion_lowpass_hz"], "lowpass", fs=fs, output="sos"),
                                                   _highpass(x, fs, 0.5), axis=0),
            }
            for label, rows in ann[~ann["label"].isin(pm.REST_LABELS)].groupby("label"):
                classes = pm.label_classes(label)
                if len(classes) != 1:
                    continue  # combined tasks are ambiguous for a single-class template
                kind = next(iter(classes))
                # one event per contiguous annotated span (union over channels)
                spans = sorted(zip(rows["start"], rows["stop"]))
                merged = []
                for a, b in spans:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                for a, b in merged:
                    if not pcfg["min_event_s"] <= b - a <= pcfg["max_event_s"]:
                        continue
                    piece = filtered[kind][int(a * fs):int(b * fs)]
                    data64 = average_reference(_pm_to_biosemi64(piece - piece.mean(0), names, raw.info))
                    templates.append(Template(kind, data64, fs, f"physiomotion:sub-{subject}_run-{run:02d}:{label}",
                                              pool, {"start_s": a}))
    out = []
    for kind in CLASSES:
        bank = [t for t in templates if t.kind == kind]
        if len(bank) > max_per_kind:
            bank = [bank[i] for i in np.sort(rng.choice(len(bank), max_per_kind, replace=False))]
        out += bank
    log.info("physiomotion %s%s: %s templates", pool, " (broadband muscle)" if broadband_muscle else "",
             {k: sum(t.kind == k for t in out) for k in CLASSES})
    return out


def save_bank(templates: list[Template], name: str):
    path = load_paths()["cache"] / "templates" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"t{i}": t.data for i, t in enumerate(templates)},
                        kinds=np.array([t.kind for t in templates]), sources=np.array([t.source for t in templates]),
                        pools=np.array([t.pool for t in templates]), fs=templates[0].fs)
    return path


def load_bank(name: str) -> list[Template]:
    with np.load(load_paths()["cache"] / "templates" / f"{name}.npz") as f:
        return [Template(str(k), f[f"t{i}"], float(f["fs"]), str(s), str(p))
                for i, (k, s, p) in enumerate(zip(f["kinds"], f["sources"], f["pools"]))]
