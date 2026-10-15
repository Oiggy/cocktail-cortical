"""Detector training data from PhysioMotion (PROJECT_PLAN.md §6-7).

Per run and detector rate: window features and labels (ocular, muscle,
motion multi-label; `rest` = clean; `unlabelled` excluded from training),
plus pop examples from a copy of the run with simulated pops injected
inside the rest blocks (only windows that contain a pop onset are kept).
Results are cached under <cache>/features/<rate>Hz/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.signal

from adr.config import load_paths, load_project
from adr.detection import features, prep
from adr.io import physiomotion as pm
from adr.preprocessing.channels import detector_channels, interpolate_bads
from adr.synth import pops as pop_sim

LABEL_COLUMNS = ("ocular", "muscle", "motion", "rest", "unlabelled")


def cache_path(subject: int, run: int, fs: float):
    return load_paths()["cache"] / "features" / f"{int(fs)}Hz" / f"sub-{subject}_run-{run:02d}.npz"


def rest_mask(starts_s: np.ndarray, window_s: float, annotations: pd.DataFrame,
              labels=pm.REST_LABELS) -> np.ndarray:
    rest = annotations[annotations["label"].isin(labels)]
    mask = np.zeros(len(starts_s), dtype=bool)
    for row in rest.itertuples(index=False):
        mask |= (starts_s >= row.start) & (starts_s + window_s <= row.stop)
    return mask


def run_features(subject: int, run: int, fs: float, seed: int | None = None, use_cache: bool = True) -> dict:
    path = cache_path(subject, run, fs)
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as f:
            return dict(f)
    cfg = features.detection_config()
    raw = pm.load_raw(subject, run)
    annotations = pm.load_annotations(subject, run)
    bads = pm.bad_channels(raw)
    raw = interpolate_bads(raw, bads)
    names = raw.ch_names
    data = prep.to_rate(raw.get_data().T, raw.info["sfreq"], fs)

    clean_data = prep.finish(data, fs, names, cfg["highpass_hz"])
    X = features.compute(clean_data, fs, detector_channels(), cfg)
    labels = pm.window_labels(annotations, len(data) / fs, cfg["window_s"], cfg["step_s"], cfg["min_pairs"])
    if len(labels) != len(X):
        raise RuntimeError(f"sub-{subject} run {run}: {len(labels)} labels vs {len(X)} windows")

    # Pop augmentation inside the rest blocks.
    rng = np.random.default_rng(seed if seed is not None else load_project()["seed"] + 1000 * subject + run)
    starts_s = labels["start"].to_numpy()
    in_rest = rest_mask(starts_s, cfg["window_s"], annotations)
    rest_rows = annotations[annotations["label"].isin(pm.REST_LABELS)]
    pops = []
    for row in rest_rows.itertuples(index=False):
        n_pops = int((row.stop - row.start) // cfg["pops"]["every_s"])
        lo, hi = int(row.start * fs), int(row.stop * fs)
        for p in pop_sim.draw(rng, hi - lo, fs, names, n_pops, cfg["pops"]["amplitude_uv"], cfg["pops"]["tau_s"]):
            pops.append(pop_sim.Pop(p.onset + lo, p.channel, p.amplitude, p.tau))
    X_pop = np.zeros((0, X.shape[1]), np.float32)
    if pops:
        popped = prep.finish(pop_sim.inject(data, fs, pops), fs, names, cfg["highpass_hz"])
        Xp = features.compute(popped, fs, detector_channels(), cfg)
        onsets = np.array([p.onset for p in pops]) / fs
        contains = ((onsets[None, :] >= starts_s[:, None]) &
                    (onsets[None, :] < starts_s[:, None] + cfg["window_s"])).any(axis=1)
        X_pop = Xp[contains & in_rest]

    # Frontal peak-to-peak (0.5-10 Hz, mean of Fp1/Fp2) per window: the eyes-open
    # rest blocks contain unannotated blinks, so this lets the clean-class
    # definition be chosen at training time without re-extracting.
    sos = scipy.signal.butter(4, [0.5, 10], "bandpass", fs=fs, output="sos")
    chs = detector_channels()
    fp = scipy.signal.sosfiltfilt(sos, clean_data[:, [chs.index("Fp1"), chs.index("Fp2")]].mean(axis=1))
    fp_windows = features.windows(fp[:, None], fs, cfg["window_s"], cfg["step_s"])[:, :, 0]

    out = {
        "X": X,
        "rest_closed": rest_mask(starts_s, cfg["window_s"], annotations, ("close_base",)) & labels["rest"].to_numpy(),
        "rest_open": rest_mask(starts_s, cfg["window_s"], annotations, ("open_base",)) & labels["rest"].to_numpy(),
        "fp_ptp_uv": np.ptp(fp_windows, axis=1) * 1e6,
        "X_pop": X_pop,
        "start_s": starts_s,
        **{c: labels[c].to_numpy() for c in LABEL_COLUMNS},
        "bads": np.array(bads, dtype=str),
        "n_pops": np.array(len(pops)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out
