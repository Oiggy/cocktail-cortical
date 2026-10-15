"""Loader for PhysioMotion Artifact (OpenNeuro ds006386, raw BIDS version).

30 subjects x 6 runs of continuous, unfiltered EEG (59 channels, 10-10
names, 1000 Hz, volts). Manual annotations live in
derivatives/Manual_Annotations/sub<N>_run<RR>.csv with columns
channel, start_time, stop_time, label (seconds from recording start). They
are on the 34 bipolar pairs of the preprocessed version (e.g. `Fp1-F7`),
over 19 10-20 electrodes, plus a channel `ALL`. Rest blocks are labelled
`close_base`/`open_base` and contain no artifact annotations
(PROJECT_PLAN.md §3.2, scripts/04_verify_openneuro.py).
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.signal

from adr.config import load_paths

SUBJECTS = tuple(range(1, 31))
RUNS = tuple(range(1, 7))
CLASSES = ("ocular", "muscle", "motion")  # `pop` is simulated, `clean` is rest

# PhysioMotion label atoms -> routing class (PROJECT_PLAN.md §6). Combined
# labels (e.g. `blink_hor_headm`) are concatenations of these atoms.
ATOM_CLASS = {
    "blink": "ocular",
    "hor_eyem": "ocular",
    "ver_eyem": "ocular",
    "hor_headm": "motion",
    "ver_headm": "motion",
    "tongue": "muscle",
    "chew": "muscle",
    "swallow": "muscle",
    "eyebrow": "muscle",
}
REST_LABELS = ("close_base", "open_base")
N_BIPOLAR_PAIRS = 34  # annotated montage; `ALL` stands for all of them


def root(path: Path | None = None) -> Path:
    return path or load_paths()["physiomotion"]


def eeg_path(subject: int, run: int, path: Path | None = None) -> Path:
    return root(path) / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-artifact_run-{run:02d}_eeg.edf"


def load_raw(subject: int, run: int, path: Path | None = None, preload: bool = True) -> mne.io.BaseRaw:
    """The raw recording, unfiltered, in volts, with a standard 10-05 montage."""
    raw = mne.io.read_raw_edf(eeg_path(subject, run, path), preload=preload, verbose="error")
    raw.set_montage("standard_1005", on_missing="raise", verbose="error")
    return raw


def load_annotations(subject: int, run: int, path: Path | None = None) -> pd.DataFrame:
    """Manual annotations with columns channel, start, stop, label."""
    csv = root(path) / "derivatives" / "Manual_Annotations" / f"sub{subject}_run{run:02d}.csv"
    frame = pd.read_csv(csv, encoding="utf-8-sig")
    return frame.rename(columns={"start_time": "start", "stop_time": "stop"})


def label_classes(label: str) -> frozenset[str]:
    """Routing classes of one annotation label; empty for rest blocks.

    >>> sorted(label_classes("blink_hor_headm"))
    ['motion', 'ocular']
    """
    if label in REST_LABELS:
        return frozenset()
    classes, rest = set(), label
    while rest:
        atom = next((a for a in sorted(ATOM_CLASS, key=len, reverse=True) if rest.startswith(a)), None)
        if atom is None:
            raise ValueError(f"Unknown PhysioMotion label: {label!r}")
        classes.add(ATOM_CLASS[atom])
        rest = rest[len(atom):].lstrip("_")
    return frozenset(classes)


def annotation_pairs(channel: str) -> int:
    """How many bipolar pairs an annotation covers (`ALL` covers all of them)."""
    return N_BIPOLAR_PAIRS if channel == "ALL" else 1


def window_labels(
    annotations: pd.DataFrame,
    duration: float,
    window: float,
    step: float,
    min_pairs: int = 1,
) -> pd.DataFrame:
    """Per-window labels from the bipolar annotations (PROJECT_PLAN.md §6).

    A window carries a class if at least `min_pairs` annotated bipolar
    pairs carry that class during the window (1 = the OR rule; 2 = the
    stricter rule). `rest` marks windows lying entirely inside a rest
    block; such windows carry no artifact (verified), so they are the
    `clean` examples. Windows that are neither rest nor annotated are
    `unlabelled` - task periods with no annotation are not assumed clean.

    Returns one row per window: start, stop, one boolean column per class,
    `rest`, and `unlabelled`.
    """
    starts = np.arange(0.0, duration - window + 1e-9, step)
    stops = starts + window
    counts = {c: np.zeros(len(starts), dtype=int) for c in CLASSES}
    rest = np.zeros(len(starts), dtype=bool)
    for row in annotations.itertuples(index=False):
        overlap = (starts < row.stop) & (stops > row.start)
        if row.label in REST_LABELS:
            rest |= (starts >= row.start) & (stops <= row.stop)
            continue
        for cls in label_classes(row.label):
            counts[cls] += overlap * annotation_pairs(row.channel)
    out = pd.DataFrame({"start": starts, "stop": stops})
    for cls in CLASSES:
        out[cls] = counts[cls] >= min_pairs
    out["rest"] = rest & ~out[list(CLASSES)].any(axis=1)
    out["unlabelled"] = ~out["rest"] & ~out[list(CLASSES)].any(axis=1)
    return out


def bad_channels(raw: mne.io.BaseRaw, low: float = 0.2, high: float = 5.0) -> list[str]:
    """Flat or persistently noisy channels, robust to intermittent artifacts.

    On a 0.5-40 Hz band-passed copy (the raw data are DC-dominated), each
    channel's standard deviation is computed in 1 s windows and summarised
    by its median across windows, so blinks, movements and muscle bursts do
    not count. A channel is bad if that median is below `low` or above
    `high` times the median across channels. (An earlier version used the
    plain standard deviation, which flagged healthy frontal channels in runs
    without rest blocks, or with blinks in the eyes-open rest.)
    """
    fs = raw.info["sfreq"]
    data = raw.get_data(picks="eeg")
    sos = scipy.signal.butter(4, [0.5, 40], "bandpass", fs=fs, output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, data, axis=1)
    n = int(fs)
    k = filtered.shape[1] // n
    sd = np.median(filtered[:, : k * n].reshape(len(filtered), k, n).std(axis=2), axis=1)
    median = np.median(sd)
    names = np.array(raw.copy().pick("eeg").ch_names)
    return list(names[(sd < low * median) | (sd > high * median)])
