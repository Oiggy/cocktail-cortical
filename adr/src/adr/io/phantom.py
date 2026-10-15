"""Loader for the phantom EEG dataset (OpenNeuro ds004784, Downey & Ferris 2023).

One 5-min trial per condition (`Brain`, `Eyes`, `Facial`, `Neck`,
`Walking`, `All`), bounded by two 65471 triggers; EEGLAB .set/.fdt at
512 Hz with 128 scalp channels (A1..D32), 128 noise-layer channels (MISC)
and 8 EMG channels. The broadcast source waveforms are in
stimuli/GTdata_croppedToRisingEdge.mat (21 channels: 1-10 brain, 11-12
eyes, 13-16 neck, 17-20 facial, 21 trigger), cropped to the sync rising
edges. The EEG trial is a few samples shorter than the ground truth (clock
skew), so the ground truth is stretched onto the EEG clock
(PROJECT_PLAN.md §3.3, scripts/04_verify_openneuro.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import yaml

from adr.config import ADR_ROOT, load_paths

CONDITIONS = ("Brain", "Eyes", "Facial", "Neck", "Walking", "All")
TRIAL_CODE = 65471
GT_CHANNELS = {
    "brain": slice(0, 10),
    "eyes": slice(10, 12),
    "neck": slice(12, 16),
    "facial": slice(16, 20),
    "trigger": slice(20, 21),
}
# Which broadcast artifact sources are switched on in each condition.
# Motion is mechanical, so it has no ground-truth channel.
ARTIFACT_SOURCES = {
    "Brain": (),
    "Eyes": ("eyes",),
    "Facial": ("facial",),
    "Neck": ("neck",),
    "Walking": ("motion",),
    "All": ("eyes", "facial", "neck", "motion"),
}


@dataclass(frozen=True)
class PhantomTrial:
    condition: str
    eeg: np.ndarray           # samples x 128 scalp channels, volts, unreferenced (CMS)
    noise: np.ndarray         # samples x 128 noise-layer channels
    emg: np.ndarray           # samples x 8 EMG reference channels
    sfreq: float
    ch_names: tuple[str, ...]
    info: mne.Info            # scalp channels only, with positions from electrodes.tsv
    ground_truth: np.ndarray  # samples x 21, stretched onto the EEG clock

    @property
    def brain_sources(self) -> np.ndarray:
        return self.ground_truth[:, GT_CHANNELS["brain"]]


def scalp_channels(ch_names) -> list[str]:
    """The 128 scalp electrodes: not noise layer (`N-...`), not EMG (`EXG...`)."""
    return [ch for ch in ch_names if not ch.startswith(("N-", "EXG"))]


def root(path: Path | None = None) -> Path:
    return path or load_paths()["phantom"]


def load_ground_truth(path: Path | None = None) -> np.ndarray:
    """Broadcast waveforms as recorded (153600 x 21 at 512 Hz)."""
    return scipy.io.loadmat(root(path) / "stimuli" / "GTdata_croppedToRisingEdge.mat")["GTdata"]


def trial_bounds(condition: str, path: Path | None = None) -> tuple[int, int]:
    """(first, second) 65471 trigger sample of the condition's trial."""
    events = pd.read_csv(root(path) / "sub-001" / "eeg" / f"sub-001_task-{condition}_events.tsv", sep="\t")
    marks = events.loc[events["value"] == TRIAL_CODE, "sample"].to_numpy()
    if len(marks) != 2:
        raise ValueError(f"{condition}: expected 2 trial triggers, found {len(marks)}")
    return int(marks[0]), int(marks[1])


def load_trial(condition: str, path: Path | None = None) -> PhantomTrial:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown phantom condition: {condition!r}")
    eeg_dir = root(path) / "sub-001" / "eeg"
    raw = mne.io.read_raw_eeglab(eeg_dir / f"sub-001_task-{condition}_eeg.set", preload=True, verbose="error")
    s0, s1 = trial_bounds(condition, path)
    trial = raw.copy().crop(s0 / raw.info["sfreq"], (s1 - 1) / raw.info["sfreq"])
    gt = scipy.signal.resample(load_ground_truth(path), s1 - s0, axis=0)

    # The EEGLAB reader types all 256 electrodes as EEG; select by name.
    scalp = trial.copy().pick(scalp_channels(trial.ch_names))
    electrodes = pd.read_csv(eeg_dir / f"sub-001_task-{condition}_electrodes.tsv", sep="\t").set_index("name")
    # electrodes.tsv is in mm, CTF/EEGLAB axes (x nose, y left, z up); MNE's
    # head frame is x right, y nose, z up.
    positions = {
        ch: np.array([-electrodes.loc[ch, "y"], electrodes.loc[ch, "x"], electrodes.loc[ch, "z"]]) / 1000
        for ch in scalp.ch_names
    }
    scalp.set_montage(mne.channels.make_dig_montage(positions, coord_frame="head"))
    return PhantomTrial(
        condition=condition,
        eeg=scalp.get_data().T,
        noise=trial.copy().pick([ch for ch in trial.ch_names if ch.startswith("N-")]).get_data().T,
        emg=trial.copy().pick([ch for ch in trial.ch_names if ch.startswith("EXG")]).get_data().T,
        sfreq=raw.info["sfreq"],
        ch_names=tuple(scalp.ch_names),
        info=scalp.info,
        ground_truth=gt,
    )


def load_split() -> dict:
    with open(ADR_ROOT / "splits" / "phantom.yaml") as f:
        return yaml.safe_load(f)


def pool_slice(pool: str, sfreq: float, n_samples: int) -> slice:
    """Sample range of a phantom pool within a trial (splits/phantom.yaml)."""
    start, stop = load_split()["pools"][pool]
    return slice(int(round(start * sfreq)), min(int(round(stop * sfreq)), n_samples))
