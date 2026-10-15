"""Loader for Das et al. Dataset I v2.0 (KU Leuven AAD dataset, Zenodo 4004271).

Each S<n>.mat holds a MATLAB cell array `trials` of 20 structs. The EEG is
already 0.5 Hz high-passed, downsampled to 128 Hz and MWF-cleaned (Zenodo
record description; the README.txt.txt inside the download omits the MWF
sentence).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

from adr.config import load_paths

N_SUBJECTS = 16


@dataclass(frozen=True)
class Trial:
    subject: str
    trial_id: int
    eeg: np.ndarray           # samples x channels, float64
    sfreq: float
    ch_names: tuple[str, ...]
    attended_ear: str         # 'L' or 'R'
    condition: str            # 'hrtf' or 'dry' (lowercase in the files)
    experiment: int           # 1, 2 or 3
    part: int
    attended_track: int       # 1 or 2
    repetition: bool
    stimuli: tuple[str, str]  # (left-ear, right-ear) wav names as presented

    @property
    def duration(self) -> float:
        return self.eeg.shape[0] / self.sfreq

    @property
    def dry_stimuli(self) -> tuple[str, str]:
        """Dry wav names for envelope extraction (PROJECT_PLAN.md §3.1)."""
        return tuple(s.replace("_hrtf.wav", "_dry.wav") for s in self.stimuli)


def subject_path(subject: int | str, root: Path | None = None) -> Path:
    name = subject if isinstance(subject, str) else f"S{subject}"
    return (root or load_paths()["dataset1"]) / f"{name}.mat"


def available_subjects(root: Path | None = None) -> list[str]:
    root = root or load_paths()["dataset1"]
    present = [f"S{i}" for i in range(1, N_SUBJECTS + 1) if (root / f"S{i}.mat").exists()]
    return present


def load_subject(subject: int | str, root: Path | None = None) -> list[Trial]:
    mat = scipy.io.loadmat(subject_path(subject, root), simplify_cells=True)
    trials = []
    for t in mat["trials"]:
        header = t["FileHeader"]
        trials.append(Trial(
            subject=t["subject"],
            trial_id=int(t["TrialID"]),
            eeg=np.asarray(t["RawData"]["EegData"], dtype=np.float64),
            sfreq=float(header["SampleRate"]),
            ch_names=tuple(ch["Label"] for ch in header["Channels"]),
            attended_ear=t["attended_ear"],
            condition=t["condition"].lower(),
            experiment=int(t["experiment"]),
            part=int(t["part"]),
            attended_track=int(t["attended_track"]),
            repetition=bool(t["repetition"]),
            stimuli=tuple(t["stimuli"]),
        ))
    return trials
