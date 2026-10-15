"""Loader for Das et al. Dataset I v2.0 (KU Leuven AAD dataset, Zenodo 4004271).

Each S<n>.mat holds a MATLAB cell array `trials` of 20 structs. The EEG is
already 0.5 Hz high-passed, downsampled to 128 Hz and MWF-cleaned (Zenodo
record description; the README.txt.txt inside the download omits the MWF
sentence).
"""
from __future__ import annotations

import re
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


_STIMULUS_NAME = re.compile(r"^(?:rep_)?part(\d+)_track(\d+)_(?:hrtf|dry)\.wav$")


def stimulus_key(wav_name: str) -> tuple[int, int]:
    """(part, track) of a stimulus wav, ignoring `rep_` and the hrtf/dry suffix.

    `rep_partP_trackT` is the first ~2 min of `partP_trackT` (identical audio),
    and the hrtf and dry versions carry the same speech, so all four names
    map to the same key.
    """
    match = _STIMULUS_NAME.match(wav_name)
    if not match:
        raise ValueError(f"Unexpected stimulus name: {wav_name!r}")
    return int(match[1]), int(match[2])


def presentation_groups(trials: list[Trial]) -> list[int]:
    """Cross-validation group label per trial (PROJECT_PLAN.md §9.4, §14).

    Trials are grouped by the original presentation, and any trials that
    share a stimulus are merged into one group (connected components over
    shared stimulus keys). In Dataset I every trial plays both tracks of one
    story part, experiment 2 replays experiment 1's pairs with attention
    switched, and the repetitions are the openings of those parts, so the
    groups end up being the story parts (labelled 1-4). Use with GroupKFold or
    leave-one-group-out; never split a group between train and test.
    """
    parent = list(range(len(trials)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_trial_with = {}
    for i, trial in enumerate(trials):
        for key in map(stimulus_key, trial.stimuli):
            if key in first_trial_with:
                parent[find(i)] = find(first_trial_with[key])
            else:
                first_trial_with[key] = i
    # Label each group by the lowest story part it contains, so labels mean
    # the same thing in every subject.
    roots = [find(i) for i in range(len(trials))]
    labels = {}
    for key, i in first_trial_with.items():
        root = find(i)
        labels[root] = min(labels.get(root, key[0]), key[0])
    return [labels[root] for root in roots]


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
