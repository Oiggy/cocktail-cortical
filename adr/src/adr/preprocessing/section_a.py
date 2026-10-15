"""Section A inputs from Dataset I v2.0 (PROJECT_PLAN.md §9.1).

Per subject: the HRTF trials (repetitions included), Das-preprocessed EEG
(32 Hz, Cz reference, 63 channels) and the attended dry-stimulus envelope
truncated to the EEG length, both z-scored over the subject's trials, plus
the story-part CV group (`presentation_groups`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adr.config import load_paths, load_yaml
from adr.envelope.das import wav_envelope
from adr.io.dataset1 import Trial, load_subject, presentation_groups
from adr.preprocessing.das import CZ_INDEX, preprocess_eeg


@dataclass(frozen=True)
class SectionATrial:
    subject: str
    trial_id: int
    attended_ear: str      # 'L' or 'R'
    group: int             # story part (CV group)
    eeg: np.ndarray        # samples x 63, z-scored over the subject
    envelope: np.ndarray   # samples, attended envelope, z-scored over the subject


def channel_names(trial: Trial) -> list[str]:
    """The 63 channel names after dropping Cz."""
    names = list(trial.ch_names)
    del names[CZ_INDEX]
    return names


def attended_stimulus(trial: Trial) -> str:
    """Dry wav of the attended track (`stimuli` is (left ear, right ear))."""
    left, right = trial.dry_stimuli
    return left if trial.attended_ear == "L" else right


def prepare_subject(subject: str) -> tuple[list[SectionATrial], list[str]]:
    paths = load_paths()
    condition = load_yaml("section_a")["condition"]
    trials = load_subject(subject)
    groups = presentation_groups(trials)
    kept = [(t, g) for t, g in zip(trials, groups) if t.condition == condition]

    eegs, envs = [], []
    for trial, _ in kept:
        eeg = preprocess_eeg(trial.eeg)
        env = wav_envelope(paths["dataset1"] / "stimuli" / attended_stimulus(trial), paths["cache"] / "envelopes")
        if len(env) < len(eeg):
            raise ValueError(f"{subject} trial {trial.trial_id}: envelope shorter than EEG")
        eegs.append(eeg)
        envs.append(env[: len(eeg)])

    all_eeg, all_env = np.concatenate(eegs), np.concatenate(envs)
    eeg_mean, eeg_sd = all_eeg.mean(0), all_eeg.std(0)
    env_mean, env_sd = all_env.mean(), all_env.std()
    out = [
        SectionATrial(
            subject=subject,
            trial_id=trial.trial_id,
            attended_ear=trial.attended_ear,
            group=group,
            eeg=(eeg - eeg_mean) / eeg_sd,
            envelope=(env - env_mean) / env_sd,
        )
        for (trial, group), eeg, env in zip(kept, eegs, envs)
    ]
    return out, channel_names(trials[0])
