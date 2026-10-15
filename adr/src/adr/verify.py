"""Data checks behind the VERIFY items in PROJECT_PLAN.md §3.1-3.3.

Used by scripts/02_verify_dataset1.py, scripts/04_verify_openneuro.py and
notebooks/01_data_checks.ipynb. Each function returns a JSON-serialisable
report and logs a one-line summary per item.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
from collections import defaultdict

import mne
import numpy as np
import pandas as pd
import scipy.io
import scipy.io.wavfile
import scipy.signal
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from adr.config import load_paths, seed_everything
from adr.io.dataset1 import available_subjects, load_subject
from adr.io.phantom import scalp_channels

log = logging.getLogger("adr.verify")

PHANTOM_TASKS = ["Brain", "Eyes", "Facial", "Neck", "Walking", "All"]
PHANTOM_TRIAL_CODE = 65471


def distance_correlation(eeg: np.ndarray, positions: np.ndarray) -> float:
    """Spearman rho between inter-electrode distance and channel correlation.

    Signed, not absolute: after average referencing, distant channels are
    anti-correlated, and taking |r| would count that as similarity.
    """
    eeg = eeg - eeg.mean(axis=1, keepdims=True)  # average reference
    corr = np.corrcoef(eeg.T)
    iu = np.triu_indices(len(positions), k=1)
    return spearmanr(pdist(positions), corr[iu]).statistic



def wav_seconds(path) -> float:
    sfreq, data = scipy.io.wavfile.read(path, mmap=True)
    return data.shape[0] / sfreq


def dataset1(subjects=None, n_null: int = 200) -> dict:
    """Channel order, Cz/C5 index, minutes per condition, stimulus lengths (§3.1, §9.1)."""
    paths = load_paths()
    rng = seed_everything()
    montage = mne.channels.make_standard_montage("biosemi64")
    standard = tuple(montage.ch_names)
    montage_pos = montage.get_positions()["ch_pos"]
    stim_dir = paths["dataset1"] / "stimuli"

    subjects = subjects or available_subjects()
    log.info("subjects present: %s (%d of 16)", subjects, len(subjects))

    report = {"subjects": subjects, "per_subject": {}}
    all_labels = set()
    stim_problems = []
    for subject in subjects:
        trials = load_subject(subject)
        labels = {t.ch_names for t in trials}
        all_labels |= labels
        ch_names = trials[0].ch_names
        positions = np.array([montage_pos[ch] for ch in ch_names])

        # Correlation vs distance under the stored labels, and under 200
        # random relabellings as a null.
        eeg = np.concatenate([t.eeg for t in trials if not t.repetition])
        rho = distance_correlation(eeg, positions)
        null = [distance_correlation(eeg, positions[rng.permutation(64)]) for _ in range(n_null)]

        minutes = defaultdict(float)
        for t in trials:
            key = f"{t.condition}/{'rep' if t.repetition else 'norep'}/attend-{t.attended_ear}"
            minutes[key] += t.duration / 60
            for name in t.dry_stimuli:
                audio_s = wav_seconds(stim_dir / name)
                if audio_s < t.duration:
                    stim_problems.append((subject, t.trial_id, name, audio_s, t.duration))

        hrtf_norep = sum(v for k, v in minutes.items() if k.startswith("hrtf/norep"))
        hrtf_all = sum(v for k, v in minutes.items() if k.startswith("hrtf/"))
        report["per_subject"][subject] = {
            "n_trials": len(trials),
            "sfreq": sorted({t.sfreq for t in trials}),
            "labels_consistent_across_trials": len(labels) == 1,
            "labels_match_biosemi64": ch_names == standard,
            "cz_index_0based": ch_names.index("Cz"),
            "c5_index_0based": ch_names.index("C5"),
            "distance_rho": rho,
            "distance_rho_null_min": float(np.min(null)),
            "minutes": dict(sorted(minutes.items())),
            "minutes_hrtf_norep": hrtf_norep,
            "minutes_hrtf_all": hrtf_all,
            "minutes_total": sum(minutes.values()),
            "trial_seconds": {t.trial_id: round(t.duration, 2) for t in trials},
        }
        s = report["per_subject"][subject]
        log.info(
            "%s: %d trials, sfreq %s, labels=biosemi64 %s, Cz idx %d, C5 idx %d, "
            "dist-rho %.2f (null min %.2f), HRTF no-rep %.1f min, HRTF all %.1f min, total %.1f min",
            subject, s["n_trials"], s["sfreq"], s["labels_match_biosemi64"], s["cz_index_0based"],
            s["c5_index_0based"], rho, s["distance_rho_null_min"], hrtf_norep, hrtf_all, s["minutes_total"],
        )

    report["labels_identical_across_subjects"] = len(all_labels) == 1
    report["ch_names"] = list(next(iter(all_labels)))
    report["stimulus_shorter_than_eeg"] = stim_problems
    log.info("labels identical across subjects: %s", report["labels_identical_across_subjects"])
    log.info("trials where a dry stimulus is shorter than the EEG: %d", len(stim_problems))

    return report


def physiomotion(root=None) -> dict:
    """Licence, channels, annotation labels and rest-block cleanliness (§3.2)."""
    root = root or load_paths()["physiomotion"]
    description = json.loads((root / "dataset_description.json").read_text())
    channels = pd.read_csv(root / "sub-1/eeg/sub-1_task-artifact_run-01_channels.tsv", sep="\t")
    names = [n.lstrip("﻿") for n in channels["name"]]
    eeg_json = json.loads((root / "sub-1/eeg/sub-1_task-artifact_run-01_eeg.json").read_text())
    biosemi = mne.channels.make_standard_montage("biosemi64").ch_names

    labels, label_channels = collections.Counter(), set()
    base, artifacts = collections.defaultdict(list), collections.defaultdict(list)
    files = sorted((root / "derivatives/Manual_Annotations").glob("*.csv"))
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                interval = (float(row["start_time"]), float(row["stop_time"]))
                labels[row["label"]] += 1
                label_channels.add(row["channel"])
                (base if row["label"].endswith("_base") else artifacts)[f.name].append(interval)
    overlaps = sum(
        s < b and e > a
        for name in base for a, b in base[name] for s, e in artifacts[name]
    )
    bipolar = sorted(c for c in label_channels if "-" in c)
    electrodes = sorted({e for pair in bipolar for e in pair.split("-")})
    return {
        "licence": description.get("License"),
        "authors": description.get("Authors"),
        "n_subjects": len(list(root.glob("sub-*"))),
        "sfreq": eeg_json["SamplingFrequency"],
        "n_channels": len(names),
        "channels": names,
        "shared_with_biosemi64": [c for c in biosemi if c in names],
        "physiomotion_only": [c for c in names if c not in biosemi],
        "biosemi64_only": [c for c in biosemi if c not in names],
        "n_annotation_files": len(files),
        "labels": dict(labels.most_common()),
        "annotation_channels_nonbipolar": sorted(label_channels - set(bipolar)),
        "annotation_bipolar_pairs": bipolar,
        "annotation_electrodes": electrodes,
        "n_base_blocks": sum(map(len, base.values())),
        "artifact_annotations_inside_base_blocks": overlaps,
    }


def r2_per_channel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return 1 - ((Y - X @ beta) ** 2).sum(0) / (Y ** 2).sum(0)


def phantom(root=None) -> dict:
    """Trials per condition, ground truth and alignment quality (§3.3)."""
    root = root or load_paths()["phantom"]
    gt = scipy.io.loadmat(root / "stimuli/GTdata_croppedToRisingEdge.mat")["GTdata"]
    electrodes = pd.read_csv(root / "sub-001/eeg/sub-001_task-Brain_electrodes.tsv", sep="\t")
    report = {"gt_shape": list(gt.shape), "n_electrode_rows": len(electrodes), "tasks": {}}
    sos = scipy.signal.butter(4, [1, 40], "bandpass", fs=512, output="sos")
    for task in PHANTOM_TASKS:
        raw = mne.io.read_raw_eeglab(root / f"sub-001/eeg/sub-001_task-{task}_eeg.set", preload=True, verbose="error")
        events = pd.read_csv(root / f"sub-001/eeg/sub-001_task-{task}_events.tsv", sep="\t")
        marks = events.loc[events["value"] == PHANTOM_TRIAL_CODE, "sample"].to_numpy()
        s0, s1 = int(marks[0]), int(marks[1])
        # The EEGLAB reader types all 256 electrodes as EEG; keep the 128
        # scalp channels (the noise layer is named N-...).
        eeg = raw.get_data(picks=scalp_channels(raw.ch_names))[:, s0:s1].T
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        Y = scipy.signal.sosfiltfilt(sos, eeg, axis=0)[10:-10]
        # The EEG trial is a few samples shorter than the ground truth (clock
        # skew); stretch the ground truth onto the EEG clock.
        X = scipy.signal.sosfiltfilt(sos, scipy.signal.resample(gt[:, :10], s1 - s0, axis=0), axis=0)[10:-10]
        by_lag = {lag: float(np.median(r2_per_channel(np.roll(X, lag, 0), Y))) for lag in range(-3, 4)}
        types = {
            "scalp": len(scalp_channels(raw.ch_names)),
            "noise_layer": sum(ch.startswith("N-") for ch in raw.ch_names),
            "emg": sum(ch.startswith("EXG") for ch in raw.ch_names),
        }
        report["tasks"][task] = {
            "sfreq": raw.info["sfreq"],
            "channels": types,
            "n_trials": len(marks) // 2,
            "trial_samples": s1 - s0,
            "trial_seconds": (s1 - s0) / raw.info["sfreq"],
            "median_r2_brain_gt_by_lag": by_lag,
        }
        log.info("phantom %-7s %d trial(s), %.1f s at %.0f Hz, median R2 (brain GT -> scalp) lag0 %.3f",
                 task, len(marks) // 2, (s1 - s0) / raw.info["sfreq"], raw.info["sfreq"], by_lag[0])
    return report
