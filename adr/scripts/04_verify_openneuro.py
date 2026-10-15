"""Check the PhysioMotion (§3.2) and phantom (§3.3) VERIFY items in PROJECT_PLAN.md.

PhysioMotion: licence, channel names and the overlap with BioSemi 64,
sampling rate, the annotation label list and channels, and whether any
artifact annotation falls inside a rest (`*_base`) block.

Phantom: sampling rate, trials per condition (trigger 65471 start/stop),
ground-truth waveform file, electrode positions, and how well the broadcast
brain sources explain the scalp EEG (alignment quality), after stretching
the ground truth onto the EEG clock and average referencing.

Writes results/verify/openneuro_verify.json and prints a summary.

    python scripts/04_verify_openneuro.py
"""
from __future__ import annotations

import collections
import csv
import json
import logging

import mne
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal

from adr.config import load_paths, package_versions

log = logging.getLogger("verify_openneuro")

PHANTOM_TASKS = ["Brain", "Eyes", "Facial", "Neck", "Walking", "All"]
PHANTOM_TRIAL_CODE = 65471


def verify_physiomotion(root) -> dict:
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


def verify_phantom(root) -> dict:
    gt = scipy.io.loadmat(root / "stimuli/GTdata_croppedToRisingEdge.mat")["GTdata"]
    electrodes = pd.read_csv(root / "sub-001/eeg/sub-001_task-Brain_electrodes.tsv", sep="\t")
    report = {"gt_shape": list(gt.shape), "n_electrode_rows": len(electrodes), "tasks": {}}
    sos = scipy.signal.butter(4, [1, 40], "bandpass", fs=512, output="sos")
    for task in PHANTOM_TASKS:
        raw = mne.io.read_raw_eeglab(root / f"sub-001/eeg/sub-001_task-{task}_eeg.set", preload=True, verbose="error")
        events = pd.read_csv(root / f"sub-001/eeg/sub-001_task-{task}_events.tsv", sep="\t")
        marks = events.loc[events["value"] == PHANTOM_TRIAL_CODE, "sample"].to_numpy()
        s0, s1 = int(marks[0]), int(marks[1])
        eeg = raw.get_data(picks="eeg")[:, s0:s1].T
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        Y = scipy.signal.sosfiltfilt(sos, eeg, axis=0)[10:-10]
        # The EEG trial is a few samples shorter than the ground truth (clock
        # skew); stretch the ground truth onto the EEG clock.
        X = scipy.signal.sosfiltfilt(sos, scipy.signal.resample(gt[:, :10], s1 - s0, axis=0), axis=0)[10:-10]
        by_lag = {lag: float(np.median(r2_per_channel(np.roll(X, lag, 0), Y))) for lag in range(-3, 4)}
        types = collections.Counter(raw.get_channel_types())
        report["tasks"][task] = {
            "sfreq": raw.info["sfreq"],
            "channel_types": dict(types),
            "n_trials": len(marks) // 2,
            "trial_samples": s1 - s0,
            "trial_seconds": (s1 - s0) / raw.info["sfreq"],
            "median_r2_brain_gt_by_lag": by_lag,
        }
        log.info("phantom %-7s %d trial(s), %.1f s at %.0f Hz, median R2 (brain GT -> scalp) lag0 %.3f",
                 task, len(marks) // 2, (s1 - s0) / raw.info["sfreq"], raw.info["sfreq"], by_lag[0])
    return report


def main() -> None:
    paths = load_paths()
    out_dir = paths["results"] / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "openneuro_verify.log", mode="w")],
    )
    log.info("package versions: %s", package_versions())

    pm = verify_physiomotion(paths["physiomotion"])
    log.info("physiomotion: %d subjects, %d ch at %.0f Hz, licence %s, %d shared with BioSemi 64",
             pm["n_subjects"], pm["n_channels"], pm["sfreq"], pm["licence"], len(pm["shared_with_biosemi64"]))
    log.info("physiomotion labels: %s", pm["labels"])
    log.info("physiomotion annotations on %d bipolar pairs over %d electrodes, plus %s; "
             "%d artifact annotations inside %d rest blocks",
             len(pm["annotation_bipolar_pairs"]), len(pm["annotation_electrodes"]),
             pm["annotation_channels_nonbipolar"], pm["artifact_annotations_inside_base_blocks"], pm["n_base_blocks"])

    phantom = verify_phantom(paths["phantom"])
    with open(out_dir / "openneuro_verify.json", "w") as f:
        json.dump({"physiomotion": pm, "phantom": phantom}, f, indent=1)


if __name__ == "__main__":
    main()
