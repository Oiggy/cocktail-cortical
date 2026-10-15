"""Check the Dataset I v2.0 VERIFY items in PROJECT_PLAN.md §3.1 and §9.1.

1. Channel order: labels stored in each trial's FileHeader, compared with
   the standard BioSemi 64 order (MNE's biosemi64 montage), and an
   empirical check that channel correlations fall off with electrode
   distance under those labels (versus shuffled labels).
2. Index of Cz (and C5, the base-TRF channel of §9.2).
3. Minutes of EEG per subject by condition / repetition / attended ear,
   for the "38 min per subject" claim.
4. Stimulus lengths versus EEG lengths (audio must be truncated to EEG).

Writes results/verify/dataset1_verify.json and prints a summary.

    python scripts/02_verify_dataset1.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import mne
import numpy as np
import scipy.io.wavfile
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from adr.config import load_paths, package_versions, seed_everything
from adr.io.dataset1 import available_subjects, load_subject

log = logging.getLogger("verify_dataset1")


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


def main() -> None:
    paths = load_paths()
    out_dir = paths["results"] / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "dataset1_verify.log", mode="w")],
    )
    rng = seed_everything()
    log.info("package versions: %s", package_versions())

    montage = mne.channels.make_standard_montage("biosemi64")
    standard = tuple(montage.ch_names)
    montage_pos = montage.get_positions()["ch_pos"]
    stim_dir = paths["dataset1"] / "stimuli"

    subjects = available_subjects()
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
        null = [distance_correlation(eeg, positions[rng.permutation(64)]) for _ in range(200)]

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

    with open(out_dir / "dataset1_verify.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
