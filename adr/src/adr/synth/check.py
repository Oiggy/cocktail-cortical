"""Dataset III synthesis check (PROJECT_PLAN.md §9.2-9.3 steps 1-5; milestone 4).

Synthesises both conditions at every SNR for one noise subject (no artifacts
yet), checks the achieved SNR, and measures how well the analysis pipeline
(Das preprocessing, per-channel z-score, ridge TRF) recovers the base TRF,
long-term (whole ~38.5 min) and short-term (120 s trials, lambda 0.2 and the
Das rule - open item 11). Set A subjects only; nothing is frozen here.

Used by scripts/06_synthesize_dataset3.py and notebooks/03_dataset3_synthesis.ipynb.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from adr.config import load_paths, load_yaml
from adr.preprocessing.das import preprocess_eeg, zscore
from adr.synth import dataset3 as d3
from adr.trf import ridge
from adr.trf.metrics import relative_mse

log = logging.getLogger("adr.dataset3_check")
FS = 32.0


def recovery(rec: d3.SynthRecording, base: np.ndarray, cfg: dict, trial_seconds: float) -> dict:
    eeg = zscore(preprocess_eeg(rec.eeg))
    env = rec.envelope[: len(eeg)]
    env = (env - env.mean()) / env.std()
    out = {"long_term": relative_mse(ridge.fit([env], [eeg], FS, cfg["tmin"], cfg["tmax"], "das").weights, base)}
    short = {"short_0.2": [], "short_das": []}
    for cut in d3.cut_trials(len(eeg), trial_seconds):
        for key, lam in (("short_0.2", cfg["lambda_short"]), ("short_das", "das")):
            short[key].append(relative_mse(ridge.fit([env[cut]], [eeg[cut]], FS, cfg["tmin"], cfg["tmax"], lam).weights, base))
    out.update({k: float(np.median(v)) for k, v in short.items()})
    out["n_trials"] = len(short["short_das"])
    return out


def run(subject: str | None = None) -> pd.DataFrame:
    section = load_yaml("section_a")
    base, lags, _ = d3.load_base_trfs()
    subject = subject or d3.load_noise_split()["set_a"][0]
    source = d3.noise_source(subject)
    rows = []
    for snr in section["snr_db"]:
        for cond in d3.CONDITIONS:
            rec = d3.synthesize(source, cond, snr, base, lags)
            row = {"subject": subject, "snr_db": snr, "condition": cond,
                   "achieved_snr_db": d3.snr_db(rec.clean, rec.eeg - rec.clean),
                   **recovery(rec, base[cond], section["trf"], section["trial_seconds"])}
            rows.append(row)
            log.info("%s SNR %+5.1f (achieved %+6.2f) %s: long-term %.3f | 120 s median: lambda 0.2 %.3f, Das %.3f",
                     subject, snr, row["achieved_snr_db"], cond, row["long_term"], row["short_0.2"], row["short_das"])
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame):
    section = load_yaml("section_a")
    base, lags, names = d3.load_base_trfs()
    ref = int(np.load(load_paths()["results"] / "section_a" / "long_term_trf.npz")["reference_lag"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    c = names.index(section["base_channel"])
    for ear, color in (("L", "tab:blue"), ("R", "tab:red")):
        axes[0].plot(1000 * lags / FS, base[ear][:, c], color=color, label=f"W_base{ear}")
    axes[0].set(xlabel="lag (ms)", title=f"Base TRF at {section['base_channel']}")
    axes[0].legend(frameon=False)
    info = mne.create_info(names, FS, "eeg")
    info.set_montage("biosemi64")
    for ax, ear in zip(axes[1:3], "LR"):
        mne.viz.plot_topomap(base[ear][ref], info, axes=ax, show=False)
        ax.set_title(f"W_base{ear} at {1000 * lags[ref] / FS:.0f} ms")
    mean = table.groupby("snr_db")[["long_term", "short_0.2", "short_das"]].mean().sort_index(ascending=False)
    for col, label in (("long_term", "long-term (38 min)"), ("short_0.2", "120 s, lambda 0.2"), ("short_das", "120 s, Das rule")):
        axes[3].plot(mean.index, mean[col], marker="o", label=label)
    axes[3].set(xlabel="SNR (dB)", ylabel="relative MSE", ylim=(0, 1.05),
                title=f"Recovery, noise subject {', '.join(table['subject'].unique())}")
    axes[3].invert_xaxis()
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


NATIVE_THRESHOLDS = (150, 200, 300, 500, None)


def native_cut_table(subjects=None, thresholds=NATIVE_THRESHOLDS) -> pd.DataFrame:
    """How much of each noise subject's base is cut at each threshold (descriptive, all subjects)."""
    from adr.io.dataset1 import available_subjects

    rows = []
    for subject in subjects or available_subjects():
        for thr in thresholds:
            src = d3.noise_source(subject, ptp_uv=thr)
            minutes = len(src.noise) / 128 / 60
            rows.append({"subject": subject, "threshold_uv": thr if thr is not None else "none",
                         "fraction_cut": src.fraction_cut, "minutes": minutes,
                         "trials_120s_per_condition": len(d3.cut_trials(len(src.noise) // 4))})
    return pd.DataFrame(rows)


def native_recovery_sensitivity(subject: str | None = None, snr_db: float = -10,
                                thresholds=NATIVE_THRESHOLDS) -> pd.DataFrame:
    """Base-TRF recovery (no artifacts) on a Set A subject at each cut threshold."""
    section = load_yaml("section_a")
    base, lags, _ = d3.load_base_trfs()
    subject = subject or d3.load_noise_split()["set_a"][0]
    rows = []
    for thr in thresholds:
        src = d3.noise_source(subject, ptp_uv=thr)
        for cond in d3.CONDITIONS:
            rec = d3.synthesize(src, cond, snr_db, base, lags)
            r = recovery(rec, base[cond], section["trf"], section["trial_seconds"])
            rows.append({"subject": subject, "threshold_uv": thr if thr is not None else "none", "condition": cond,
                         "fraction_cut": src.fraction_cut, **r})
            log.info("%s threshold %s %s: cut %.1f %%, long-term relMSE %.3f, 120 s median relMSE (Das) %.3f, %d trials",
                     subject, thr, cond, 100 * src.fraction_cut, r["long_term"], r["short_das"], r["n_trials"])
    return pd.DataFrame(rows)
