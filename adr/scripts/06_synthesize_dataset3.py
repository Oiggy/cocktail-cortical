"""Milestone 4: Dataset III synthesis check (PROJECT_PLAN.md §9.2-9.3 steps 1-5).

Builds the base TRFs from the milestone-3 long-term TRFs, synthesises both
conditions at every SNR in configs/section_a.yaml for one Set A noise
subject (no artifacts yet), checks the achieved SNR, and measures how well
the analysis pipeline (Das preprocessing, per-channel z-score, ridge TRF)
recovers the base TRF:

- long-term: one TRF over the whole ~38.5 min recording (Das lambda rule);
- short-term preview: per 120 s trial with lambda 0.2 and with the Das rule
  (informs open item 11; Set A only, nothing is frozen here).

Writes results/section_a/base_trfs.npz, dataset3_check.png and a log.

    python scripts/06_synthesize_dataset3.py
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from adr.config import load_paths, load_yaml, package_versions, seed_everything
from adr.preprocessing.das import preprocess_eeg, zscore
from adr.synth import dataset3 as d3
from adr.trf import ridge
from adr.trf.metrics import relative_mse

log = logging.getLogger("synthesize_dataset3")
FS = 32.0


def recovery(rec: d3.SynthRecording, base: np.ndarray, cfg: dict, trial_seconds: float):
    eeg = zscore(preprocess_eeg(rec.eeg))
    env = rec.envelope[: len(eeg)]
    env = (env - env.mean()) / env.std()
    long_term = relative_mse(ridge.fit([env], [eeg], FS, cfg["tmin"], cfg["tmax"], "das").weights, base)
    short = {"0.2": [], "das": []}
    for cut in d3.cut_trials(len(eeg), trial_seconds):
        for key, lam in (("0.2", cfg["lambda_short"]), ("das", "das")):
            trf = ridge.fit([env[cut]], [eeg[cut]], FS, cfg["tmin"], cfg["tmax"], lam)
            short[key].append(relative_mse(trf.weights, base))
    return long_term, {k: float(np.median(v)) for k, v in short.items()}, len(short["das"])


def main() -> None:
    paths = load_paths()
    section = load_yaml("section_a")
    cfg = section["trf"]
    out_dir = paths["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "dataset3_check.log", mode="w")])
    seed_everything()
    log.info("package versions: %s", package_versions())

    base, lags, names = d3.load_base_trfs()
    np.savez(out_dir / "base_trfs.npz", base_left=base["L"], base_right=base["R"], lags=lags, ch_names=np.array(names))
    split = d3.load_noise_split()
    subject = split["set_a"][0]
    source = d3.noise_source(subject)
    log.info("noise subject %s (Set A): %.1f min, %d trials; noise sd (analysis space) %.1f-%.1f uV",
             subject, len(source.noise) / 128 / 60, len(source.envelopes),
             1e0 * source.noise_sd.min(), 1e0 * source.noise_sd.max())

    rows = []
    for snr in section["snr_db"]:
        for cond in d3.CONDITIONS:
            rec = d3.synthesize(source, cond, snr, base, lags)
            achieved = d3.snr_db(rec.clean, rec.eeg - rec.clean)
            lt, st, n_trials = recovery(rec, base[cond], cfg, section["trial_seconds"])
            rows.append((snr, cond, achieved, lt, st["0.2"], st["das"]))
            log.info("SNR %+5.1f dB (achieved %+6.2f) cond %s: long-term relMSE %.3f | short-term (%d x 120 s) "
                     "median relMSE: lambda 0.2 %.3f, Das rule %.3f", snr, achieved, cond, lt, n_trials, st["0.2"], st["das"])

    rows = np.array([(r[0], r[3], r[4], r[5]) for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    c5 = names.index(section["base_channel"])
    for ear, color in (("L", "tab:blue"), ("R", "tab:red")):
        axes[0].plot(1000 * lags / FS, base[ear][:, c5], color=color, label=f"W_base{ear}")
    axes[0].set(xlabel="lag (ms)", title=f"Base TRF at {section['base_channel']}")
    axes[0].legend(frameon=False)
    info = mne.create_info(names, FS, "eeg")
    info.set_montage("biosemi64")
    ref = int(np.load(out_dir / "long_term_trf.npz")["reference_lag"])
    for ax, ear in zip(axes[1:3], "LR"):
        mne.viz.plot_topomap(base[ear][ref], info, axes=ax, show=False)
        ax.set_title(f"W_base{ear} at {1000 * lags[ref] / FS:.0f} ms")
    snrs = sorted(set(rows[:, 0]), reverse=True)
    for col, label in ((1, "long-term (38 min)"), (2, "120 s, lambda 0.2"), (3, "120 s, Das rule")):
        axes[3].plot(snrs, [rows[rows[:, 0] == s, col].mean() for s in snrs], marker="o", label=label)
    axes[3].set(xlabel="SNR (dB)", ylabel="relative MSE", ylim=(0, 1.05), title=f"Recovery, noise subject {subject}")
    axes[3].invert_xaxis()
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset3_check.png", dpi=150)
    log.info("saved %s", out_dir / "dataset3_check.png")


if __name__ == "__main__":
    main()
