"""Milestone 3: long-term TRFs on Dataset I v2.0 (PROJECT_PLAN.md §9.1-9.2 step 1-2).

Fits forward TRFs (attended envelope -> 63-channel EEG, lags 0-400 ms)
across all subjects' HRTF trials, separately for attend-left (W_L) and
attend-right (W_R), and finds the reference lag where W_L and W_R jointly
have maximum norm. As a sanity check that the TRF is "sensible", it also
cross-validates prediction with whole story parts held out (the CV grouping
rule, §9.4) and compares against a circularly shifted envelope (null).

Writes results/section_a/long_term_trf.npz, long_term_trf.png and a log.

    python scripts/05_long_term_trf.py
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from adr.config import load_paths, load_yaml, package_versions, seed_everything
from adr.io.dataset1 import available_subjects
from adr.preprocessing.section_a import prepare_subject
from adr.trf import ridge

log = logging.getLogger("long_term_trf")


def fit_side(trials, ear, cfg, fs):
    side = [t for t in trials if t.attended_ear == ear]
    return ridge.fit([t.envelope for t in side], [t.eeg for t in side], fs,
                     cfg["tmin"], cfg["tmax"], cfg["lambda_long"])


def reference_lag(w_left: np.ndarray, w_right: np.ndarray) -> int:
    """Index of the lag where [W_L, W_R] has the largest norm across channels."""
    return int(np.argmax(np.sqrt((w_left ** 2).sum(1) + (w_right ** 2).sum(1))))


def grouped_cv(trials, cfg, fs, shift_s=None):
    """Mean prediction r per channel, holding out one story part at a time."""
    rs = []
    for group in sorted({t.group for t in trials}):
        train = [t for t in trials if t.group != group]
        test = [t for t in trials if t.group == group]
        models = {ear: fit_side(train, ear, cfg, fs) for ear in "LR"}
        for t in test:
            env = np.roll(t.envelope, int(shift_s * fs)) if shift_s else t.envelope
            pred = models[t.attended_ear].predict(env)
            rs.append([np.corrcoef(pred[:, c], t.eeg[:, c])[0, 1] for c in range(t.eeg.shape[1])])
    return np.array(rs)


def main() -> None:
    paths = load_paths()
    cfg = load_yaml("section_a")["trf"]
    base_channel = load_yaml("section_a")["base_channel"]
    out_dir = paths["results"] / "section_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "long_term_trf.log", mode="w")])
    seed_everything()
    log.info("package versions: %s", package_versions())
    fs = 32.0

    trials, ch_names = [], None
    for subject in available_subjects():
        subject_trials, ch_names = prepare_subject(subject)
        trials += subject_trials
    minutes = sum(len(t.envelope) for t in trials) / fs / 60
    log.info("%d HRTF trials from %d subjects, %.1f min total (%.1f min/subject)",
             len(trials), len({t.subject for t in trials}), minutes, minutes / len({t.subject for t in trials}))

    w = {ear: fit_side(trials, ear, cfg, fs) for ear in "LR"}
    lags = w["L"].lags
    ref = reference_lag(w["L"].weights, w["R"].weights)
    c5 = ch_names.index(base_channel)
    log.info("reference lag: %d samples = %.0f ms", lags[ref], 1000 * lags[ref] / fs)
    for ear in "LR":
        trf = w[ear].weights[:, c5]
        log.info("W_%s at %s (lag ms: weight): %s", ear, base_channel,
                 ", ".join(f"{1000 * l / fs:.0f}: {v:+.3f}" for l, v in zip(lags, trf)))
    log.info("left/right TRF correlation across channels at the reference lag: %.2f",
             np.corrcoef(w["L"].weights[ref], w["R"].weights[ref])[0, 1])

    r_true = grouped_cv(trials, cfg, fs)
    r_null = grouped_cv(trials, cfg, fs, shift_s=60)
    log.info("story-part-held-out prediction r (mean over channels, mean over trials): %.4f; "
             "shifted-envelope null: %.4f; best channel %s r = %.4f",
             r_true.mean(), r_null.mean(), ch_names[int(r_true.mean(0).argmax())], r_true.mean(0).max())
    log.info("fraction of trials with mean r above their null: %.2f", (r_true.mean(1) > r_null.mean(1)).mean())

    np.savez(out_dir / "long_term_trf.npz", w_left=w["L"].weights, w_right=w["R"].weights,
             bias_left=w["L"].bias, bias_right=w["R"].bias, lags=lags, fs=fs, reference_lag=ref,
             ch_names=np.array(ch_names), cv_r=r_true, cv_r_null=r_null)

    info = mne.create_info(ch_names, fs, "eeg")
    info.set_montage("biosemi64")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ear, color in (("L", "tab:blue"), ("R", "tab:red")):
        axes[0].plot(1000 * lags / fs, w[ear].weights[:, c5], color=color, label=f"attend {ear}")
    axes[0].axvline(1000 * lags[ref] / fs, color="k", ls=":", lw=1)
    axes[0].set(xlabel="lag (ms)", ylabel="weight (a.u.)", title=f"Long-term TRF at {base_channel}")
    axes[0].legend(frameon=False)
    vmax = np.abs(np.r_[w["L"].weights[ref], w["R"].weights[ref]]).max()
    for ax, ear in zip(axes[1:], "LR"):
        mne.viz.plot_topomap(w[ear].weights[ref], info, axes=ax, show=False, vlim=(-vmax, vmax))
        ax.set_title(f"W_{ear} at {1000 * lags[ref] / fs:.0f} ms")
    fig.tight_layout()
    fig.savefig(out_dir / "long_term_trf.png", dpi=150)
    log.info("saved %s", out_dir / "long_term_trf.png")


if __name__ == "__main__":
    main()
