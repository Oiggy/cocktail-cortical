"""Long-term TRFs on Dataset I v2.0 (PROJECT_PLAN.md §9.2 steps 1-2; milestone 3).

Forward TRFs (attended envelope -> 63-channel EEG, lags 0-400 ms) across all
subjects' HRTF trials, separately for attend-left (W_L) and attend-right
(W_R); the reference lag is where W_L and W_R jointly have maximum norm.
A sanity check cross-validates prediction with whole story parts held out
(§9.4 grouping) against a circularly shifted envelope.

Used by scripts/05_long_term_trf.py and notebooks/02_long_term_trf.ipynb.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np

from adr.config import load_paths, load_yaml
from adr.io.dataset1 import available_subjects
from adr.preprocessing.section_a import SectionATrial, prepare_subject
from adr.trf import ridge

log = logging.getLogger("adr.long_term_trf")
FS = 32.0


@dataclass
class LongTermResult:
    w: dict                 # 'L'/'R' -> ridge.TRF
    reference_lag: int      # index into lags
    ch_names: list
    cv_r: np.ndarray        # trials x channels, story-part held out
    cv_r_null: np.ndarray   # same with a 60 s circular shift of the envelope
    n_trials: int
    minutes: float

    @property
    def lags(self) -> np.ndarray:
        return self.w["L"].lags


def load_trials(subjects=None) -> tuple[list[SectionATrial], list[str]]:
    trials, names = [], None
    for subject in subjects or available_subjects():
        subject_trials, names = prepare_subject(subject)
        trials += subject_trials
    return trials, names


def fit_side(trials, ear: str, cfg: dict) -> ridge.TRF:
    side = [t for t in trials if t.attended_ear == ear]
    return ridge.fit([t.envelope for t in side], [t.eeg for t in side], FS, cfg["tmin"], cfg["tmax"], cfg["lambda_long"])


def reference_lag(w_left: np.ndarray, w_right: np.ndarray) -> int:
    """Index of the lag where [W_L, W_R] has the largest norm across channels."""
    return int(np.argmax(np.sqrt((w_left ** 2).sum(1) + (w_right ** 2).sum(1))))


def grouped_cv(trials, cfg: dict, shift_s: float | None = None) -> np.ndarray:
    """Prediction r per trial and channel, holding out one story part at a time."""
    rs = []
    for group in sorted({t.group for t in trials}):
        train = [t for t in trials if t.group != group]
        models = {ear: fit_side(train, ear, cfg) for ear in "LR"}
        for t in (t for t in trials if t.group == group):
            env = np.roll(t.envelope, int(shift_s * FS)) if shift_s else t.envelope
            pred = models[t.attended_ear].predict(env)
            rs.append([np.corrcoef(pred[:, c], t.eeg[:, c])[0, 1] for c in range(t.eeg.shape[1])])
    return np.array(rs)


def run(subjects=None) -> LongTermResult:
    cfg = load_yaml("section_a")["trf"]
    trials, names = load_trials(subjects)
    minutes = sum(len(t.envelope) for t in trials) / FS / 60
    w = {ear: fit_side(trials, ear, cfg) for ear in "LR"}
    result = LongTermResult(w=w, reference_lag=reference_lag(w["L"].weights, w["R"].weights), ch_names=names,
                            cv_r=grouped_cv(trials, cfg), cv_r_null=grouped_cv(trials, cfg, shift_s=60),
                            n_trials=len(trials), minutes=minutes)
    log.info("%d HRTF trials, %.1f min; reference lag %.0f ms; held-out r %.4f (null %.4f)",
             len(trials), minutes, 1000 * result.lags[result.reference_lag] / FS, result.cv_r.mean(),
             result.cv_r_null.mean())
    return result


def save(result: LongTermResult, path=None):
    path = path or load_paths()["results"] / "section_a" / "long_term_trf.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w_left=result.w["L"].weights, w_right=result.w["R"].weights, bias_left=result.w["L"].bias,
             bias_right=result.w["R"].bias, lags=result.lags, fs=FS, reference_lag=result.reference_lag,
             ch_names=np.array(result.ch_names), cv_r=result.cv_r, cv_r_null=result.cv_r_null)
    return path


def plot(result: LongTermResult, base_channel: str | None = None):
    base_channel = base_channel or load_yaml("section_a")["base_channel"]
    lags, ref, c = result.lags, result.reference_lag, result.ch_names.index(base_channel)
    info = mne.create_info(result.ch_names, FS, "eeg")
    info.set_montage("biosemi64")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ear, color in (("L", "tab:blue"), ("R", "tab:red")):
        axes[0].plot(1000 * lags / FS, result.w[ear].weights[:, c], color=color, label=f"attend {ear}")
    axes[0].axvline(1000 * lags[ref] / FS, color="k", ls=":", lw=1)
    axes[0].set(xlabel="lag (ms)", ylabel="weight (a.u.)", title=f"Long-term TRF at {base_channel}")
    axes[0].legend(frameon=False)
    vmax = np.abs(np.r_[result.w["L"].weights[ref], result.w["R"].weights[ref]]).max()
    for ax, ear in zip(axes[1:], "LR"):
        mne.viz.plot_topomap(result.w[ear].weights[ref], info, axes=ax, show=False, vlim=(-vmax, vmax))
        ax.set_title(f"W_{ear} at {1000 * lags[ref] / FS:.0f} ms")
    fig.tight_layout()
    return fig
