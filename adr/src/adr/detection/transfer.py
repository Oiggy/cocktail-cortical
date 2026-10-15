"""Detector transfer to the phantom (PROJECT_PLAN.md §7; milestone 5).

Applies the frozen PhysioMotion detectors to the held-out phantom pool
(`phantom_check`, 200-300 s of each condition; nothing is tuned). The
phantom's 128 scalp channels are spline-mapped to BioSemi 64, then go
through the same detector preprocessing and features as training.

Labels come from which artifact sources are switched on (condition level):
Brain = clean; Eyes = ocular; Facial, Neck = muscle; Walking = motion;
All = ocular + muscle + motion. They are coarse (an "on" eye source does
not blink in every window), so the report gives the ROC-AUC of each
condition's windows against the brain-only windows, and the fraction of
windows flagged at the default threshold.

Used by scripts/09_phantom_transfer.py and notebooks/04_detectors.ipynb.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adr.detection import features, prep
from adr.detection import models as M
from adr.detection.train import load_model
from adr.io import phantom
from adr.preprocessing import spline
from adr.preprocessing.channels import detector_channels

log = logging.getLogger("adr.phantom_transfer")
CONDITION_CLASSES = {
    "Brain": (), "Eyes": ("ocular",), "Facial": ("muscle",), "Neck": ("muscle",),
    "Walking": ("motion",), "All": ("ocular", "muscle", "motion"),
}
TARGETS = ("artifact", "ocular", "muscle", "motion")


def phantom_features(condition: str, fs: float, pool: str = "phantom_check") -> np.ndarray:
    cfg = features.detection_config()
    trial = phantom.load_trial(condition)
    span = phantom.pool_slice(pool, trial.sfreq, len(trial.eeg))
    data64, names = spline.to_biosemi64(trial.eeg[span], spline.positions(trial.info))
    data = prep.to_rate(data64, trial.sfreq, fs)
    return features.compute(prep.finish(data, fs, names, cfg["highpass_hz"]), fs, detector_channels(), cfg)


def run(rates=None, models=M.MODELS) -> pd.DataFrame:
    rows = []
    for fs in rates or [float(r) for r in features.detection_config()["rates"]]:
        X = {c: phantom_features(c, fs) for c in CONDITION_CLASSES}
        for model_name in models:
            for target in TARGETS:
                model = load_model(fs, model_name, target)
                scores = {c: M.score(model_name, model, X[c]) for c in X}
                thr = M.DEFAULT_THRESHOLD[model_name]
                for condition, classes in CONDITION_CLASSES.items():
                    if condition == "Brain":
                        continue
                    y = np.r_[np.zeros(len(scores["Brain"])), np.ones(len(scores[condition]))]
                    rows.append({
                        "rate": int(fs), "model": model_name, "target": target, "condition": condition,
                        "source_on_for_target": target == "artifact" or target in classes,
                        "auc_vs_brain": roc_auc_score(y, np.r_[scores["Brain"], scores[condition]]),
                        "flag_rate": float(np.mean(scores[condition] >= thr)),
                        "brain_flag_rate": float(np.mean(scores["Brain"] >= thr)),
                    })
                    r = rows[-1]
                    log.info("%d Hz %-8s %-8s %-7s (source on: %-5s) AUC vs brain %.3f | flagged %.2f (brain %.2f)",
                             fs, model_name, target, condition, r["source_on_for_target"], r["auc_vs_brain"],
                             r["flag_rate"], r["brain_flag_rate"])
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, model: str = "lightgbm"):
    rates = sorted(table["rate"].unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 3.8), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        sub = table[(table["rate"] == rate) & (table["model"] == model)]
        pivot = sub.pivot(index="condition", columns="target", values="auc_vs_brain")
        im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for (i, j), v in np.ndenumerate(pivot.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{rate} Hz {model}: AUC vs brain-only (phantom)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
