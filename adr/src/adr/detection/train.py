"""Detector training and participant-grouped CV on PhysioMotion (PROJECT_PLAN.md §7; milestone 5).

Clean = rest-block windows whose frontal peak-to-peak is at most
`clean_max_frontal_ptp_uv` (configs/detection.yaml); other rest windows
(unannotated blinks in the eyes-open rest) are left out.

For a detector rate and model: 5-fold CV grouped by participant over the
detector-training group, targets "artifact" (any class vs clean) and
one-vs-rest "ocular", "muscle", "motion", "pop". Window metrics use
labelled windows (rest = clean; unannotated task windows excluded) plus the
pop examples; event metrics and false alarms per hour use each held-out run
in time order. `fit_final` refits on all detector-training participants
(frozen; only the decision threshold is tuned later, on Section A Set A).

Used by scripts/08_train_detectors.py and notebooks/04_detectors.ipynb.
"""
from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from adr.config import ADR_ROOT, load_paths, load_project
from adr.detection import evaluate
from adr.detection import models as M
from adr.detection.dataset import run_features
from adr.detection.features import detection_config, feature_names
from adr.io.physiomotion import RUNS
from adr.preprocessing.channels import detector_channels

log = logging.getLogger("adr.train_detectors")
TARGETS = ("artifact", "ocular", "muscle", "motion", "pop")
ARTIFACT_CLASSES = ("ocular", "muscle", "motion")


def training_subjects() -> list[int]:
    with open(ADR_ROOT / "splits" / "physiomotion_participants.yaml") as f:
        return yaml.safe_load(f)["groups"]["detector_training"]


def output_dir(fs: float):
    path = load_paths()["results"] / "detectors" / f"{int(fs)}Hz"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rate(fs: float, subjects=None):
    """Training rows (labelled windows + pop examples), targets, groups, and per-run sequences."""
    max_ptp = detection_config()["clean_max_frontal_ptp_uv"]
    X, Y, groups, runs = [], [], [], []
    for s in subjects or training_subjects():
        for r in RUNS:
            f = run_features(s, r, fs)
            clean = f["rest"] & (f["fp_ptp_uv"] <= max_ptp)
            labelled = ~f["unlabelled"] & ~(f["rest"] & ~clean)  # drop blinky rest windows
            cls = {c: f[c][labelled] for c in ARTIFACT_CLASSES}
            X.append(f["X"][labelled])
            Y.append(np.column_stack([cls["ocular"] | cls["muscle"] | cls["motion"], cls["ocular"],
                                      cls["muscle"], cls["motion"], np.zeros(labelled.sum(), bool)]))
            groups.append(np.full(labelled.sum(), s))
            if len(f["X_pop"]):
                n = len(f["X_pop"])
                X.append(f["X_pop"])
                Y.append(np.column_stack([np.ones(n, bool), np.zeros((n, 3), bool), np.ones(n, bool)]))
                groups.append(np.full(n, s))
            runs.append({"subject": s, "run": r, "X": f["X"], "rest": clean, **{c: f[c] for c in ARTIFACT_CLASSES}})
    return np.concatenate(X), np.concatenate(Y), np.concatenate(groups), runs


def _run_target(run: dict, target: str) -> np.ndarray:
    return run["ocular"] | run["muscle"] | run["motion"] if target == "artifact" else run[target]


def cross_validate(fs: float, models=M.MODELS, targets=TARGETS, data=None) -> pd.DataFrame:
    cfg, seed = detection_config(), load_project()["seed"]
    X, Y, groups, runs = data or load_rate(fs)
    log.info("%d Hz: %d rows x %d features; positives %s", fs, *X.shape,
             {t: int(Y[:, TARGETS.index(t)].sum()) for t in targets})
    folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
    rows = []
    for model_name in models:
        for target in targets:
            ti = TARGETS.index(target)
            scores, event_rows = np.full(len(X), np.nan), []
            for k, (train, test) in enumerate(folds):
                model = M.fit(model_name, X[train], Y[train, ti], seed + k)
                scores[test] = M.score(model_name, model, X[test])
                if target != "pop":
                    held_out = set(np.unique(groups[test]))
                    for run in (r for r in runs if r["subject"] in held_out):
                        pred = M.score(model_name, model, run["X"]) >= M.DEFAULT_THRESHOLD[model_name]
                        event_rows.append(evaluate.event_metrics(_run_target(run, target), pred, run["rest"], cfg["step_s"]))
            metrics = evaluate.window_metrics(Y[:, ti], scores, M.DEFAULT_THRESHOLD[model_name])
            metrics.update(evaluate.summarize_events(event_rows) if event_rows else {})
            rows.append({"rate": int(fs), "model": model_name, "target": target, **metrics})
            log.info("%d Hz %-8s %-8s PR-AUC %.3f ROC-AUC %.3f F1 %.3f sens %.3f spec %.3f | events %.3f FA/h %.1f",
                     fs, model_name, target, metrics["pr_auc"], metrics["roc_auc"], metrics["f1"], metrics["sensitivity"],
                     metrics["specificity"], metrics.get("event_sensitivity", np.nan),
                     metrics.get("false_alarms_per_hour", np.nan))
    return pd.DataFrame(rows)


def fit_final(fs: float, models=M.MODELS, targets=TARGETS, data=None) -> None:
    seed = load_project()["seed"]
    X, Y, _, _ = data or load_rate(fs)
    out = output_dir(fs)
    (out / "feature_names.txt").write_text("\n".join(feature_names(detector_channels(), fs)))
    for model_name in models:
        for target in targets:
            joblib.dump(M.fit(model_name, X, Y[:, TARGETS.index(target)], seed), out / f"{model_name}_{target}.joblib")
    log.info("%d Hz: saved %d frozen models to %s", fs, len(models) * len(targets), out)


def load_model(fs: float, model_name: str, target: str):
    return joblib.load(output_dir(fs) / f"{model_name}_{target}.joblib")


def plot_cv(table: pd.DataFrame, metric: str = "pr_auc"):
    rates = sorted(table["rate"].unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 3.5), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        pivot = table[table["rate"] == rate].pivot(index="target", columns="model", values=metric).reindex(TARGETS)
        pivot.plot.bar(ax=ax, rot=0)
        ax.set(title=f"{rate} Hz detector: {metric} (participant-grouped 5-fold CV)", ylim=(0, 1), xlabel="")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
