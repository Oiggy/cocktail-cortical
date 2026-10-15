"""Supervised, non-neural detectors (PROJECT_PLAN.md §2, §7).

LightGBM (main), LDA (baseline) and an RBF SVM. Each target is a binary
problem (artifact vs clean, or one class vs everything else); class
imbalance is handled with class weights. Every model returns a score where
larger means "more likely artifact": a probability for LightGBM and LDA,
the decision function for the SVM.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("lightgbm", "lda", "svm")
SVM_MAX_TRAIN = 20000  # RBF SVM training cost grows ~quadratically; stratified subsample
DEFAULT_THRESHOLD = {"lightgbm": 0.5, "lda": 0.5, "svm": 0.0}


def make(name: str, seed: int):
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced",
            random_state=seed, n_jobs=8, verbose=-1,
        )
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                                          priors=[0.5, 0.5]))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                                                   random_state=seed))
    raise ValueError(name)


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-stratified subsample of size <= n."""
    if len(y) <= n:
        return np.arange(len(y))
    idx = []
    for value in np.unique(y):
        members = np.flatnonzero(y == value)
        take = max(1, int(round(n * len(members) / len(y))))
        idx.append(rng.choice(members, min(take, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def fit(name: str, X: np.ndarray, y: np.ndarray, seed: int):
    model = make(name, seed)
    if name == "svm":
        keep = stratified_subsample(y, SVM_MAX_TRAIN, np.random.default_rng(seed))
        X, y = X[keep], y[keep]
    model.fit(X, y)
    return model


def score(name: str, model, X: np.ndarray) -> np.ndarray:
    if name == "svm":
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]
