"""Ridge-regression TRFs in the mTRF convention (PROJECT_PLAN.md §4, §9).

Forward model: EEG(t, ch) = b(ch) + sum_lag w(lag, ch) * stimulus(t - lag).
Matches mTRFpy (`mtrf.TRF(direction=1, zeropad=True).train`) exactly, which
follows mTRF-Toolbox v2:

- the lagged design is zero-padded at each trial's edges (no lag reaches
  across a trial boundary) and has a leading bias column;
- X'X and X'Y are *averaged* over trials;
- w = (X'X + lambda * fs * M)^-1 X'Y * fs, with M the identity except
  M[0, 0] = 0 (the bias is not penalised).

So the plan's lambda (e.g. 0.2 for short-term TRFs) means what it means in
mTRF. Checked against mtrf 2.1.2 to machine precision
(tests/test_trf_ridge.py runs the comparison when mtrf is installed).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def lag_matrix(stimulus: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """samples x (lags * features) design, lag-major; positive lags = stimulus leads."""
    stimulus = np.asarray(stimulus, dtype=np.float64)
    if stimulus.ndim == 1:
        stimulus = stimulus[:, None]
    n, n_features = stimulus.shape
    out = np.zeros((n, n_features * len(lags)))
    for j, lag in enumerate(lags):
        cols = slice(j * n_features, (j + 1) * n_features)
        if lag >= 0:
            out[lag:, cols] = stimulus[: n - lag]
        else:
            out[:lag, cols] = stimulus[-lag:]
    return out


def lags_for(tmin: float, tmax: float, fs: float) -> np.ndarray:
    """Integer lags from tmin to tmax s, inclusive (mTRF `lagGen`)."""
    return np.arange(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1)


@dataclass(frozen=True)
class TRF:
    weights: np.ndarray  # lags x channels (single stimulus feature), mTRF units (x fs)
    bias: np.ndarray     # channels
    lags: np.ndarray
    fs: float

    @property
    def times(self) -> np.ndarray:
        return self.lags / self.fs

    def predict(self, stimulus: np.ndarray) -> np.ndarray:
        return self.bias + lag_matrix(stimulus, self.lags) @ self.weights / self.fs


def covariances(stimuli, responses, lags) -> tuple[np.ndarray, np.ndarray]:
    """Trial-averaged X'X and X'Y with the bias column (mTRF convention)."""
    xtx, xty = 0.0, 0.0
    for stimulus, response in zip(stimuli, responses):
        X = lag_matrix(stimulus, lags)
        X = np.column_stack([np.ones(len(X)), X])
        xtx = xtx + X.T @ X
        xty = xty + X.T @ np.asarray(response, dtype=np.float64)
    return xtx / len(stimuli), xty / len(stimuli)


def solve(xtx: np.ndarray, xty: np.ndarray, lam: float, fs: float, lags) -> TRF:
    penalty = np.eye(len(xtx))
    penalty[0, 0] = 0.0
    w = np.linalg.solve(xtx + lam * fs * penalty, xty) * fs
    return TRF(weights=w[1:], bias=w[0] / fs, lags=np.asarray(lags), fs=fs)


def das_lambda(xtx: np.ndarray, fs: float) -> float:
    """mTRF lambda whose penalty equals max |X'X| over the lag block.

    Das et al. (2020) choose the ridge parameter "as the maximum absolute
    value among all the elements in the ... speech covariance matrix"; in
    the mTRF convention the penalty is lambda * fs, hence the division.
    """
    return float(np.abs(xtx[1:, 1:]).max() / fs)


def fit(
    stimuli: Sequence[np.ndarray],
    responses: Sequence[np.ndarray],
    fs: float,
    tmin: float,
    tmax: float,
    lam: float | str,
) -> TRF:
    """Forward TRF over one or more trials (one stimulus feature).

    `lam` is the mTRF ridge parameter, or "das" for `das_lambda`.
    """
    lags = lags_for(tmin, tmax, fs)
    xtx, xty = covariances(stimuli, responses, lags)
    if lam == "das":
        lam = das_lambda(xtx, fs)
    return solve(xtx, xty, lam, fs, lags)
