"""Multi-channel Wiener filter for artifact removal (Somers et al. 2018; PROJECT_PLAN.md §4, §8).

Model y = d + n (d: artifact, n: neural), with y optionally stacked with
time lags. R_yy is estimated on artifact-flagged samples, R_nn on clean
ones. A generalized eigenvalue decomposition R_yy x = lambda R_nn x,
normalised so X' R_nn X = I, gives R_dd = X^-T (Lambda - I) X^-1 and the
filter

    W = X diag(g) X^-1,   g_i = (lambda_i - 1) / (lambda_i - 1 + mu)

for the kept components (lambda_i > 1 for rank "poseig", lambda_i > T for
"eig>T", or the top r) and
g_i = 0 otherwise (the GEVD low-rank approximation of Somers et al.). The
artifact estimate is d_hat(t) = W' y(t) (the rows for the lag-0 channels),
and the cleaned EEG is y - d_hat. mu = 1 is the plain MWF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def stack_lags(data: np.ndarray, delay: int) -> np.ndarray:
    """samples x (channels * (2 delay + 1)) with lags -delay..delay (zero-padded)."""
    if delay == 0:
        return data
    n, c = data.shape
    out = np.zeros((n, c * (2 * delay + 1)))
    for j, lag in enumerate(range(-delay, delay + 1)):
        cols = slice(j * c, (j + 1) * c)
        if lag >= 0:
            out[lag:, cols] = data[: n - lag]
        else:
            out[:lag, cols] = data[-lag:]
    return out


@dataclass
class MWF:
    weights: np.ndarray       # (channels * lags) x channels
    delay: int
    eigenvalues: np.ndarray   # generalized eigenvalues, descending
    rank: int
    baseline: np.ndarray      # clean-sample channel means (the filter acts on deviations from them)

    def artifact(self, data: np.ndarray) -> np.ndarray:
        return stack_lags(data - self.baseline, self.delay) @ self.weights

    def clean(self, data: np.ndarray) -> np.ndarray:
        return data - self.artifact(data)


def fit(data: np.ndarray, artifact_mask: np.ndarray, clean_mask: np.ndarray | None = None,
        delay: int = 0, rank: int | str = "poseig", mu: float = 1.0) -> MWF:
    """Fit the GEVD-based MWF from flagged (artifact) and clean samples of one recording."""
    artifact_mask = np.asarray(artifact_mask, bool)
    clean_mask = ~artifact_mask if clean_mask is None else np.asarray(clean_mask, bool)
    baseline = data[clean_mask].mean(0)
    y = stack_lags(data - baseline, delay)
    r_yy = np.cov(y[artifact_mask], rowvar=False)
    r_nn = np.cov(y[clean_mask], rowvar=False)
    r_nn += 1e-10 * np.trace(r_nn) / len(r_nn) * np.eye(len(r_nn))  # numerical floor
    lam, X = scipy.linalg.eigh(r_yy, r_nn)       # X' R_nn X = I, ascending
    order = np.argsort(lam)[::-1]
    lam, X = lam[order], X[:, order]
    n_pos = int(np.sum(lam > 1))
    if rank == "poseig":
        r = n_pos
    elif isinstance(rank, str) and rank.startswith("eig>"):
        r = int(np.sum(lam > float(rank[4:])))   # components where artifact power exceeds (T - 1) x neural power
    else:
        r = min(int(rank), n_pos)
    gains = np.zeros(len(lam))
    gains[:r] = (lam[:r] - 1) / (lam[:r] - 1 + mu)
    W = X @ np.diag(gains) @ np.linalg.inv(X)    # stacked x stacked
    c = data.shape[1]
    lag0 = slice(delay * c, (delay + 1) * c)
    return MWF(weights=W[:, lag0], delay=delay, eigenvalues=lam, rank=r, baseline=baseline)
