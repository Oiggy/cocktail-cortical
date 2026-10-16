"""Learned spatial interpolation for muscle and electrode-pop artifacts (PROJECT_PLAN.md §8).

Ridge regression trained on the recording's clean (unflagged) samples
predicts a channel from its neighbours; contaminated samples are replaced
by the prediction. The detector flags windows, not channels, so within each
flagged window the contaminated channels are found first: a channel is bad
if the robust z-score of its contamination score (log 20-60 Hz power for
muscle, log largest first difference for pops), relative to that channel's
clean-window distribution, exceeds `bad_z`. Each bad channel is then
predicted from its `n_neighbours` nearest good channels, with ridge weights
solved from the clean covariance (cheap per window). The penalty (`alpha`,
relative to the mean clean variance) matters: neighbouring channels are
nearly collinear, and a near-unpenalised fit amplifies any contamination
left in the neighbours. Windows with too many bad channels are left
to a fallback (MWF in the router).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from adr.preprocessing import spline


def neighbour_order(ch_names) -> np.ndarray:
    """channels x channels: other channels sorted by distance on the BioSemi 64 sphere."""
    names, pos = spline.biosemi64_positions()
    p = pos[[names.index(c) for c in ch_names]]
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    return np.argsort(d, axis=1)[:, 1:]


def window_scores(data: np.ndarray, fs: float, starts: np.ndarray, win: int, band=None) -> np.ndarray:
    """windows x channels contamination score.

    band = (lo, hi): log band power (muscle); band = "step": log of the
    largest absolute first difference, which picks out the abrupt step of an
    electrode pop far better than the variance of a decaying window;
    band = None: log broadband variance.
    """
    if band == "step":
        jumps = np.abs(np.diff(data, axis=0, prepend=data[:1]))
        return np.array([np.log10(jumps[s:s + win].max(axis=0) + 1e-30) for s in starts])
    if band is not None:
        sos = scipy.signal.butter(4, band, "bandpass", fs=fs, output="sos")
        data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    return np.array([np.log10(data[s:s + win].var(axis=0) + 1e-30) for s in starts])


@dataclass
class Interpolator:
    cov: np.ndarray             # clean covariance (channels x channels)
    mean: np.ndarray
    order: np.ndarray           # neighbour order
    n_neighbours: int
    alpha: float

    def predict(self, data: np.ndarray, bad: list[int]) -> np.ndarray:
        """Predictions for the `bad` channels (samples x len(bad)) from good neighbours."""
        bad_set = set(bad)
        out = np.zeros((len(data), len(bad)))
        scale = self.alpha * np.trace(self.cov) / len(self.cov)
        for j, b in enumerate(bad):
            nb = [i for i in self.order[b] if i not in bad_set][: self.n_neighbours]
            C = self.cov[np.ix_(nb, nb)] + scale * np.eye(len(nb))
            beta = np.linalg.solve(C, self.cov[nb, b])
            out[:, j] = self.mean[b] + (data[:, nb] - self.mean[nb]) @ beta
        return out


def fit(data: np.ndarray, clean_mask: np.ndarray, ch_names, n_neighbours: int = 8, alpha: float = 1e-3) -> Interpolator:
    clean = data[np.asarray(clean_mask, bool)]
    return Interpolator(np.cov(clean, rowvar=False), clean.mean(0), neighbour_order(ch_names), n_neighbours, alpha)


def bad_channels(scores: np.ndarray, clean_scores: np.ndarray, bad_z: float) -> np.ndarray:
    """windows x channels boolean: robust z of each window's score vs the clean windows."""
    median = np.median(clean_scores, axis=0)
    mad = np.median(np.abs(clean_scores - median), axis=0) * 1.4826 + 1e-12
    return (scores - median) / mad > bad_z
