"""Frontal-reference ridge regression for ocular artifacts (PROJECT_PLAN.md §8).

With no EOG channels, Fp1, Fp2, AF7 and AF8 stand in for EOG. Within the
ocular-flagged samples, every other channel is regressed on these reference
channels (ridge) and the predicted part is subtracted - the classic
regression approach to EOG correction, fitted only where the detector says
eye artifacts are present. The reference channels themselves are left to
the MWF (they cannot be corrected by regressing on themselves).
"""
from __future__ import annotations

import numpy as np


def correct(data: np.ndarray, mask: np.ndarray, ch_names, reference=("Fp1", "Fp2", "AF7", "AF8"),
            alpha: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Corrected copy of `data` (only flagged samples change) and the betas (refs x others)."""
    mask = np.asarray(mask, bool)
    ch_names = list(ch_names)
    ref = [ch_names.index(c) for c in reference]
    others = [i for i in range(len(ch_names)) if i not in ref]
    out = data.copy()
    if mask.sum() < len(ref) + 1:
        return out, np.zeros((len(ref), len(others)))
    # Centre on the clean-sample baseline, not on the flagged samples: an eye
    # artifact is not zero-mean within its own windows (a blink is one-signed),
    # so centring there would leave its mean behind as an offset.
    baseline = data[~mask].mean(0) if (~mask).any() else np.zeros(data.shape[1])
    R = data[mask][:, ref] - baseline[ref]
    Y = data[mask][:, others] - baseline[others]
    penalty = alpha * np.trace(R.T @ R) / len(ref)
    beta = np.linalg.solve(R.T @ R + penalty * np.eye(len(ref)), R.T @ Y)
    out[np.ix_(mask, others)] -= R @ beta
    return out, beta
