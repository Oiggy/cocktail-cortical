"""TRF accuracy metrics (PROJECT_PLAN.md §9.4)."""
from __future__ import annotations

import numpy as np


def relative_mse(estimate: np.ndarray, base: np.ndarray) -> float:
    """Das et al. (2020) Eq. 14-15: relative MSE after least-squares scaling.

    All channels' TRFs are concatenated into one vector; the estimate is
    scaled by the alpha minimising ||base - alpha * estimate||^2, and the
    squared error is normalised by ||base||^2, giving a value in [0, 1]
    (0 = identical shape, 1 = orthogonal or zero estimate).
    """
    w_hat, w = np.ravel(estimate), np.ravel(base)
    denom = w_hat @ w_hat
    alpha = (w_hat @ w) / denom if denom > 0 else 0.0
    return float(np.sum((w - alpha * w_hat) ** 2) / np.sum(w ** 2))
