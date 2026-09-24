"""
Stimulus-Informed Generalized Eigenvalue Decomposition (SI-GEVD) spatial filter.

Ported from the original MATLAB implementation:
https://github.com/exporl/si-gevd (Das et al., 6/6/2019)
find_sigevd_filter.m + trfestim.m + lag_data.m

Paper: de Cheveigne et al. (2019), "Stimulus-aware spatial filtering for
single-trial neural response and temporal response function estimation in
high-density EEG with applications in auditory research", NeuroImage.
https://doi.org/10.1016/j.neuroimage.2019.116211

Generalized here to a list of (stim, resp) pairs rather than a single
left/right split, so the same function covers every condition in this
project:
  - clean:           one pair - [(attend_stim, resp)] (the paper's own
    single-TRF/CCA-equivalent case, since there is no second stream)
  - diotic/binaural: two pairs sharing the *same* EEG - [(attend_stim,
    resp), (ignore_stim, resp)] (both streams present in the same samples)
  - dichotic:        two pairs on *disjoint* EEG - [(stim_L, resp_L),
    (stim_R, resp_R)] (each trial belongs to exactly one side)

Fitting (`find_sigevd_filter`) only produces the filter; nothing is
denoised until `apply_sigevd_filter` is called separately.
"""
import numpy as np
from scipy.linalg import eigh


def lag_data(data, lags):
    """Build a lagged design matrix via circular shift + zero-padding.

    Parameters
    ----------
    data : (n_time, n_channels) array
    lags : sequence of int
        Lags in samples (can be negative, zero, positive).

    Returns
    -------
    (n_time, n_channels * len(lags)) array
    """
    data = np.asarray(data, dtype=float)
    n_time, n_ch = data.shape
    lagged = np.zeros((n_time, n_ch * len(lags)))
    for i, lag in enumerate(lags):
        shifted = np.roll(data, lag, axis=0)
        if lag < 0:
            shifted[lag:] = 0
        elif lag > 0:
            shifted[:lag] = 0
        lagged[:, i * n_ch:(i + 1) * n_ch] = shifted
    return lagged


def trf_estim(lagged_stim, resp, lam):
    """Ridge-regularized forward TRF.

    Parameters
    ----------
    lagged_stim : (n_lag_features, n_time) array
    resp : (n_channels, n_time) array
    lam : float
        Ridge regularization parameter.

    Returns
    -------
    (n_lag_features, n_channels) array
    """
    xxt = lagged_stim @ lagged_stim.T
    xyt = lagged_stim @ resp.T
    xxt = xxt + np.eye(xxt.shape[0]) * lam * np.max(np.abs(xxt))
    return np.linalg.solve(xxt, xyt)


def find_sigevd_filter(condition_pairs, full_resp, fs, start, fin, no_of_comps, lam):
    """Fit the SI-GEVD spatial filter from one or more stimulus-following conditions.

    Parameters
    ----------
    condition_pairs : list of (stim, resp) tuples
        One entry per condition whose stimulus-following response should
        be averaged into the bias covariance R_xx (see module docstring
        for the three shapes this project uses).
        `stim`: (n_time,) or (n_time, 1) array, not yet lagged.
        `resp`: (n_time, n_channels) array, same n_time as its `stim`.
    full_resp : (n_time, n_channels) array
        EEG used for the R_yy ("everything") covariance - the single
        shared `resp` for clean/diotic/binaural, or the concatenation of
        resp_L + resp_R for dichotic.
    fs : float
        Sampling rate in Hz.
    start, fin : float
        Start/stop of the TRF lag window, in milliseconds.
    no_of_comps : int
        Number of GEVD components (k) to keep.
    lam : float
        Ridge regularization parameter for each condition's TRF fit.

    Returns
    -------
    p_k : (n_channels, no_of_comps) array
        Top generalized eigenvectors.
    w : (n_channels, n_channels) array
        Full denoising filter - see `apply_sigevd_filter`.
    stim_lags : (n_lags,) int array
        Lag indices (in samples) used.
    """
    idx_s = int(np.floor(start / 1e3 * fs))
    idx_f = int(np.ceil(fin / 1e3 * fs))
    stim_lags = np.arange(idx_s, idx_f + 1)

    r_list = []
    for stim, resp in condition_pairs:
        stim = np.asarray(stim, dtype=float).reshape(-1, 1)
        resp = np.asarray(resp, dtype=float)

        lagged = lag_data(stim, stim_lags)
        lagged = lagged - lagged.mean(axis=0, keepdims=True)
        resp_c = resp - resp.mean(axis=0, keepdims=True)

        lagged_t, resp_t = lagged.T, resp_c.T  # (features, time), (channels, time)
        trf = trf_estim(lagged_t, resp_t, lam)  # (n_lag_features, n_channels)
        x_hat = trf.T @ lagged_t  # (n_channels, n_time)
        r_list.append((x_hat @ x_hat.T) / x_hat.shape[1])

    r_xx = sum(r_list) / len(r_list)
    r_xx = (r_xx + r_xx.T) / 2  # defensive symmetrization

    full_resp = np.asarray(full_resp, dtype=float)
    full_resp_c = full_resp - full_resp.mean(axis=0, keepdims=True)
    r_yy = (full_resp_c.T @ full_resp_c) / full_resp_c.shape[0]
    # Average-referenced/ICA-cleaned EEG is often rank-deficient, which
    # leaves r_yy only positive *semi*-definite and breaks the Cholesky
    # factorization eigh() needs for the generalized eigenproblem.
    r_yy = r_yy + np.eye(r_yy.shape[0]) * (1e-8 * np.trace(r_yy) / r_yy.shape[0])

    eigenvalues, eigenvectors = eigh(r_xx, r_yy)  # ascending
    order = np.argsort(eigenvalues.real)[::-1]
    eigenvectors = eigenvectors[:, order]

    p_k = eigenvectors[:, :no_of_comps]
    q_full = np.linalg.inv(eigenvectors).T
    q_k = q_full[:, :no_of_comps]
    w = p_k @ q_k.T

    return p_k, w, stim_lags


def apply_sigevd_filter(resp_test, w):
    """Denoise EEG with an already-fit SI-GEVD filter.

    This is the actual denoising step - `find_sigevd_filter` only produces
    the filter (`w`); nothing is denoised until this is called.

    Parameters
    ----------
    resp_test : (n_time, n_channels) array
        EEG to denoise - any condition, any subject the filter was fit for.
    w : (n_channels, n_channels) array
        From `find_sigevd_filter`.

    Returns
    -------
    (n_time, n_channels) array, same shape as `resp_test`.
    """
    return np.asarray(resp_test, dtype=float) @ w


def cross_validate_sigevd(trials, fs, start, fin, comps_grid, lam_grid):
    """Leave-one-trial-out cross-validation for no_of_comps/lam.

    Matches the original paper's methodology (choose the number of
    components/regularization that maximizes held-out performance, rather
    than fixing them). On real data, score by correlation between a
    TRF-predicted response and the actual (denoised) EEG on the held-out
    trial; here `trials` carries whatever the caller wants scored against
    (see `score_fn`).

    Parameters
    ----------
    trials : list of dicts, one per held-out-able trial/segment, each with:
        'condition_pairs': list of (stim, resp) pairs for THIS trial only.
        'full_resp': this trial's EEG for R_yy.
        'score_fn': callable(denoised_resp) -> float, higher is better
            (e.g. correlation with a TRF-predicted response fit on the
            training trials).
    comps_grid, lam_grid : sequences of candidate values to try.

    Returns
    -------
    best_comps, best_lam : the grid point with the best average
        leave-one-out score.
    scores : dict {(comps, lam): mean_score} for every grid point.
    """
    n_trials = len(trials)
    scores = {}
    for comps in comps_grid:
        for lam in lam_grid:
            trial_scores = []
            for held_out in range(n_trials):
                train_trials = [t for i, t in enumerate(trials) if i != held_out]
                pooled_pairs = [pair for t in train_trials for pair in t['condition_pairs']]
                pooled_resp = np.vstack([t['full_resp'] for t in train_trials])

                _, w, _ = find_sigevd_filter(pooled_pairs, pooled_resp, fs, start, fin, comps, lam)

                test = trials[held_out]
                denoised = apply_sigevd_filter(test['full_resp'], w)
                trial_scores.append(test['score_fn'](denoised))
            scores[(comps, lam)] = float(np.mean(trial_scores))

    best_comps, best_lam = max(scores, key=scores.get)
    return best_comps, best_lam, scores
