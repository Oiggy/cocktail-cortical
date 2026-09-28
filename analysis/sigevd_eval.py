"""
Numeric/statistical primitives for cortical_analysis_3.ipynb's raw-vs-SI-GEVD
evaluation (TRF-accuracy MSE, paired significance tests, and the attention-
decoding pipeline) - kept separate from sigevd.py (the filter itself) and
experiment_sigevd.py (the eelbrain raw-pipeline integration) so each piece
stays independently testable, matching the same split this project already
uses.

Every function here is pure numpy/scipy, with no eelbrain dependency, so it
can be validated on synthetic data without eelbrain or the real dataset -
see the docstring notes on what was actually validated this way. None of
this has been run against real data or against eelbrain's actual output
shapes; the notebook that calls these is unexecuted.
"""
import numpy as np
from scipy.stats import wilcoxon

from sigevd import lag_data, trf_estim


def relative_mse(estimate, ground_truth):
    """Relative mean squared error between one estimated TRF and a reference
    ("ground truth") TRF, both 1D arrays over the same time axis.

    ||estimate - ground_truth||^2 / ||ground_truth||^2 - scale-free, so it
    doesn't depend on the TRF's raw amplitude units.
    """
    estimate = np.asarray(estimate, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    return float(np.sum((estimate - ground_truth) ** 2) / np.sum(ground_truth ** 2))


def paired_wilcoxon(a, b, alpha=0.025):
    """Wilcoxon signed-rank test between two paired per-subject 1D arrays
    (e.g. one method's per-subject scores vs the other's).

    Returns (statistic, p, significant) - `significant` is `p < alpha`
    (default 0.025: two-comparison Bonferroni correction of 0.05).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    statistic, p = wilcoxon(a, b)
    return float(statistic), float(p), bool(p < alpha)


def per_sensor_wilcoxon_mask(a, b, alpha=0.025):
    """Per-sensor Wilcoxon signed-rank test between two paired
    (n_subjects, n_sensors) arrays - e.g. raw vs SI-GEVD unique-contribution
    maps - used to mask a group-mean difference topomap by significance
    when there is no single eelbrain `load_model_test()` result spanning
    both methods to borrow a cluster-corrected mask from (see
    cortical_analysis_3.ipynb's own notes on this).

    Returns a boolean (n_sensors,) array, True where p < alpha. A sensor
    where every subject's a-b difference is exactly zero makes
    scipy.stats.wilcoxon raise (nothing to rank) - treated as not
    significant (p=1) rather than propagating the error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_sensors = a.shape[1]
    mask = np.zeros(n_sensors, dtype=bool)
    for i in range(n_sensors):
        try:
            _, p = wilcoxon(a[:, i], b[:, i])
        except ValueError:
            p = 1.0
        mask[i] = p < alpha
    return mask


def decode_attention_accuracy(trials, fs, start, fin, lam, trial_lengths):
    """Leave-one-trial-out attended-speaker decoding accuracy, at each of
    several trial lengths (the Biesmans et al. 2017-style stimulus-
    reconstruction paradigm: fit one TRF on the attended stream, apply it
    to both streams, and pick whichever correlates better with the real
    response).

    Parameters
    ----------
    trials : list of dicts, one per trial, each with:
        'attend_stim' : (n_time,) or (n_time, n_features) attended predictor
        'ignore_stim' : (n_time,) or (n_time, n_features) ignored predictor,
            same shape as 'attend_stim'
        'resp' : (n_time, n_channels) EEG
    fs : float
        Sampling rate in Hz.
    start, fin : float
        TRF lag window, in milliseconds (same convention as
        sigevd.find_sigevd_filter).
    lam : float
        Ridge regularization parameter for the TRF fit.
    trial_lengths : sequence of float
        Window lengths, in seconds, to evaluate decoding accuracy at. Each
        held-out trial is cut into non-overlapping windows of this length
        (a trial shorter than the window contributes no window at that
        length).

    Returns
    -------
    dict {trial_length: accuracy} - accuracy is the fraction of windows,
    pooled across every held-out trial, correctly classified as attended.

    Validated on synthetic data: a response driven by the attended stream
    (convolved with a fixed kernel) plus a much larger-magnitude
    independent noise term, with an unrelated ignored stream, decodes
    above chance (0.5) at every trial length tested. Not validated against
    real EEG or real eelbrain-shaped data.
    """
    idx_s = int(np.floor(start / 1e3 * fs))
    idx_f = int(np.ceil(fin / 1e3 * fs))
    lags = np.arange(idx_s, idx_f + 1)

    def _as_2d(stim):
        stim = np.asarray(stim, dtype=float)
        return stim.reshape(-1, 1) if stim.ndim == 1 else stim

    def _lagged_centered(stim):
        lagged = lag_data(_as_2d(stim), lags)
        return lagged - lagged.mean(axis=0, keepdims=True)

    accuracy = {}
    for length_s in trial_lengths:
        window = int(round(length_s * fs))
        correct, total = 0, 0
        for held_out in range(len(trials)):
            train = [t for i, t in enumerate(trials) if i != held_out]
            pooled_stim = np.vstack([_as_2d(t['attend_stim']) for t in train])
            pooled_resp = np.vstack([t['resp'] for t in train])
            lagged = _lagged_centered(pooled_stim)
            resp_c = pooled_resp - pooled_resp.mean(axis=0, keepdims=True)
            trf = trf_estim(lagged.T, resp_c.T, lam)

            test = trials[held_out]
            n_time = len(test['resp'])
            for start_i in range(0, n_time - window + 1, window):
                sl = slice(start_i, start_i + window)
                attend_lagged = _lagged_centered(np.asarray(test['attend_stim'], dtype=float)[sl])
                ignore_lagged = _lagged_centered(np.asarray(test['ignore_stim'], dtype=float)[sl])
                resp_seg = test['resp'][sl]
                resp_seg_c = resp_seg - resp_seg.mean(axis=0, keepdims=True)
                pred_attend = attend_lagged @ trf
                pred_ignore = ignore_lagged @ trf
                r_attend = np.corrcoef(pred_attend.mean(axis=1), resp_seg_c.mean(axis=1))[0, 1]
                r_ignore = np.corrcoef(pred_ignore.mean(axis=1), resp_seg_c.mean(axis=1))[0, 1]
                if r_attend > r_ignore:
                    correct += 1
                total += 1
        accuracy[length_s] = correct / total if total else float('nan')
    return accuracy
