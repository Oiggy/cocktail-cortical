import numpy as np
import pytest

from adr.trf import ridge

FS = 32


def toy(seed=0, n_trials=4, n=3000):
    rng = np.random.default_rng(seed)
    lags = ridge.lags_for(0, 0.4, FS)
    true = np.zeros((len(lags), 3))
    true[3, 0], true[6, 1], true[:, 2] = 1.0, -0.5, np.hanning(len(lags))
    stims = [rng.standard_normal(n) for _ in range(n_trials)]
    resp = [ridge.lag_matrix(s, lags) @ true + 0.5 * rng.standard_normal((n, 3)) for s in stims]
    return stims, resp, true, lags


def test_lags():
    lags = ridge.lags_for(0, 0.4, FS)
    assert lags[0] == 0 and lags[-1] == 13 and len(lags) == 14  # Das et al.: 14 samples


def test_recovers_known_trf():
    stims, resp, true, lags = toy()
    trf = ridge.fit(stims, resp, FS, 0, 0.4, lam=0.2)
    np.testing.assert_allclose(trf.weights / FS, true, atol=0.03)
    assert trf.lags[np.argmax(trf.weights[:, 0])] == 3


def test_prediction_matches_signal():
    stims, resp, true, lags = toy()
    trf = ridge.fit(stims, resp, FS, 0, 0.4, lam=0.2)
    clean = ridge.lag_matrix(stims[0], lags) @ true
    assert np.corrcoef(trf.predict(stims[0])[:, 0], clean[:, 0])[0, 1] > 0.99


def test_matches_mtrf():
    mtrf = pytest.importorskip("mtrf")
    stims, resp, _, _ = toy()
    for lam in (0.2, 100.0):
        model = mtrf.TRF(direction=1, kind="multi", zeropad=True)
        model.train([s[:, None] for s in stims], resp, FS, 0, 0.4, lam, verbose=False)
        ours = ridge.fit(stims, resp, FS, 0, 0.4, lam)
        np.testing.assert_allclose(ours.weights, model.weights[0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(ours.bias * FS, np.ravel(model.bias), rtol=1e-8, atol=1e-10)


def test_das_lambda_smooths():
    stims, resp, _, lags = toy()
    xtx, _ = ridge.covariances(stims, resp, lags)
    lam = ridge.das_lambda(xtx, FS)
    assert np.isclose(lam * FS, np.abs(xtx[1:, 1:]).max())
    loose = ridge.fit(stims, resp, FS, 0, 0.4, 0.2)
    das = ridge.fit(stims, resp, FS, 0, 0.4, "das")
    assert np.abs(das.weights).max() < np.abs(loose.weights).max()  # shrinks
