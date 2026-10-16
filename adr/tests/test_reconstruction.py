import numpy as np
import pytest
import scipy.signal

from adr.reconstruction import mwf

FS = 128.0


def colored_noise(rng, n, c):
    b, a = scipy.signal.butter(2, 30, fs=FS)
    mix = rng.standard_normal((c, c)) * 0.3 + np.eye(c)
    return scipy.signal.lfilter(b, a, rng.standard_normal((n, c)), axis=0) @ mix


def test_mwf_removes_rank_one_artifact():
    """Plan §4: validate the MWF on a known rank-1 artifact before use."""
    rng = np.random.default_rng(0)
    n, c = int(FS * 120), 16
    neural = colored_noise(rng, n, c)
    mask = np.zeros(n, bool)
    for start in range(500, n - 300, 1500):
        mask[start:start + 250] = True
    v = rng.standard_normal(c)
    source = np.where(mask, 8 * np.sin(2 * np.pi * 2 * np.arange(n) / FS), 0.0)
    y = neural + np.outer(source, v)
    model = mwf.fit(y, mask)
    cleaned = model.clean(y)
    err_before = np.linalg.norm((y - neural)[mask]) / np.linalg.norm(neural[mask])
    err_after = np.linalg.norm((cleaned - neural)[mask]) / np.linalg.norm(neural[mask])
    assert model.eigenvalues[0] > 10 and model.rank >= 1
    assert err_after < 0.15 * err_before          # artifact largely removed
    # the estimated artifact spatial pattern matches v
    est = model.artifact(y)[mask]
    u = np.linalg.svd(est, full_matrices=False)[2][0]
    assert abs(u @ v) / np.linalg.norm(v) > 0.99


def test_mwf_rank_and_delay_options():
    rng = np.random.default_rng(1)
    n, c = int(FS * 60), 8
    y = colored_noise(rng, n, c)
    mask = np.zeros(n, bool); mask[1000:2000] = True
    y[mask] += np.outer(np.sin(np.arange(mask.sum()) / 5), rng.standard_normal(c)) * 5
    m1 = mwf.fit(y, mask, rank=1)
    assert m1.rank == 1
    md = mwf.fit(y, mask, delay=2)
    assert md.weights.shape == (c * 5, c)
    assert mwf.stack_lags(y, 2).shape == (n, c * 5)
    # with no artifact, the filter leaves clean data almost untouched
    m0 = mwf.fit(colored_noise(rng, n, c), mask)
    assert np.abs(m0.artifact(y)).mean() < 0.2 * np.abs(y).mean()


def test_frontal_ridge_removes_propagated_eog():
    from adr.preprocessing.channels import detector_channels
    from adr.reconstruction import ridge_frontal
    rng = np.random.default_rng(2)
    names = list(detector_channels())
    n = int(FS * 60)
    neural = rng.standard_normal((n, len(names)))
    eog = np.zeros(n); mask = np.zeros(n, bool)
    for s in range(300, n - 200, 900):
        eog[s:s + 100] = 40 * np.hanning(100); mask[s:s + 100] = True
    gains = rng.uniform(0.1, 1.0, len(names))
    ref = [names.index(c) for c in ("Fp1", "Fp2", "AF7", "AF8")]
    gains[ref] = 1.0
    y = neural + np.outer(eog, gains)
    out, beta = ridge_frontal.correct(y, mask, names)
    others = [i for i in range(len(names)) if i not in ref]
    before = np.abs((y - neural)[np.ix_(mask, others)]).mean()
    after = np.abs((out - neural)[np.ix_(mask, others)]).mean()
    assert after < 0.25 * before
    np.testing.assert_array_equal(out[~mask], y[~mask])        # only flagged samples change
    np.testing.assert_array_equal(out[:, ref], y[:, ref])      # references untouched


def test_spatial_interp_recovers_corrupted_channel():
    from adr.preprocessing.channels import detector_channels
    from adr.reconstruction import spatial_interp as si
    rng = np.random.default_rng(3)
    names = list(detector_channels())
    _, pos = __import__("adr.preprocessing.spline", fromlist=["x"]).biosemi64_positions()
    allnames = __import__("adr.preprocessing.spline", fromlist=["x"]).biosemi64_positions()[0]
    p = pos[[allnames.index(c) for c in names]]
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    # smooth scalp field: a few sources with broad topographies + small sensor noise
    n = int(FS * 60)
    topo = np.stack([np.exp(-np.linalg.norm(p - p[i], axis=1) ** 2 / 0.5) for i in rng.choice(len(names), 6)], 1)
    field = rng.standard_normal((n, 6)) @ topo.T + 0.05 * rng.standard_normal((n, len(names)))
    clean_mask = np.ones(n, bool); clean_mask[2000:3000] = False
    model = si.fit(field, clean_mask, names, n_neighbours=8)
    t7 = names.index("T7")
    pred = model.predict(field[2000:3000], [t7])[:, 0]
    assert np.corrcoef(pred, field[2000:3000, t7])[0, 1] > 0.95
    # contamination scoring flags the corrupted channel in the corrupted window
    corrupted = field.copy(); corrupted[2048:2176, t7] += 20 * rng.standard_normal(128)
    starts = np.arange(0, n - 128 + 1, 64)
    scores = si.window_scores(corrupted, FS, starts, 128)
    bad = si.bad_channels(scores, scores[starts < 1900], 3.0)
    hit = np.flatnonzero(starts == 2048)[0]
    assert bad[hit, t7] and bad[hit].sum() <= 3


def test_router_only_touches_flagged_samples_and_tapers():
    from adr.preprocessing import spline
    from adr.reconstruction import router
    rng = np.random.default_rng(4)
    names = spline.biosemi64_positions()[0]
    n = int(FS * 120)
    eeg = colored_noise(rng, n, 64)
    starts = np.arange(0, n / FS - 1.0 + 1e-9, 0.5)
    flags = {c: np.zeros(len(starts), bool) for c in router.CLASSES}
    flags["ocular"][[20, 21, 60, 61]] = True
    flags["muscle"][[100, 101, 150]] = True
    v = rng.standard_normal(64)
    art = np.zeros_like(eeg)
    om = router.mask_from_windows(flags["ocular"], starts, 1.0, n, FS)
    art[om] = np.outer(10 * np.hanning(om.sum()), v)
    mm = router.mask_from_windows(flags["muscle"], starts, 1.0, n, FS)
    t7 = names.index("T7")
    art[mm, t7] += 15 * rng.standard_normal(mm.sum())
    res = router.route(eeg + art, FS, names, flags, starts, 1.0)
    union = om | mm
    np.testing.assert_array_equal(res.eeg[~union], (eeg + art)[~union])        # untouched outside
    err_before = np.linalg.norm(art[union]); err_after = np.linalg.norm((res.eeg - eeg)[union])
    assert err_after < 0.5 * err_before
    assert res.info["fraction_modified"] == pytest.approx(union.mean())
    w = router.taper_weights(np.r_[np.zeros(10, bool), np.ones(40, bool), np.zeros(10, bool)], 8)
    assert w[9] == 0 and 0 < w[10] < 0.1 and w[30] == 1 and w[50] == 0


def test_over_cleaning_cap_keeps_top_windows():
    from adr.reconstruction import router
    starts = np.arange(0, 99.0 + 1e-9, 1.0)          # non-overlapping 1 s windows, 100 s
    flags = {"ocular": np.ones(100, bool)}
    scores = {"ocular": np.linspace(0, 1, 100)}
    capped = router.cap_windows(flags, scores, starts, 1.0, int(FS * 100), FS, 0.3)
    assert capped["ocular"].sum() == 30 and capped["ocular"][-30:].all()


def test_single_mwf_baseline():
    from adr.reconstruction import router
    rng = np.random.default_rng(5)
    n = int(FS * 60)
    eeg = colored_noise(rng, n, 16)
    mask = np.zeros(n, bool); mask[1000:2500] = True
    art = np.zeros_like(eeg); art[mask] = np.outer(np.sin(np.arange(mask.sum()) / 4) * 6, rng.standard_normal(16))
    res = router.single_mwf(eeg + art, mask, taper_s=0)
    assert np.linalg.norm((res.eeg - eeg)[mask]) < 0.2 * np.linalg.norm(art[mask])
    np.testing.assert_array_equal(res.eeg[~mask], (eeg + art)[~mask])


def test_eig_threshold_rank_and_native_mask():
    from adr.reconstruction import check
    rng = np.random.default_rng(6)
    n, c = int(FS * 60), 8
    y = colored_noise(rng, n, c)
    mask = np.zeros(n, bool); mask[1000:2500] = True
    y[mask] += np.outer(np.sin(np.arange(mask.sum()) / 4) * 6, rng.standard_normal(c))
    m = mwf.fit(y, mask, rank="eig>4")
    assert m.rank == int(np.sum(m.eigenvalues > 4)) and 1 <= m.rank < c
    eeg = rng.standard_normal((int(FS * 10), 4)) * 1e-5
    eeg[600:620, 2] += 5e-4                       # 500 uV transient
    native = check.native_transient_mask(eeg, FS, 200)
    assert native[610] and not native[100] and native.mean() < 0.3
    assert not check.native_transient_mask(eeg, FS, None).any()
