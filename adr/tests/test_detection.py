import numpy as np
import pytest

from adr.detection import evaluate, features
from adr.detection import models as M
from adr.preprocessing.channels import detector_channels
from adr.synth import pops

CH = detector_channels()


def test_bands_avoid_mains_and_nyquist():
    for fs in (128, 250):
        for lo, hi in features.bands_for(fs).values():
            assert hi < fs / 2
            for mains in (50, 60, 100, 120):
                assert not (lo <= mains <= hi), (lo, hi, mains)
    assert len(features.bands_for(250)) == len(features.bands_for(128)) + 2


@pytest.mark.parametrize("fs", [128.0, 250.0])
def test_feature_shapes_and_names(fs):
    data = np.random.default_rng(0).standard_normal((int(fs * 20), len(CH))) * 1e-5
    X = features.compute(data, fs, CH)
    assert X.shape == (39, len(features.feature_names(CH, fs)))  # 20 s, 1 s windows, 0.5 s step
    assert np.isfinite(X).all() and X.dtype == np.float32


def test_features_respond_to_artifacts():
    fs = 128.0
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((int(fs * 10), len(CH))) * 1e-5
    names = features.feature_names(CH, fs)
    # blink-like: large slow deflection on frontal channels
    blink = clean.copy()
    t = np.arange(len(clean)) / fs
    for ch in ("Fp1", "Fp2", "AF7", "AF8", "Fpz"):
        blink[:, CH.index(ch)] += 2e-4 * np.exp(-((t - 5) ** 2) / 0.02)
    Xc, Xb = features.compute(clean, fs, CH), features.compute(blink, fs, CH)
    i = names.index("log_frontal_posterior_ratio")
    assert Xb[:, i].max() > Xc[:, i].max() + 0.5
    # muscle-like: broadband high-frequency burst on a temporal channel
    emg = clean.copy()
    emg[:, CH.index("T7")] += rng.standard_normal(len(clean)) * 1e-4
    j = names.index("log_power_low_gamma:T7")
    assert features.compute(emg, fs, CH)[:, j].mean() > Xc[:, j].mean() + 1


def test_pop_waveform_and_injection():
    fs = 128.0
    w = pops.waveform(256, fs, 100, 1e-4, 0.1)
    assert np.all(w[:100] == 0) and w[100] == pytest.approx(1e-4)
    k = 12  # samples after onset
    assert w[100 + k] == pytest.approx(1e-4 * np.exp(-k / fs / 0.1))
    rng = np.random.default_rng(2)
    drawn = pops.draw(rng, 1000, fs, CH, 200)
    assert all(CH[p.channel] != "Cz" for p in drawn)
    assert all(50e-6 <= abs(p.amplitude) <= 500e-6 and 0.05 <= p.tau <= 0.5 for p in drawn)
    data = np.zeros((1000, len(CH)))
    out = pops.inject(data, fs, drawn[:1])
    assert np.count_nonzero(out.any(axis=0)) == 1 and np.all(data == 0)


def test_window_and_event_metrics():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0], bool)
    s = np.array([.1, .2, .9, .8, .1, .1, .3, .7, .1, .1])
    m = evaluate.window_metrics(y, s, 0.5)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(6 / 7)
    rest = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], bool)
    e = evaluate.event_metrics(y, s >= 0.5, rest, step_s=0.5)
    assert e["events"] == 2 and e["detected"] == 1 and e["false_alarms"] == 1
    assert e["rest_hours"] == pytest.approx(5 * 0.5 / 3600)


def test_stratified_subsample_keeps_classes():
    y = np.r_[np.zeros(9000), np.ones(1000)]
    idx = M.stratified_subsample(y, 1000, np.random.default_rng(0))
    assert len(idx) == 1000 and y[idx].mean() == pytest.approx(0.1, abs=0.01)


def test_spline_reproduces_smooth_field():
    from adr.preprocessing import spline
    names, p64 = spline.biosemi64_positions()
    unit = p64 / np.linalg.norm(p64, axis=1, keepdims=True)
    field = lambda p: p[:, 0] + 0.5 * p[:, 2]          # smooth (linear) scalp field
    keep = np.arange(64) % 4 != 0
    M64 = spline.matrix(p64[keep], p64[~keep])
    np.testing.assert_allclose(M64 @ field(unit[keep]), field(unit[~keep]), atol=0.05)


def test_normalize_per_recording():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((500, 4)) * [1, 10, 0.1, 5] + [3, -7, 100, 0]
    Z = features.normalize(X)
    np.testing.assert_allclose(np.median(Z, axis=0), 0, atol=1e-6)
    q75, q25 = np.percentile(Z, [75, 25], axis=0)
    np.testing.assert_allclose((q75 - q25) / 1.349, 1, rtol=1e-3)
    # an offset/scale change of the whole recording leaves the features unchanged
    np.testing.assert_allclose(features.normalize(X * 3 + 2), Z, atol=1e-4)
    # pop examples reuse their run's statistics
    stats = features.recording_stats(X)
    np.testing.assert_allclose(features.normalize(X[:10], stats), Z[:10], atol=1e-6)
    assert features.apply_variant(X, "raw") is X
