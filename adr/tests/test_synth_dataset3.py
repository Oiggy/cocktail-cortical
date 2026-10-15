import numpy as np
import pytest

from adr.preprocessing.das import CZ_INDEX, preprocess_eeg
from adr.synth import dataset3 as d3
from adr.trf import ridge
from adr.trf.metrics import relative_mse

FS = 32.0


def test_relative_mse():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((14, 63))
    assert relative_mse(3.7 * base, base) == pytest.approx(0.0, abs=1e-12)   # scale-free
    assert relative_mse(np.zeros_like(base), base) == pytest.approx(1.0)
    other = rng.standard_normal((14, 63))
    other -= (other.ravel() @ base.ravel()) / (base.ravel() @ base.ravel()) * base
    assert relative_mse(other, base) == pytest.approx(1.0)                  # orthogonal


def test_base_trfs_rank_one():
    rng = np.random.default_rng(1)
    names = [f"ch{i}" for i in range(63)]
    wl, wr = rng.standard_normal((14, 63)), rng.standard_normal((14, 63))
    base = d3.base_trfs(wl, wr, names, reference_lag=8, base_channel="ch13")
    for ear, w in (("L", wl), ("R", wr)):
        assert np.linalg.matrix_rank(base[ear]) == 1
        np.testing.assert_allclose(base[ear][8], w[8])        # real topography at the reference lag
        np.testing.assert_allclose(base[ear][:, 13], w[:, 13])  # real TRF at the base channel


def test_cz_insert_survives_rereference():
    x63 = np.random.default_rng(2).standard_normal((1000, 63))
    x64 = d3.insert_cz(x63)
    assert x64.shape == (1000, 64) and np.all(x64[:, CZ_INDEX] == 0)
    np.testing.assert_allclose(np.delete(x64 - x64[:, [CZ_INDEX]], CZ_INDEX, axis=1), x63)


def test_scale_to_snr_hits_target():
    rng = np.random.default_rng(3)
    signal, noise = rng.standard_normal((128 * 60, 64)), 5 * rng.standard_normal((128 * 60, 64))
    for target in (0.0, -12.5, -25.0):
        scaled, _ = d3.scale_to_snr(signal, noise, target)
        assert d3.snr_db(scaled, noise) == pytest.approx(target, abs=1e-6)


def _band_limited(rng, n):
    from adr.preprocessing.das import bandpass, decimate
    return decimate(bandpass(rng.standard_normal(n * 4)))


def test_known_trf_recovered_at_high_snr():
    """Synthesis -> Das preprocessing -> ridge TRF recovers the base TRF (plan §15)."""
    rng = np.random.default_rng(4)
    lags = ridge.lags_for(0, 0.4, FS)
    shape = np.sin(np.linspace(0, 2 * np.pi, len(lags))) * np.hanning(len(lags))
    base = {"L": np.outer(shape, rng.standard_normal(63))}
    envelopes = [_band_limited(rng, 32 * 300) for _ in range(4)]    # 4 x 5 min
    n128 = sum(len(e) for e in envelopes) * 4
    noise = rng.standard_normal((n128, 64)) * rng.uniform(5, 20, 64)  # channel-dependent amplitude
    source = d3.NoiseSource(
        subject="toy", noise=noise, envelopes=envelopes,
        groups32=np.zeros(n128 // 4, int), trial32=np.zeros(n128 // 4, int),
        ch_names=tuple(f"ch{i}" for i in range(64)), noise_sd=d3.analysis_space(noise).std(axis=0),
    )
    errors = {}
    for snr in (20.0, -20.0):
        rec = d3.synthesize(source, "L", snr, base, lags)
        assert d3.snr_db(rec.clean, noise) == pytest.approx(snr, abs=1e-6)
        eeg = preprocess_eeg(rec.eeg)
        # Standardise each channel by its noise sd, which is what the
        # analysis z-score converges to in the plan's noise-dominated range.
        eeg = (eeg - eeg.mean(0)) / source.noise_sd
        env = rec.envelope[: len(eeg)]
        trf = ridge.fit([(env - env.mean()) / env.std()], [eeg], FS, 0, 0.4, "das")
        errors[snr] = relative_mse(trf.weights, base["L"])
    assert errors[20.0] < 0.02
    assert errors[-20.0] > errors[20.0]
