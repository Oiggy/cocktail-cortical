import numpy as np
import pytest
import scipy.signal

from adr.envelope import das as env_das
from adr.preprocessing import das


def test_bandpass_meets_matlab_spec():
    taps = das.bandpass_taps()
    assert das._meets_spec(taps, das.FS_IN)
    np.testing.assert_allclose(taps, taps[::-1])  # linear phase


def test_preprocess_eeg_shape_and_reference():
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((128 * 60, 64)) + 5.0  # common offset
    out = das.preprocess_eeg(eeg)
    assert out.shape == (32 * 60, 63)
    # Cz reference + band-pass: a signal shared by all channels vanishes
    shared = np.tile(np.sin(2 * np.pi * 4 * np.arange(128 * 60) / 128)[:, None], (1, 64))
    assert np.abs(das.preprocess_eeg(shared)).max() < 1e-10


def test_centre_frequencies():
    cf = env_das.centre_frequencies()
    assert len(cf) == 15                                   # Das et al.: 15 filters
    assert np.any(np.isclose(cf, 1000.0))                  # erbspacebw hits 1 kHz
    assert 150 <= cf[0] and cf[-1] <= 4000
    np.testing.assert_allclose(np.diff(env_das.freq_to_erb(cf)), 1.5)


def test_gammatone_unit_gain_at_centre():
    for fc, g in zip(env_das.centre_frequencies(), env_das.gammatone_bank()):
        _, h = scipy.signal.freqz(g, worN=[fc], fs=env_das.AUDIO_FS)
        assert np.isclose(np.abs(h[0]), 1.0, rtol=1e-3)


def test_envelope_tracks_modulation():
    fs = 16000
    t = np.arange(fs * 20) / fs
    modulator = 1 + 0.8 * np.sin(2 * np.pi * 4 * t)        # 4 Hz, inside 1-9 Hz
    rng = np.random.default_rng(0)
    audio = modulator * rng.standard_normal(len(t))
    env = env_das.envelope(audio, fs)
    assert len(env) == 32 * 20
    ref = modulator[:: fs // 32][: len(env)]
    core = slice(32, -32)                                   # skip filter edges
    assert np.corrcoef(env[core], ref[core])[0, 1] > 0.9


def test_section_a_subject():
    from adr.io.dataset1 import available_subjects
    if not available_subjects():
        pytest.skip("Dataset I not available locally")
    from adr.preprocessing.section_a import prepare_subject
    trials, names = prepare_subject(available_subjects()[0])
    assert len(trials) == 10 and len(names) == 63 and "Cz" not in names
    minutes = sum(len(t.envelope) for t in trials) / 32 / 60
    assert 38 < minutes < 39
    eeg = np.concatenate([t.eeg for t in trials])
    np.testing.assert_allclose(eeg.mean(0), 0, atol=1e-8)
    np.testing.assert_allclose(eeg.std(0), 1, atol=1e-8)
    assert {t.group for t in trials} <= {1, 2, 3, 4} and len({t.group for t in trials}) == 2
