import numpy as np
import pytest
import scipy.signal

from adr.io import phantom

try:
    HAVE_DATA = (phantom.root() / "stimuli" / "GTdata_croppedToRisingEdge.mat").exists()
except Exception:
    HAVE_DATA = False


def test_scalp_channels():
    names = ["A1", "D32", "EXG1", "N-A1"]
    assert phantom.scalp_channels(names) == ["A1", "D32"]


@pytest.mark.skipif(not HAVE_DATA, reason="phantom data not available locally")
def test_brain_trial_alignment():
    trial = phantom.load_trial("Brain")
    n = len(trial.eeg)
    assert trial.eeg.shape == (n, 128) and trial.noise.shape == (n, 128) and trial.emg.shape == (n, 8)
    assert trial.ground_truth.shape == (n, 21) and trial.sfreq == 512
    # Pools tile the trial without gaps or overlap.
    pools = [phantom.pool_slice(p, trial.sfreq, n) for p in ("set_a_templates", "set_b_templates", "phantom_check")]
    assert pools[0].start == 0 and pools[-1].stop == n
    assert all(a.stop == b.start for a, b in zip(pools, pools[1:]))
    # The broadcast brain sources explain most of the scalp EEG (first 60 s).
    sos = scipy.signal.butter(4, [1, 40], "bandpass", fs=512, output="sos")
    seg = slice(0, 60 * 512)
    Y = scipy.signal.sosfiltfilt(sos, trial.eeg[seg] - trial.eeg[seg].mean(1, keepdims=True), axis=0)
    X = scipy.signal.sosfiltfilt(sos, trial.brain_sources[seg], axis=0)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - ((Y - X @ beta) ** 2).sum(0) / (Y ** 2).sum(0)
    assert np.median(r2) > 0.5
