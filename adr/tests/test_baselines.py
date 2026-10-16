import numpy as np
import pytest
import scipy.signal

from adr.baselines import ica_iclabel, mwf_auto
from adr.preprocessing import spline

FS = 128.0
NAMES = spline.biosemi64_positions()[0]


def background(rng, minutes=3):
    n = int(FS * 60 * minutes)
    pink = scipy.signal.lfilter([1], [1, -0.95], rng.standard_normal((n, 64)), axis=0)
    return pink * 2e-6


def test_iclabel_removal_rule():
    labels = ["brain", "eye blink", "muscle artifact", "other", "eye blink", "line noise"]
    probs = np.array([0.95, 0.80, 0.69, 0.99, 0.50, 0.90])
    reject = ["eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"]
    assert ica_iclabel.components_to_remove(labels, probs, reject, 0.7) == [1, 5]
    assert ica_iclabel.components_to_remove(labels, probs, reject, 0.5) == [1, 2, 4, 5]
    assert ica_iclabel.components_to_remove(labels, probs, reject, 0.9) == [5]


def test_heuristic_flags_find_blinks_and_muscle():
    rng = np.random.default_rng(0)
    eeg = background(rng)
    n = len(eeg)
    blink_at = np.arange(int(FS * 10), n - int(FS * 2), int(FS * 15))
    frontal = [NAMES.index(c) for c in ("Fp1", "Fp2", "AF7", "AF8", "Fpz")]
    for s in blink_at:
        eeg[s:s + 40, frontal] += 150e-6 * np.hanning(40)[:, None]
    emg_span = slice(int(FS * 100), int(FS * 104))
    eeg[emg_span, NAMES.index("T7")] += 40e-6 * rng.standard_normal(int(FS * 4))
    # Mechanism test with an explicit threshold (the frozen config value is
    # calibrated to Section A data, not to this toy background).
    from adr.config import load_yaml
    cfg = dict(load_yaml("baselines")["mwf_auto"], muscle_slope_threshold=-0.59)
    flags, starts = mwf_auto.heuristic_flags(eeg, FS, NAMES, cfg)
    hit_blink = [np.any(flags["ocular"][(starts <= s / FS) & (starts + 1 > s / FS)]) for s in blink_at]
    assert np.mean(hit_blink) > 0.9
    in_emg = (starts >= 100) & (starts + 1 <= 104)
    assert flags["muscle"][in_emg].all()
    assert flags["ocular"].mean() < 0.3


def test_auto_mwf_runs_and_reduces_blinks():
    rng = np.random.default_rng(1)
    clean = background(rng)
    eeg = clean.copy()
    frontal = [NAMES.index(c) for c in ("Fp1", "Fp2", "AF7", "AF8", "Fpz")]
    v = np.zeros(64); v[frontal] = 1.0; v[[NAMES.index(c) for c in ("F3", "F4", "Fz")]] = 0.4
    for s in range(int(FS * 5), len(eeg) - 100, int(FS * 6)):
        eeg[s:s + 40] += 150e-6 * np.outer(np.hanning(40), v)
    out, info = mwf_auto.run(eeg, FS, NAMES)
    assert np.linalg.norm(out - clean) < 0.5 * np.linalg.norm(eeg - clean)
    assert info["blink_mwf_rank"] >= 1
