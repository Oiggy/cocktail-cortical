import numpy as np
import pandas as pd
import pytest

from adr.preprocessing.channels import detector_channels
from adr.synth import inject as I
from adr.synth.templates import Template

FS = 128.0


def bank(kind="ocular", n=5, seed=0):
    rng = np.random.default_rng(seed)
    return [Template(kind, rng.standard_normal((int(FS * 2), 64)) * 1e-5, FS, f"toy:{i}", "set_a_templates")
            for i in range(n)]


def eeg(minutes=5, seed=1):
    return np.random.default_rng(seed).standard_normal((int(FS * 60 * minutes), 64)) * 2e-5


def test_onsets_respect_gap_and_end():
    rng = np.random.default_rng(0)
    durations = [2.0] * 1000
    starts = I.onsets(rng, int(FS * 600), FS, durations, events_per_minute=6, min_gap_s=1.0)
    gaps = np.diff(starts) / FS
    assert np.all(gaps >= 3.0 - 1e-9)                  # 2 s event + 1 s gap
    assert starts[-1] / FS + 2.0 < 600


@pytest.mark.parametrize("sar", [-10.0, 0.0, 10.0])
def test_template_injection_hits_sar(sar):
    data = eeg()
    out = I.inject(data, FS, "ocular", sar, np.random.default_rng(2), bank=bank())
    assert len(out.log) > 5
    np.testing.assert_allclose(out.eeg - out.artifact, data)
    for row in out.log.itertuples():
        a, b = int(row.onset_s * FS), int(row.offset_s * FS)
        achieved = 20 * np.log10(np.sqrt((data[a:b] ** 2).mean(0)).mean() / np.sqrt((out.artifact[a:b] ** 2).mean(0)).mean())
        assert achieved == pytest.approx(sar, abs=1e-6)


def test_pops_hit_sar_and_avoid_cz():
    names = list(detector_channels()) + [f"x{i}" for i in range(7)]
    names[0] = "Cz"
    data = eeg()
    out = I.inject(data, FS, "pop", 0.0, np.random.default_rng(3), ch_names=names)
    assert len(out.log) > 5 and "Cz" not in set(out.log["channels"])
    for row in out.log.itertuples():
        a, b, c = int(row.onset_s * FS), int(row.offset_s * FS), names.index(row.channels)
        assert np.count_nonzero(out.artifact[a:b].any(axis=0)) == 1
        achieved = 20 * np.log10(np.sqrt((data[a:b, c] ** 2).mean()) / np.sqrt((out.artifact[a:b, c] ** 2).mean()))
        assert achieved == pytest.approx(0.0, abs=1e-6)


def test_missing_templates_raise():
    with pytest.raises(ValueError):
        I.inject(eeg(1), FS, "motion", 0.0, np.random.default_rng(0), bank=bank("ocular"))


def test_window_truth():
    log = pd.DataFrame([{"type": "muscle", "onset_s": 10.2, "offset_s": 11.0},
                        {"type": "pop", "onset_s": 20.0, "offset_s": 21.0}])
    truth = I.window_truth(log, int(FS * 30), FS, 1.0, 0.5)
    at = lambda t: truth.loc[truth.start == t].iloc[0]
    assert at(9.5)["muscle"] and at(10.5)["muscle"] and not at(11.0)["muscle"]
    assert at(19.5)["pop"] and at(19.5)["artifact"] and not at(5.0)["artifact"]


def test_phantom_source_activity_shape():
    from adr.io import phantom
    gt = np.zeros((512 * 10, 21))
    gt[512 * 3: 512 * 3 + 50, 10] = 1.0            # one blink at 3 s
    gt[512 * 6: 512 * 8, 16:20] = 0.5              # facial burst 6-8 s
    act = phantom.source_activity(gt, 512, 1.0, 0.5)
    assert len(act) == 19
    assert act.loc[act.start == 3.0, "blink"].item() and not act.loc[act.start == 5.0, "blink"].item()
    assert act.loc[act.start == 6.5, "facial"].item() and not act.loc[act.start == 2.0, "facial"].item()
