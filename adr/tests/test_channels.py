import numpy as np
import pytest

from adr.preprocessing.channels import average_reference, detector_channels, label_electrodes, pick, resample


def test_pick_reorders_and_checks():
    data = np.arange(12).reshape(4, 3)
    out = pick(data, ["a", "b", "c"], ["c", "a"])
    np.testing.assert_array_equal(out, data[:, [2, 0]])
    with pytest.raises(KeyError):
        pick(data, ["a", "b", "c"], ["d"])


def test_average_reference():
    data = np.random.default_rng(0).standard_normal((100, 5))
    np.testing.assert_allclose(average_reference(data).mean(axis=1), 0, atol=1e-12)


@pytest.mark.parametrize("sfreq_in, sfreq_out", [(1000, 128), (1000, 250), (512, 128), (512, 250)])
def test_resample_lengths(sfreq_in, sfreq_out):
    data = np.zeros((sfreq_in * 10, 2))
    assert resample(data, sfreq_in, sfreq_out).shape == (sfreq_out * 10, 2)


def test_channel_sets():
    assert len(detector_channels()) == 57
    assert set(label_electrodes()) <= set(detector_channels())
