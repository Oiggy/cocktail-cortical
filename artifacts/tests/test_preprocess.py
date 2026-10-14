import mne
import numpy as np
import pytest
from scipy.signal import welch

from artifacts import config
from artifacts.preprocessing.preprocess import detect_line_frequency, preprocess_raw, read_resampled
from artifacts.preprocessing.segments import Segment

from .synthetic import UV, blink_shape, pink_noise

NATIVE = 1024.  # lower than the real ~4096 Hz to keep the test fast; same code path


def _native_raw(duration=80., line=60., seed=0):
    """Native-rate recording with a large common-mode signal (what BioSemi's
    CMS/DRL leaves in every channel), mains noise, and a blink on the
    superior EOG electrode only."""
    rng = np.random.default_rng(seed)
    n = int(duration * NATIVE)
    t = np.arange(n) / NATIVE
    # 2 Hz: passes the 0.5 Hz high-pass, so only re-referencing can remove it
    common = 500 * UV * np.sin(2 * np.pi * 2. * t)
    names = list(config.EEG_CHANNELS) + list(config.MASTOIDS) + list(config.EOG_CHANNELS)
    # mains noise differs per channel, so re-referencing can't cancel it - only the notch can
    data = np.stack([common + pink_noise(n, rng, 5 * UV)
                     + rng.uniform(5, 30) * UV * np.sin(2 * np.pi * line * t + rng.uniform(0, 2 * np.pi))
                     for _ in names])
    blink = np.zeros(n)
    shape = blink_shape(NATIVE)
    blink[int(30 * NATIVE):int(30 * NATIVE) + len(shape)] = 150 * UV * shape
    data[names.index('EOG-LS')] += blink
    types = ['eeg'] * 66 + ['eog'] * 4
    raw = mne.io.RawArray(data, mne.create_info(names, NATIVE, types), verbose=False)
    return raw, blink


def _segments(onsets_durations):
    return [Segment(index=i, onset=o, duration=d, condition='clean', side='', fg=f'male_{i}', bg='')
            for i, (o, d) in enumerate(onsets_durations, start=1)]


def test_read_resampled_length_and_content():
    n = int(10 * NATIVE)
    t = np.arange(n) / NATIVE
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 5 * t)] * 3), mne.create_info(3, NATIVE, 'eeg'), verbose=False)
    out = read_resampled(raw, [0, 1, 2], 0, n, 256., group_size=2)
    assert out.shape == (3, 2560)
    expected = np.sin(2 * np.pi * 5 * np.arange(2560) / 256.)
    assert np.abs(out[:, 100:-100] - expected[100:-100]).max() < 0.01


@pytest.mark.parametrize('line', [50., 60.])
def test_detect_line_frequency(line):
    rng = np.random.default_rng(1)
    t = np.arange(int(20 * 256)) / 256.
    data = np.stack([pink_noise(len(t), rng) + 5 * UV * np.sin(2 * np.pi * line * t) for _ in range(4)])
    assert detect_line_frequency(data, 256.)[0] == line


def test_no_clear_line_peak():
    rng = np.random.default_rng(2)
    data = np.stack([pink_noise(20 * 256, rng) for _ in range(4)])
    assert detect_line_frequency(data, 256.)[0] is None


def test_preprocess_raw():
    raw, blink = _native_raw()
    segments = _segments([(5., 20.), (28., 20.), (55., 20.)])
    out, table, info = preprocess_raw(raw, segments, bads=['Cz'], sfreq=256., pad=4.)

    assert out.info['sfreq'] == 256.
    assert out.ch_names == list(config.EEG_CHANNELS) + ['VEOG', 'HEOG']
    assert out.n_times == 3 * 20 * 256
    assert list(table['start']) == [0, 5120, 10240] and list(table['onset']) == [0., 20., 40.]
    assert info['line_freq'] == 60. and info['line_freq_source'] == 'detected' and info['interpolated'] == ['Cz']
    assert info['reference'] == 'mastoids:A1+A2'

    eeg = out.get_data(picks='eeg')
    # mastoid reference removes the common-mode signal (~500 uV) down to noise level
    assert np.abs(eeg).max() < 100 * UV
    # the interpolated channel carries signal, not zeros
    assert eeg[config.EEG_CHANNELS.index('Cz')].std() > 0.5 * eeg.std()
    # mains noise notched out
    freqs, psd = welch(eeg, fs=256., nperseg=512)
    at_line = psd[:, np.argmin(np.abs(freqs - 60))].mean()
    nearby = psd[:, (np.abs(freqs - 60) > 3) & (np.abs(freqs - 60) < 8)].mean()
    assert at_line < 3 * nearby

    # the blink (on EOG-LS only, at 30.0 s = 2.0 s into segment 2) shows up
    # in VEOG at the right time, at full amplitude; HEOG stays quiet
    veog, heog = out.get_data(picks=['VEOG', 'HEOG'])
    peak = np.argmax(veog)
    assert abs(peak / 256. - (20. + 2. + 0.15)) < 0.02
    assert 120 * UV < veog.max() < 170 * UV
    assert np.abs(heog).max() < 50 * UV


def test_average_reference_and_bad_mastoid():
    raw, _ = _native_raw(duration=40.)
    segments = _segments([(5., 20.)])
    out, _, info = preprocess_raw(raw, segments, sfreq=256., pad=4., reference='average')
    assert info['reference'] == 'average'
    assert np.abs(out.get_data(picks='eeg').mean(axis=0)).max() < 1e-9
    _, _, info = preprocess_raw(raw, segments, bads=['A1'], sfreq=256., pad=4.)
    assert info['reference'] == 'mastoids:A2'


def test_truncated_recording():
    raw, _ = _native_raw(duration=40.)
    _, table, info = preprocess_raw(raw, _segments([(30., 20.)]), sfreq=256., pad=4.)
    assert bool(table['truncated'][0]) and info['truncated_segments'] == [1]
    assert table['duration'][0] == pytest.approx(10., abs=1 / 256)
