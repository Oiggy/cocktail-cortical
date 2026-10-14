"""Memory-safe preprocessing, one speech segment at a time.

For each of the 12 speech segments:
  1. Read the segment (plus SEGMENT_PAD seconds either side) from the lazy
     ~4096 Hz recording, a few channels at a time, downsampling each group
     straight to SFREQ with a polyphase anti-aliasing filter. Only one
     group of one segment is ever in memory at native rate.
  2. High-pass and line-noise notch on the padded segment, then trim the
     padding, which absorbs the filters' edge transients.
  3. Bipolar VEOG (above minus below the left eye) and HEOG (left minus
     right outer canthus) from the four EOG electrodes. Bipolar
     derivations cancel the BioSemi CMS/DRL reference, so they need no
     re-referencing.
The segments are then concatenated, bad scalp channels interpolated (so
every subject ends with the same 64 channels, which the detection
features need), and the scalp channels re-referenced. BioSemi records
without a fixed reference, so this happens before any feature is
computed.

Only speech segments are kept: breaks between stories are not continuous
speech listening, which is this project's target. Downstream steps must
work per segment (see the segments table), never across a boundary.
"""
import json
from fractions import Fraction

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from .. import config
from ..paths import deriv_path
from .loading import read_eelbrain_bads, read_raw
from .segments import segments_from_annotations

EOG_DERIVED = ('VEOG', 'HEOG')


def _resample_ratio(sfreq_in, sfreq_out):
    ratio = Fraction(sfreq_out / sfreq_in).limit_denominator(10000)
    return ratio.numerator, ratio.denominator


def read_resampled(raw, picks, start, stop, sfreq_out, group_size=8):
    """Samples [start, stop) of `picks` from a lazy Raw, downsampled to
    sfreq_out, reading `group_size` channels at a time."""
    up, down = _resample_ratio(raw.info['sfreq'], sfreq_out)
    blocks = []
    for i in range(0, len(picks), group_size):
        block = raw.get_data(picks=list(picks[i:i + group_size]), start=start, stop=stop)
        blocks.append(resample_poly(block, up, down, axis=1) if (up, down) != (1, 1) else block)
    return np.concatenate(blocks, axis=0)


def detect_line_frequency(data, sfreq, candidates=(50., 60.), min_ratio=3.):
    """Mains frequency: the candidate with the strongest spectral peak
    relative to its neighbouring bins, median across channels. None if no
    candidate stands at least `min_ratio` above its neighbours - without a
    clear peak the choice would be noise."""
    freqs, psd = welch(data, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
    psd = np.median(psd, axis=0)
    scores = {}
    for line in candidates:
        if line >= sfreq / 2:
            continue
        distance = np.abs(freqs - line)
        neighbours = psd[(distance > 2) & (distance < 6)]
        scores[line] = float(psd[np.argmin(distance)] / np.median(neighbours))
    best = max(scores, key=scores.get)
    return (best if scores[best] >= min_ratio else None), scores


def _filter(data, sfreq, highpass, line_freq):
    data = mne.filter.filter_data(data, sfreq, l_freq=highpass, h_freq=None, verbose=False)
    if line_freq:
        notches = np.arange(line_freq, sfreq / 2, line_freq)
        data = mne.filter.notch_filter(data, sfreq, notches, verbose=False)
    return data


def preprocess_raw(raw, segments, bads=(), sfreq=config.SFREQ, highpass=config.HIGHPASS,
                   reference=config.REFERENCE, line_freq=config.LINE_FREQ, pad=config.SEGMENT_PAD,
                   group_size=8):
    """Preprocess a lazy native-rate Raw into one SFREQ Raw of speech segments.

    Returns (raw_out, segments_df, info): raw_out has the 64 scalp
    channels plus VEOG and HEOG; segments_df records where each segment
    landed (onset/duration in raw_out's time, orig_onset in the original
    recording's); info records every choice made (for the JSON sidecar).
    """
    native = raw.info['sfreq']
    picks = list(config.EEG_CHANNELS) + list(config.MASTOIDS) + list(config.EOG_CHANNELS)
    missing = [ch for ch in picks if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"recording is missing channels: {missing}")
    n_eeg, n_mast = len(config.EEG_CHANNELS), len(config.MASTOIDS)
    total = raw.n_times / native

    chunks, rows, start_out = [], [], 0
    use_line, line_source, line_scores = line_freq, 'config', None
    for segment in segments:
        t0 = segment.onset - raw.first_time
        t1 = min(t0 + segment.duration, total)
        pad0, pad1 = min(pad, t0), min(pad, total - t1)
        start = int(round((t0 - pad0) * native))
        stop = int(round((t1 + pad1) * native))
        data = read_resampled(raw, picks, start, stop, sfreq, group_size)

        if line_source == 'config' and line_freq is None:
            # decided once, from the first segment: an explicit setting, else a
            # clear spectral peak, else the BIDS PowerLineFrequency, else no notch
            use_line, line_scores = detect_line_frequency(data[:n_eeg], sfreq)
            line_source = 'detected'
            if use_line is None:
                use_line = raw.info.get('line_freq')
                line_source = 'bids' if use_line else 'none'
        data = _filter(data, sfreq, highpass, use_line)

        first = int(round(pad0 * sfreq))
        n = int(round((t1 - t0) * sfreq))
        data = data[:, first:first + n]

        eog = dict(zip(config.EOG_CHANNELS, data[n_eeg + n_mast:]))
        veog = eog[config.VEOG_PAIR[0]] - eog[config.VEOG_PAIR[1]]
        heog = eog[config.HEOG_PAIR[0]] - eog[config.HEOG_PAIR[1]]
        chunks.append(np.vstack([data[:n_eeg + n_mast], veog, heog]))
        rows.append({
            'index': segment.index, 'condition': segment.condition, 'side': segment.side,
            'fg': segment.fg, 'bg': segment.bg, 'orig_onset': segment.onset,
            'onset': start_out / sfreq, 'duration': n / sfreq, 'start': start_out, 'stop': start_out + n,
            'truncated': bool(t1 - t0 < segment.duration - 1. / sfreq),
        })
        start_out += n

    data = np.concatenate(chunks, axis=1)
    ch_names = list(config.EEG_CHANNELS) + list(config.MASTOIDS) + list(EOG_DERIVED)
    ch_types = ['eeg'] * (n_eeg + n_mast) + ['eog'] * 2
    out = mne.io.RawArray(data, mne.create_info(ch_names, sfreq, ch_types), verbose=False)

    montage = mne.channels.read_custom_montage(config.MONTAGE_FILE)
    out.set_montage(montage, on_missing='ignore', verbose=False)

    bads = sorted(set(bads))
    eeg_bads = [ch for ch in bads if ch in config.EEG_CHANNELS]
    eog_bads = [ch for ch in bads if ch in config.EOG_CHANNELS]
    good_mastoids = [ch for ch in config.MASTOIDS if ch not in bads]
    # A bad mastoid is marked too, so it can't leak into the interpolation
    # of other channels; it's never used as reference either way.
    out.info['bads'] = eeg_bads + [ch for ch in config.MASTOIDS if ch not in good_mastoids]
    if out.info['bads']:
        out.interpolate_bads(reset_bads=True, verbose=False)

    if reference == 'mastoids' and good_mastoids:
        out.set_eeg_reference(ref_channels=good_mastoids, verbose=False)
        used_reference = 'mastoids:' + '+'.join(good_mastoids)
        out.drop_channels(list(config.MASTOIDS))
    else:
        out.drop_channels(list(config.MASTOIDS))  # average over the 64 scalp sites only
        out.set_eeg_reference('average', verbose=False)
        used_reference = 'average' if reference == 'average' else 'average (both mastoids bad)'

    segments_df = pd.DataFrame(rows)
    out.set_annotations(mne.Annotations(
        onset=segments_df['onset'], duration=segments_df['duration'],
        description=[f'segment_{i:02d}_{c}' for i, c in zip(segments_df['index'], segments_df['condition'])],
    ))

    info = {
        'native_sfreq': native, 'sfreq': sfreq, 'highpass': highpass,
        'line_freq': use_line, 'line_freq_source': line_source, 'line_freq_scores': line_scores,
        'reference': used_reference, 'interpolated': eeg_bads,
        'eog_bads': eog_bads, 'veog': '-'.join(config.VEOG_PAIR), 'heog': '-'.join(config.HEOG_PAIR),
        'n_segments': len(segments_df), 'truncated_segments': segments_df.loc[segments_df['truncated'], 'index'].tolist(),
        'mne_version': mne.__version__,
    }
    return out, segments_df, info


def preprocess_subject(subject, save=True, extra_bads=(), **kwargs):
    raw = read_raw(subject)
    segments = segments_from_annotations(raw.annotations)
    bads = set(read_eelbrain_bads(subject)) | set(extra_bads)
    out, segments_df, info = preprocess_raw(raw, segments, bads=bads, **kwargs)
    info['subject'] = subject
    if save:
        out.save(deriv_path(subject, 'preproc', 'eeg', 'fif'), fmt='single', overwrite=True, verbose=False)
        segments_df.to_csv(deriv_path(subject, 'preproc', 'segments', 'tsv'), sep='\t', index=False)
        deriv_path(subject, 'preproc', 'eeg', 'json').write_text(json.dumps(info, indent=2))
    return out, segments_df, info


def load_preprocessed(subject, rebuild=False):
    """(raw, segments_df, info) from the cache, preprocessing first if needed."""
    fif = deriv_path(subject, 'preproc', 'eeg', 'fif', mkdir=False)
    if rebuild or not fif.exists():
        return preprocess_subject(subject)
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    segments_df = pd.read_csv(deriv_path(subject, 'preproc', 'segments', 'tsv', mkdir=False), sep='\t', keep_default_na=False)
    info = json.loads(deriv_path(subject, 'preproc', 'eeg', 'json', mkdir=False).read_text())
    return raw, segments_df, info
