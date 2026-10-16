"""Dataset III synthesis on Dataset I v2.0 (PROJECT_PLAN.md §9.2, §9.3 steps 1-5).

Base TRFs: the C5 long-term TRF shape (one for attend-left, one for
attend-right), scaled per channel by that channel's long-term TRF value at
the reference lag - a rank-1 TRF per condition.

For one noise subject, the attended envelopes of that subject's HRTF trials
(~38.5 min, 32 Hz) are convolved with W_baseL to give X_L and with W_baseR
to give X_R ("38 min simulated per condition", Das et al. 2020), each
trial separately so no lag crosses a trial boundary. X is upsampled to
128 Hz (it is band-limited to 1-9 Hz). The noise is the same subject's
concatenated 128 Hz HRTF EEG, time-reversed, and is added to both
conditions, giving two 64-channel recordings per noise subject. Native
transients of the "clean" base are cut out of the noise first (open item
13, option b; `native_transient_ptp_uv` in configs/section_a.yaml).

X lives in the Cz-referenced space (the TRFs were fitted on Cz-referenced
EEG), so it is added with Cz = 0: after the pipeline's Cz re-referencing it
is recovered exactly. The base TRFs come from z-scored EEG, so each channel
of X is multiplied by that channel's noise standard deviation (analysis
space) before it is added: once the analysis z-scores the (noise-dominated)
EEG, the base topography is recovered rather than one distorted by how much
each channel's background EEG varies. For a target SNR the signal is scaled and the noise
keeps its real amplitude, so the background stays at realistic microvolt
levels for the amplitude-based detector features.

SNR = 20 log10(mean_ch rms(X) / mean_ch rms(noise)), both 1-9 Hz
band-passed and measured in the analysis space (Cz reference, 63
channels), as in Das et al.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal
import yaml

from adr.config import ADR_ROOT, load_paths, load_yaml
from adr.envelope.das import wav_envelope
from adr.io.dataset1 import load_subject, presentation_groups
from adr.preprocessing.das import CZ_INDEX, FS_IN, FS_OUT, bandpass
from adr.preprocessing.section_a import attended_stimulus
from adr.trf.ridge import lag_matrix

UPSAMPLE = int(FS_IN / FS_OUT)  # 32 -> 128 Hz
CONDITIONS = ("L", "R")


def base_trfs(w_left: np.ndarray, w_right: np.ndarray, ch_names, reference_lag: int,
              base_channel: str = "C5") -> dict[str, np.ndarray]:
    """W_baseL / W_baseR (lags x channels), rank-1: shape(C5) x amplitude(ref lag).

    The C5 shape is normalised by its own value at the reference lag, so
    W_base equals the long-term TRF at C5 (all lags) and at the reference
    lag (all channels). Any other normalisation differs only by a global
    factor, which SNR scaling and the relative MSE ignore.
    """
    c = list(ch_names).index(base_channel)
    out = {}
    for ear, w in (("L", w_left), ("R", w_right)):
        shape = w[:, c] / w[reference_lag, c]
        out[ear] = np.outer(shape, w[reference_lag])
    return out


def load_base_trfs() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Base TRFs from the milestone-3 long-term TRFs (results/section_a/long_term_trf.npz)."""
    cfg = load_yaml("section_a")
    lt = np.load(load_paths()["results"] / "section_a" / "long_term_trf.npz")
    names = [str(n) for n in lt["ch_names"]]
    base = base_trfs(lt["w_left"], lt["w_right"], names, int(lt["reference_lag"]), cfg["base_channel"])
    return base, lt["lags"], names


def convolve(envelopes, w_base: np.ndarray, lags) -> np.ndarray:
    """Concatenated X (samples x channels) at 32 Hz, convolving trial by trial."""
    return np.concatenate([lag_matrix(env, lags) @ w_base for env in envelopes])


def upsample(x: np.ndarray, factor: int = UPSAMPLE) -> np.ndarray:
    return scipy.signal.resample_poly(x, factor, 1, axis=0)


def insert_cz(x63: np.ndarray, cz_index: int = CZ_INDEX) -> np.ndarray:
    """63-channel Cz-referenced data -> 64 channels with Cz = 0."""
    return np.insert(x63, cz_index, 0.0, axis=1)


def analysis_space(x64: np.ndarray, cz_index: int = CZ_INDEX) -> np.ndarray:
    """Cz reference, 1-9 Hz band-pass, drop Cz (at 128 Hz) - where SNR is measured."""
    referenced = x64 - x64[:, [cz_index]]
    return np.delete(bandpass(referenced, FS_IN), cz_index, axis=1)


def mean_rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean(axis=0)).mean())


def snr_db(signal64: np.ndarray, noise64: np.ndarray) -> float:
    return 20 * np.log10(mean_rms(analysis_space(signal64)) / mean_rms(analysis_space(noise64)))


def scale_to_snr(signal64: np.ndarray, noise64: np.ndarray, target_db: float) -> tuple[np.ndarray, float]:
    """Signal rescaled so snr_db(signal, noise) == target_db, and the gain used."""
    gain = 10 ** ((target_db - snr_db(signal64, noise64)) / 20)
    return signal64 * gain, gain


@dataclass(frozen=True)
class NoiseSource:
    subject: str
    noise: np.ndarray            # samples x 64 at 128 Hz, time-reversed concatenated HRTF EEG
    envelopes: list[np.ndarray]  # attended envelope per HRTF trial, 32 Hz, truncated to EEG
    groups32: np.ndarray         # story-part group per 32 Hz sample (forward time)
    trial32: np.ndarray          # source trial id per 32 Hz sample (forward time)
    ch_names: tuple[str, ...]    # 64 BioSemi names
    noise_sd: np.ndarray         # 63, per-channel sd of the noise in the analysis space
    fraction_cut: float = 0.0    # fraction of the base cut out as native transients


def transient_mask(eeg: np.ndarray, fs: float, ptp_uv: float | None, window_s: float = 1.0,
                   step_s: float = 0.5, margin_s: float = 0.25) -> np.ndarray:
    """Samples inside 1 s windows where any channel exceeds `ptp_uv` peak-to-peak (+ margin)."""
    mask = np.zeros(len(eeg), bool)
    if ptp_uv is None:
        return mask
    win, step, margin = int(window_s * fs), int(step_s * fs), int(margin_s * fs)
    for s0 in range(0, len(eeg) - win + 1, step):
        if np.ptp(eeg[s0:s0 + win], axis=0).max() > ptp_uv * 1e-6:
            mask[max(0, s0 - margin):s0 + win + margin] = True
    return mask


def splice(eeg: np.ndarray, keep: np.ndarray, fs: float, crossfade_s: float = 0.25,
           min_segment_s: float = 2.0) -> np.ndarray:
    """Concatenate the kept segments (>= min_segment_s) with raised-cosine crossfades.

    Each join overlaps the end of one segment with the start of the next over
    `crossfade_s`, so cutting out transients does not create new steps.
    """
    padded = np.concatenate([[False], keep, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    segments = [eeg[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= int(min_segment_s * fs)]
    if not segments:
        raise ValueError("no clean segment left")
    n = int(crossfade_s * fs)
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n))[:, None]
    out = segments[0]
    for seg in segments[1:]:
        if n == 0:
            out = np.concatenate([out, seg])
            continue
        blended = out[-n:] * ramp[::-1] + seg[:n] * ramp
        out = np.concatenate([out[:-n], blended, seg[n:]])
    return out


def noise_source(subject: str, ptp_uv: float | None | str = "config") -> NoiseSource:
    """Noise and envelopes for one noise subject.

    Native transients of the MWF-cleaned base (1 s windows with any channel
    above `native_transient_ptp_uv` peak-to-peak, plus 0.25 s margins) are
    cut out of the concatenated EEG before time reversal (open item 13,
    option b) and the remaining segments joined with crossfades; X is then
    trimmed to the cleaned noise length, so a recording is a little shorter
    than 38.5 min. `ptp_uv=None` keeps the noise uncut.
    """
    paths = load_paths()
    section = load_yaml("section_a")
    condition = section["condition"]
    if ptp_uv == "config":
        ptp_uv = section.get("native_transient_ptp_uv")
    trials = load_subject(subject)
    groups = presentation_groups(trials)
    kept = [(t, g) for t, g in zip(trials, groups) if t.condition == condition]
    eeg, envelopes, groups32, trial32 = [], [], [], []
    for trial, group in kept:
        # Trim each trial to a multiple of 4 samples so 128 Hz and 32 Hz align.
        n128 = len(trial.eeg) // UPSAMPLE * UPSAMPLE
        n32 = n128 // UPSAMPLE
        env = wav_envelope(paths["dataset1"] / "stimuli" / attended_stimulus(trial), paths["cache"] / "envelopes")
        eeg.append(trial.eeg[:n128])
        envelopes.append(env[:n32])
        groups32.append(np.full(n32, group))
        trial32.append(np.full(n32, trial.trial_id))
    forward = np.concatenate(eeg)
    cut = transient_mask(forward, FS_IN, ptp_uv)
    if cut.any():
        forward = splice(forward, ~cut, FS_IN)
    n32 = len(forward) // UPSAMPLE
    forward = forward[: n32 * UPSAMPLE]
    noise = forward[::-1].copy()
    # Trim the envelope sequence (and its labels) to the cleaned noise length.
    trimmed, total = [], 0
    for env in envelopes:
        take = min(len(env), n32 - total)
        if take <= 0:
            break
        trimmed.append(env[:take])
        total += take
    return NoiseSource(
        subject=subject,
        noise=noise,
        envelopes=trimmed,
        groups32=np.concatenate(groups32)[:n32],
        trial32=np.concatenate(trial32)[:n32],
        ch_names=kept[0][0].ch_names,
        noise_sd=analysis_space(noise).std(axis=0),
        fraction_cut=float(cut.mean()),
    )


@dataclass(frozen=True)
class SynthRecording:
    subject: str              # noise subject
    condition: str            # 'L' or 'R' (attended side of the base TRF)
    snr_db: float
    eeg: np.ndarray           # samples x 64 at 128 Hz: clean + noise
    clean: np.ndarray         # samples x 64 at 128 Hz: scaled X, Cz = 0
    envelope: np.ndarray      # 32 Hz attended envelope (raw, not z-scored)
    groups32: np.ndarray
    trial32: np.ndarray
    ch_names: tuple[str, ...]
    gain: float


def synthesize(source: NoiseSource, condition: str, target_snr_db: float,
               base: dict[str, np.ndarray], lags) -> SynthRecording:
    x32 = convolve(source.envelopes, base[condition], lags) * source.noise_sd
    clean = insert_cz(upsample(x32))
    clean, gain = scale_to_snr(clean, source.noise, target_snr_db)
    return SynthRecording(
        subject=source.subject,
        condition=condition,
        snr_db=target_snr_db,
        eeg=clean + source.noise,
        clean=clean,
        envelope=np.concatenate(source.envelopes),
        groups32=source.groups32,
        trial32=source.trial32,
        ch_names=source.ch_names,
        gain=gain,
    )


def cut_trials(n32: int, trial_seconds: float = 120.0, fs: float = FS_OUT) -> list[slice]:
    """Consecutive 120 s trials (32 Hz samples); the remainder is dropped."""
    length = int(trial_seconds * fs)
    return [slice(i * length, (i + 1) * length) for i in range(n32 // length)]


def load_noise_split() -> dict[str, list[str]]:
    with open(ADR_ROOT / "splits" / "dataset1_noise_subjects.yaml") as f:
        return yaml.safe_load(f)["sets"]
