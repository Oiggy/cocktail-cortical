"""Artifact injection with an injection log (PROJECT_PLAN.md §9.3 step 6; milestone 6).

One artifact type per recording (ocular, muscle, motion or pop) so results
can be reported per type (§9.4 grid: SNR x SAR x artifact type x
pipeline). Event onsets follow a Poisson process (`events_per_minute`) with
at least `min_gap_s` between events; each event is a random template from
the bank (or a simulated pop) scaled to the target signal-to-artifact ratio

    SAR = 20 log10( rms(EEG under the event) / rms(artifact) )

with RMS averaged over channels ("all", default) or taken on the channels
the artifact actually touches ("affected", ablation §12). Pops always use
the affected channel and are never placed on Cz. The log (one row per
event: type, onset/offset, channels, SAR, gain, template source) is the
ground truth for detection scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adr.config import load_yaml
from adr.synth import pops as pop_sim
from adr.synth.templates import Template

TYPES = ("ocular", "muscle", "motion", "pop")


@dataclass
class Injected:
    eeg: np.ndarray           # samples x 64 with artifacts added
    artifact: np.ndarray      # samples x 64, the added artifacts only
    log: pd.DataFrame


def _rms(x: np.ndarray, channels=None) -> float:
    if channels is not None:
        x = x[:, channels]
    return float(np.sqrt((x ** 2).mean(axis=0)).mean())


def onsets(rng: np.random.Generator, n_samples: int, fs: float, durations_s, events_per_minute: float,
           min_gap_s: float) -> list[int]:
    """Poisson onsets that leave `min_gap_s` between consecutive events."""
    out, t = [], 0.0
    total = n_samples / fs
    k = 0
    while k < len(durations_s):
        t += rng.exponential(60.0 / events_per_minute)
        if t + durations_s[k] >= total:
            break
        out.append(int(t * fs))
        t += durations_s[k] + min_gap_s
        k += 1
    return out


def inject(eeg: np.ndarray, fs: float, kind: str, sar_db: float, rng: np.random.Generator,
           bank: list[Template] | None = None, ch_names=None, cfg: dict | None = None,
           sar_reference: str | None = None) -> Injected:
    """Inject one artifact type into a 64-channel recording (volts, `fs` Hz)."""
    cfg = cfg or load_yaml("injection")
    sar_reference = sar_reference or cfg["sar_reference"]
    artifact = np.zeros_like(eeg)
    rows = []
    if kind == "pop":
        if ch_names is None:
            raise ValueError("pops need ch_names (to avoid Cz)")
        pop_cfg = load_yaml("detection")["pops"]
        durations = [5 * pop_cfg["tau_s"][1]] * 10_000
        for onset in onsets(rng, len(eeg), fs, durations, cfg["events_per_minute"], cfg["min_gap_s"]):
            (p,) = pop_sim.draw(rng, 1, fs, ch_names, 1, pop_cfg["amplitude_uv"], pop_cfg["tau_s"])
            length = min(int(5 * p.tau * fs), len(eeg) - onset)
            wave = pop_sim.waveform(length, fs, 0, 1.0, p.tau) * np.sign(p.amplitude)
            under = eeg[onset:onset + length, [p.channel]]
            gain = _rms(under) / (np.sqrt((wave ** 2).mean()) * 10 ** (sar_db / 20))
            artifact[onset:onset + length, p.channel] += gain * wave
            rows.append({"type": "pop", "onset_s": onset / fs, "offset_s": (onset + length) / fs,
                         "channels": ch_names[p.channel], "sar_db": sar_db, "gain": gain,
                         "source": f"simulated:tau={p.tau:.3f}"})
    else:
        candidates = [t for t in bank or [] if t.kind == kind]
        if not candidates:
            raise ValueError(f"no {kind} templates in the bank")
        picks = [candidates[i] for i in rng.integers(0, len(candidates), 10_000)]
        starts = onsets(rng, len(eeg), fs, [t.duration for t in picks], cfg["events_per_minute"], cfg["min_gap_s"])
        for template, onset in zip(picks, starts):
            length = len(template.data)
            under = eeg[onset:onset + length]
            if sar_reference == "affected":
                affected = np.flatnonzero(np.abs(template.data).max(0) >= 0.5 * np.abs(template.data).max())
                gain = _rms(under, affected) / (_rms(template.data, affected) * 10 ** (sar_db / 20))
            else:
                gain = _rms(under) / (_rms(template.data) * 10 ** (sar_db / 20))
            artifact[onset:onset + length] += gain * template.data
            rows.append({"type": kind, "onset_s": onset / fs, "offset_s": (onset + length) / fs,
                         "channels": "all", "sar_db": sar_db, "gain": gain, "source": template.source})
    return Injected(eeg=eeg + artifact, artifact=artifact, log=pd.DataFrame(rows))


def window_truth(log: pd.DataFrame, n_samples: int, fs: float, window_s: float, step_s: float) -> pd.DataFrame:
    """Per-window ground truth from an injection log (window overlaps an event)."""
    starts = np.arange(0, n_samples - int(window_s * fs) + 1, int(step_s * fs)) / fs
    out = pd.DataFrame({"start": starts})
    for kind in TYPES:
        events = log[log["type"] == kind] if len(log) else log
        mask = np.zeros(len(starts), dtype=bool)
        for row in events.itertuples(index=False):
            mask |= (starts < row.offset_s) & (starts + window_s > row.onset_s)
        out[kind] = mask
    out["artifact"] = out[list(TYPES)].any(axis=1)
    return out
