"""Simulated electrode pops (PROJECT_PLAN.md §3.4).

Standard model: an abrupt step on one channel followed by an exponential
decay back to baseline, with randomised amplitude, decay constant and
polarity. Pops are never placed on Cz (Cz re-referencing would spread them
to every channel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pop:
    onset: int        # sample
    channel: int      # column index
    amplitude: float  # signed, in data units (volts)
    tau: float        # s


def waveform(n: int, fs: float, onset: int, amplitude: float, tau: float) -> np.ndarray:
    """Step of `amplitude` at `onset` decaying with time constant `tau`."""
    out = np.zeros(n)
    t = np.arange(n - onset) / fs
    out[onset:] = amplitude * np.exp(-t / tau)
    return out


def draw(rng: np.random.Generator, n: int, fs: float, ch_names, n_pops: int,
         amplitude_uv=(50, 500), tau_s=(0.05, 0.5), exclude=("Cz",)) -> list[Pop]:
    """Random pops: uniform onset, channel (not in `exclude`), |amplitude|, tau; random sign."""
    allowed = [i for i, ch in enumerate(ch_names) if ch not in exclude]
    pops = []
    for _ in range(n_pops):
        sign = rng.choice([-1.0, 1.0])
        pops.append(Pop(
            onset=int(rng.integers(0, n)),
            channel=int(rng.choice(allowed)),
            amplitude=float(sign * rng.uniform(*amplitude_uv) * 1e-6),
            tau=float(rng.uniform(*tau_s)),
        ))
    return pops


def inject(data: np.ndarray, fs: float, pops: list[Pop]) -> np.ndarray:
    """Copy of `data` (samples x channels) with the pops added."""
    out = data.copy()
    for pop in pops:
        out[:, pop.channel] += waveform(len(data), fs, pop.onset, pop.amplitude, pop.tau)
    return out
