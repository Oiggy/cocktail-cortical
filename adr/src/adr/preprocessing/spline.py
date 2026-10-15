"""Spherical-spline mapping between electrode layouts (PROJECT_PLAN.md §6, §9.3).

Used to map the phantom's BioSemi 128 layout (and, for the template
ablation, PhysioMotion's 10-10 layout) onto the BioSemi 64 positions.
Both the phantom and MNE's biosemi64 positions are ideal spheres centred
at the origin (radius 56 mm and 95 mm) with the same lowest electrode
height, so the mapping is done on the unit sphere (Perrin et al. 1989
splines via MNE's interpolation matrix; MNE normalises positions to the
sphere internally).
"""
from __future__ import annotations

import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix  # private but long-stable


def positions(info: mne.Info, picks=None) -> np.ndarray:
    picks = range(len(info["ch_names"])) if picks is None else picks
    return np.array([info["chs"][i]["loc"][:3] for i in picks])


def biosemi64_positions() -> tuple[list[str], np.ndarray]:
    montage = mne.channels.make_standard_montage("biosemi64")
    pos = montage.get_positions()["ch_pos"]
    return list(pos), np.array(list(pos.values()))


def matrix(pos_from: np.ndarray, pos_to: np.ndarray, alpha: float = 1e-5) -> np.ndarray:
    """n_to x n_from interpolation matrix (rows sum to ~1)."""
    unit = lambda p: p / np.linalg.norm(p, axis=1, keepdims=True)
    return _make_interpolation_matrix(unit(pos_from), unit(pos_to), alpha=alpha)


def to_biosemi64(data: np.ndarray, pos_from: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """samples x n_from -> samples x 64 at the BioSemi 64 positions."""
    names, pos_to = biosemi64_positions()
    return data @ matrix(pos_from, pos_to).T, names
