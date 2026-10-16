"""ICA + ICLabel comparison method (PROJECT_PLAN.md §11.1).

Follows §11.1 and mne-icalabel 0.10.0 (verified 2026-10-15): ICLabel expects
a common-average-referenced copy band-passed 1-100 Hz and an extended
infomax decomposition, either ``ICA(method="infomax",
fit_params=dict(extended=True))`` or ``ICA(method="picard",
fit_params=dict(ortho=False, extended=True))``; it warns (does not fail)
otherwise. Here: picard (extended, non-orthogonal), fitted on an
average-referenced copy filtered 1 Hz to min(100 Hz, just below Nyquist) -
so 1-63 Hz in Section A (128 Hz), a stated limitation of the baseline -
with as many components as the data rank (63 after average referencing).
A component is removed if its top ICLabel class is eye blink, muscle
artifact, heart beat, line noise or channel noise with probability >= the
threshold (0.7; 0.5 and 0.9 as a sensitivity check). The unmixing is then
applied to the pipeline's own (average-referenced, unfiltered) data; the
downstream Cz re-reference makes the reference choice irrelevant.

ICLabel is itself a neural network; it is allowed here because this is the
baseline, never part of the proposed method.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import mne
import numpy as np
from mne_icalabel import label_components

from adr.config import load_project, load_yaml

log = logging.getLogger("adr.ica_iclabel")


@dataclass
class ICAResult:
    eeg: np.ndarray            # cleaned, samples x channels, average-referenced
    labels: list[str]          # top ICLabel class per component
    probabilities: np.ndarray  # probability of that class
    excluded: list[int]
    n_components: int
    fit_band: tuple[float, float]

    def removed_by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.excluded:
            out[self.labels[i]] = out.get(self.labels[i], 0) + 1
        return out


def components_to_remove(labels, probabilities, reject_labels, threshold: float) -> list[int]:
    """§11.1 rule: top class in `reject_labels` with probability >= threshold."""
    return [i for i, (lab, p) in enumerate(zip(labels, probabilities)) if lab in reject_labels and p >= threshold]


def _raw(eeg: np.ndarray, fs: float, ch_names) -> mne.io.RawArray:
    info = mne.create_info(list(ch_names), fs, "eeg")
    raw = mne.io.RawArray(eeg.T.copy(), info, verbose="error")
    raw.set_montage("biosemi64", verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw


def fit_and_label(eeg: np.ndarray, fs: float, ch_names, cfg: dict | None = None, seed: int | None = None):
    """Fit ICA on the ICLabel copy and label the components. Returns (ica, labels, probs, band)."""
    cfg = cfg or load_yaml("baselines")["ica_iclabel"]
    seed = load_project()["seed"] if seed is None else seed
    fit_raw = _raw(eeg, fs, ch_names)
    band = (cfg["fit_highpass_hz"], min(cfg["fit_lowpass_hz"], fs / 2 - 1))
    fit_raw.filter(*band, verbose="error")
    # Estimated from the data: rank="info" ignores the applied average reference
    # and returns 64, but the referenced data have rank 63.
    rank = mne.compute_rank(fit_raw, rank=None, verbose="error")["eeg"]
    fit_params = dict(ortho=False, extended=True) if cfg["method"] == "picard" else dict(extended=True)
    ica = mne.preprocessing.ICA(n_components=rank, method=cfg["method"], fit_params=fit_params,
                                max_iter=cfg["max_iter"], random_state=seed, verbose="error")
    ica.fit(fit_raw, decim=cfg["fit_decim"], verbose="error")
    with warnings.catch_warnings():
        # expected below 250 Hz: the copy cannot be band-passed up to 100 Hz
        warnings.filterwarnings("ignore", message=".*not filtered between 1 and 100 Hz.*")
        out = label_components(fit_raw, ica, method="iclabel")
    return ica, list(out["labels"]), np.asarray(out["y_pred_proba"]), band


def apply(eeg: np.ndarray, fs: float, ch_names, ica, labels, probabilities, threshold: float,
          reject_labels, band) -> ICAResult:
    excluded = components_to_remove(labels, probabilities, reject_labels, threshold)
    raw = _raw(eeg, fs, ch_names)
    cleaned = ica.apply(raw.copy(), exclude=excluded, verbose="error").get_data().T
    return ICAResult(cleaned, labels, probabilities, excluded, ica.n_components_, band)


def run(eeg: np.ndarray, fs: float, ch_names, thresholds=None, cfg: dict | None = None) -> dict[float, ICAResult]:
    """ICA + ICLabel cleaning at the main threshold and the sensitivity thresholds (one fit)."""
    cfg = cfg or load_yaml("baselines")["ica_iclabel"]
    thresholds = thresholds or [cfg["threshold"], *cfg["sensitivity_thresholds"]]
    ica, labels, probs, band = fit_and_label(eeg, fs, ch_names, cfg)
    results = {t: apply(eeg, fs, ch_names, ica, labels, probs, t, cfg["reject_labels"], band) for t in thresholds}
    main = results[thresholds[0]]
    log.info("ICA+ICLabel: %d components (fit %.0f-%.0f Hz); removed at p>=%.1f: %s", main.n_components, *band,
             thresholds[0], main.removed_by_class())
    return results
