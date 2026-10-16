"""Reconstruction sanity check with oracle flags (milestone 7).

On a Set A noise subject's Dataset III recording, inject one artifact type
at a time and reconstruct with the routed methods and with a single MWF,
both given the *true* (injection-log) window flags - the oracle upper bound
the plan uses in the detection-quality ablation. Reports the relative RMS
error against the clean synthetic EEG, broadband and in the analysis space
(Cz reference, 1-9 Hz), over the injected samples and over the whole
recording. Samples inside native transients of the "clean" base (any
channel > `native_transient_ptp_uv` peak-to-peak in a 1 s window,
configs/section_a.yaml) are excluded from the "_excl_native" metrics; the
plain metrics score everything. Set A only; nothing is tuned or frozen here.

Used by scripts/11_reconstruction_check.py and notebooks/06_reconstruction.ipynb.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from adr.config import load_yaml
from adr.reconstruction import router
from adr.synth import dataset3 as d3
from adr.synth import inject as I
from adr.synth import templates as T

log = logging.getLogger("adr.reconstruction_check")


def native_transient_mask(eeg: np.ndarray, fs: float, ptp_uv: float | None, window_s: float = 1.0,
                          step_s: float = 0.5) -> np.ndarray:
    """Samples inside 1 s windows of the clean base where any channel exceeds `ptp_uv` peak-to-peak."""
    if ptp_uv is None:
        return np.zeros(len(eeg), bool)
    win, step = int(window_s * fs), int(step_s * fs)
    starts = np.arange(0, len(eeg) - win + 1, step)
    bad = np.array([np.ptp(eeg[s:s + win], axis=0).max() > ptp_uv * 1e-6 for s in starts])
    return router.mask_from_windows(bad, starts / fs, window_s, len(eeg), fs)


def rrmse(estimate: np.ndarray, truth: np.ndarray, mask=None) -> float:
    if mask is not None:
        estimate, truth = estimate[mask], truth[mask]
    return float(np.sqrt(((estimate - truth) ** 2).mean()) / np.sqrt((truth ** 2).mean()))


def run(subject: str | None = None, snr_db: float = -10, sar_db: float = 0, bank_name: str = "phantom_set_a_templates",
        seed: int = 0) -> pd.DataFrame:
    det = load_yaml("detection")
    fs, window_s, step_s = 128.0, det["window_s"], det["step_s"]
    base, lags, _ = d3.load_base_trfs()
    subject = subject or d3.load_noise_split()["set_a"][0]
    rec = d3.synthesize(d3.noise_source(subject), "L", snr_db, base, lags)
    bank = T.load_bank(bank_name)
    names = list(rec.ch_names)
    clean_an = d3.analysis_space(rec.eeg)
    native = native_transient_mask(rec.eeg, fs, load_yaml("section_a").get("native_transient_ptp_uv"))
    log.info("%s: native transients cover %.1f %% of samples (excluded from *_excl_native)", subject, 100 * native.mean())
    rng = np.random.default_rng(seed)
    rows = []
    for kind in I.TYPES:
        inj = I.inject(rec.eeg, fs, kind, sar_db, rng, bank=bank, ch_names=names)
        truth = I.window_truth(inj.log, len(rec.eeg), fs, window_s, step_s)
        starts = truth["start"].to_numpy()
        flags = {c: (truth[c].to_numpy() if c == kind else np.zeros(len(truth), bool)) for c in router.CLASSES}
        art_mask = router.mask_from_windows(truth[kind].to_numpy(), starts, window_s, len(rec.eeg), fs)
        outputs = {
            "no correction": inj.eeg,
            "routed (oracle flags)": router.route(inj.eeg, fs, names, flags, starts, window_s).eeg,
            "single MWF (oracle flags)": router.single_mwf(inj.eeg, art_mask).eeg,
        }
        for name, out in outputs.items():
            out_an = d3.analysis_space(out)
            rows.append({"subject": subject, "sar_db": sar_db, "type": kind, "pipeline": name, "events": len(inj.log),
                         "fraction_flagged": float(art_mask.mean()),
                         "rrmse_broadband_flagged": rrmse(out, rec.eeg, art_mask),
                         "rrmse_analysis_flagged": rrmse(out_an, clean_an, art_mask),
                         "rrmse_analysis_all": rrmse(out_an, clean_an),
                         "rrmse_broadband_flagged_excl_native": rrmse(out, rec.eeg, art_mask & ~native),
                         "rrmse_analysis_flagged_excl_native": rrmse(out_an, clean_an, art_mask & ~native)})
            r = rows[-1]
            log.info("%s SAR %+d %-6s %-26s RRMSE flagged 1-9 Hz %.3f (excl. native %.3f), broadband %.3f (excl. native %.3f)",
                     subject, sar_db, kind, name, r["rrmse_analysis_flagged"], r["rrmse_analysis_flagged_excl_native"],
                     r["rrmse_broadband_flagged"], r["rrmse_broadband_flagged_excl_native"])
    return pd.DataFrame(rows)
