"""Section A evaluation: one noise subject x scenario -> pipelines -> outcomes (PROJECT_PLAN.md §9.4, §11).

A scenario is (noise subject, SNR, SAR, artifact type); each has an
attend-left and an attend-right recording (Dataset III). Per recording:
synthesise (native transients cut), inject one artifact type from the
set's template bank, run each pipeline, then the Section A outcomes:

- short-term TRFs per 120 s trial (Das rule; lambda 0.2 as a check) and the
  median relative MSE against the base TRF (Eq. 14-15) - the tuning primary;
- 1-30 Hz relative RMS error against the clean synthetic EEG over the
  recording (Cz reference; native transients excluded) - tuning secondary;
  plus 1-9 Hz and spectral errors;
- detection against the injection log (window level) and fraction modified.

Detector scores depend only on the recording, so `recording_scores` is
computed once and reused across thresholds and reconstruction settings.
Forward prediction (2-fold within subject, story part left out) needs both
conditions and lives in `forward_prediction`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.signal

from adr.config import load_yaml
from adr.detection import features, prep
from adr.detection import models as M
from adr.detection.evaluate import window_metrics
from adr.detection.train import load_model
from adr.preprocessing.channels import detector_channels
from adr.preprocessing.das import CZ_INDEX, FS_IN, preprocess_eeg, zscore
from adr.reconstruction import router
from adr.reconstruction.check import native_transient_mask
from adr.synth import dataset3 as d3
from adr.synth import inject as I
from adr.trf import ridge
from adr.trf.metrics import relative_mse

FS = FS_IN
TARGETS = ("ocular", "muscle", "motion", "pop", "artifact")


@dataclass(frozen=True)
class Scenario:
    subject: str
    snr_db: float
    sar_db: float
    kind: str

    def seed(self, condition: str) -> int:
        key = f"{self.subject}|{self.snr_db}|{self.sar_db}|{self.kind}|{condition}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


@dataclass
class Recording:
    scenario: Scenario
    condition: str
    rec: d3.SynthRecording          # clean synthetic recording
    injected: I.Injected
    native: np.ndarray              # samples excluded from scoring
    truth: dict                     # per-window ground truth from the injection log
    starts: np.ndarray              # window starts (s)
    _cache: dict | None = None      # clean recording's filtered versions (same for every pipeline)

    def clean_band(self, name: str) -> np.ndarray:
        if self._cache is None:
            self._cache = {}
        if name not in self._cache:
            self._cache[name] = _cz_band(self.rec.eeg, (1, 30)) if name == "1_30" else d3.analysis_space(self.rec.eeg)
        return self._cache[name]


def build(source: d3.NoiseSource, scenario: Scenario, condition: str, bank, base, lags) -> Recording:
    rec = d3.synthesize(source, condition, scenario.snr_db, base, lags)
    inj = I.inject(rec.eeg, FS, scenario.kind, scenario.sar_db, np.random.default_rng(scenario.seed(condition)),
                   bank=bank, ch_names=list(rec.ch_names))
    det = load_yaml("detection")
    truth = I.window_truth(inj.log, len(rec.eeg), FS, det["window_s"], det["step_s"])
    native = native_transient_mask(rec.eeg, FS, load_yaml("section_a").get("native_transient_ptp_uv"))
    return Recording(scenario, condition, rec, inj, native, {c: truth[c].to_numpy() for c in (*I.TYPES, "artifact")},
                     truth["start"].to_numpy())


def recording_scores(eeg: np.ndarray, ch_names, variant: str, models=None) -> dict:
    """LightGBM scores per window and target for one recording (128 Hz detector)."""
    cfg = features.detection_config()
    X = features.compute(prep.finish(eeg, FS, list(ch_names), cfg["highpass_hz"]), FS, detector_channels(), cfg)
    X = features.apply_variant(X, variant)
    models = models or {t: load_model(FS, "lightgbm", t, variant) for t in TARGETS}
    return {t: M.score("lightgbm", models[t], X) for t in TARGETS}


def flags_from_scores(scores: dict, thresholds: dict) -> dict:
    return {c: scores[c] >= thresholds[c] for c in router.CLASSES}


def run_proposed(r: Recording, scores: dict, thresholds: dict, recon_cfg: dict) -> router.Result:
    n = min(len(r.starts), len(scores["ocular"]))
    flags = {c: f[:n] for c, f in flags_from_scores(scores, thresholds).items()}
    cls_scores = {c: scores[c][:n] for c in router.CLASSES}
    return router.route(r.injected.eeg, FS, list(r.rec.ch_names), flags, r.starts[:n], 1.0, scores=cls_scores,
                        cfg=recon_cfg)


def run_oracle_mwf(r: Recording, recon_cfg: dict) -> router.Result:
    mask = router.mask_from_windows(r.truth["artifact"], r.starts, 1.0, len(r.rec.eeg), FS)
    return router.single_mwf(r.injected.eeg, mask, cfg=recon_cfg)


def _cz_band(x: np.ndarray, band) -> np.ndarray:
    sos = scipy.signal.butter(4, band, "bandpass", fs=FS, output="sos")
    ref = x - x[:, [CZ_INDEX]]
    return np.delete(scipy.signal.sosfiltfilt(sos, ref, axis=0), CZ_INDEX, axis=1)


def _rrmse(est, truth, mask):
    return float(np.sqrt(((est[mask] - truth[mask]) ** 2).mean()) / np.sqrt((truth[mask] ** 2).mean()))


def _spectral_rrmse(est, truth, mask):
    f, p_est = scipy.signal.welch(est[mask], fs=FS, nperseg=256, axis=0)
    _, p_true = scipy.signal.welch(truth[mask], fs=FS, nperseg=256, axis=0)
    keep = (f >= 1) & (f <= 30)
    return float(np.sqrt(((np.log10(p_est[keep]) - np.log10(p_true[keep])) ** 2).mean()))


def trf_outcomes(eeg_out: np.ndarray, r: Recording, base: np.ndarray, cfg: dict | None = None,
                 check: bool = True) -> dict:
    """Short-term TRF relMSE (median over 120 s trials) with the Das rule (and lambda 0.2 if `check`)."""
    section = cfg or load_yaml("section_a")
    trf = section["trf"]
    eeg32 = zscore(preprocess_eeg(eeg_out))
    env = r.rec.envelope[: len(eeg32)]
    env = (env - env.mean()) / env.std()
    das, loose = [], []
    for cut in d3.cut_trials(len(eeg32), section["trial_seconds"]):
        das.append(relative_mse(ridge.fit([env[cut]], [eeg32[cut]], 32.0, trf["tmin"], trf["tmax"], trf["lambda_short"]).weights, base))
        if check:
            loose.append(relative_mse(ridge.fit([env[cut]], [eeg32[cut]], 32.0, trf["tmin"], trf["tmax"], trf["lambda_short_check"]).weights, base))
    out = {"trf_relmse": float(np.median(das)), "n_trials": len(das)}
    if check:
        out["trf_relmse_lambda02"] = float(np.median(loose))
    return out


def signal_outcomes(eeg_out: np.ndarray, r: Recording, light: bool = False) -> dict:
    keep = ~r.native
    out_30, clean_30 = _cz_band(eeg_out, (1, 30)), r.clean_band("1_30")
    out = {"rrmse_1_30": _rrmse(out_30, clean_30, keep)}
    if not light:
        out["rrmse_1_9"] = _rrmse(d3.analysis_space(eeg_out), r.clean_band("1_9"), keep)
        out["spectral_rrmse_1_30"] = _spectral_rrmse(out_30, clean_30, keep)
    return out


def detection_outcomes(r: Recording, flags: dict) -> dict:
    n = min(len(r.starts), len(next(iter(flags.values()))))
    truth = r.truth[r.scenario.kind][:n]
    any_flag = np.any([flags[c][:n] for c in router.CLASSES], axis=0)
    m = window_metrics(truth, any_flag.astype(float), 0.5)
    cls = flags[r.scenario.kind][:n]
    return {"det_sensitivity": m["sensitivity"], "det_specificity": m["specificity"], "det_f1": m["f1"],
            "det_class_sensitivity": float(cls[truth].mean()) if truth.any() else np.nan}


def evaluate(eeg_out: np.ndarray, r: Recording, base: np.ndarray, fraction_modified: float, light: bool = False) -> dict:
    """All outcomes; `light=True` (tuning) keeps only the Das-rule TRF and the 1-30 Hz error."""
    return {**trf_outcomes(eeg_out, r, base, check=not light), **signal_outcomes(eeg_out, r, light),
            "fraction_modified": fraction_modified}


def forward_prediction(outputs: dict, recs: dict, base: dict, cfg: dict | None = None) -> dict:
    """2-fold forward prediction within one scenario (open item 9).

    `outputs` / `recs`: condition -> processed EEG / Recording. Each 120 s
    trial belongs to one story part; trials straddling two parts share
    stimuli with both groups and are left out (stimulus rule). TRFs are
    fitted per condition on one part (Das rule) and predict the other part;
    returns mean correlation with the processed EEG and with the noise-free
    synthetic signal X.
    """
    section = cfg or load_yaml("section_a")
    trf = section["trf"]
    trials = {}
    for cond, r in recs.items():
        eeg32 = zscore(preprocess_eeg(outputs[cond]))
        x32 = preprocess_eeg(r.rec.clean)
        env = r.rec.envelope[: len(eeg32)]
        env = (env - env.mean()) / env.std()
        groups = r.rec.groups32[: len(eeg32)]
        for cut in d3.cut_trials(len(eeg32), section["trial_seconds"]):
            g = np.unique(groups[cut])
            if len(g) == 1:
                trials.setdefault(cond, []).append((int(g[0]), env[cut], eeg32[cut], x32[cut]))
    r_eeg, r_x, n_excluded = [], [], 0
    parts = sorted({t[0] for c in trials for t in trials[c]})
    if len(parts) < 2:
        return {"fwd_r_eeg": np.nan, "fwd_r_clean": np.nan, "fwd_folds": 0}
    for test_part in parts:
        for cond, items in trials.items():
            train = [t for t in items if t[0] != test_part]
            test = [t for t in items if t[0] == test_part]
            if not train or not test:
                continue
            model = ridge.fit([t[1] for t in train], [t[2] for t in train], 32.0, trf["tmin"], trf["tmax"], trf["lambda_short"])
            for _, env, eeg, x in test:
                pred = model.predict(env)
                r_eeg.append(np.mean([np.corrcoef(pred[:, c], eeg[:, c])[0, 1] for c in range(eeg.shape[1])]))
                r_x.append(np.mean([np.corrcoef(pred[:, c], x[:, c])[0, 1] for c in range(x.shape[1])]))
    return {"fwd_r_eeg": float(np.mean(r_eeg)), "fwd_r_clean": float(np.mean(r_x)), "fwd_folds": len(parts),
            "fwd_test_trials": len(r_eeg)}
