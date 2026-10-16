"""Section A evaluation on the held-out set with frozen settings (PROJECT_PLAN.md §9.4, §11, §14).

For each noise subject of the test set: the full grid (configs/section_a.yaml
`snr_db` x injection `sar_db` x 4 types x 2 conditions), every §11 pipeline -
clean reference, no correction, the proposed method (settings frozen on the
other set), MWF with oracle flags, ICA + ICLabel (p >= 0.7, plus 0.5 / 0.9)
and the automatic MWF - and every Section A outcome, including the 2-fold
forward prediction per scenario. Uses the test set's own template pool.
Run only after configs/frozen/section_a_tuned_on_<other set>.yaml exists.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

from adr.baselines import ica_iclabel, mwf_auto
from adr.config import load_paths, load_yaml
from adr.evaluation import section_a as E
from adr.evaluation import tuning
from adr.synth import dataset3 as d3
from adr.synth import templates as T

log = logging.getLogger("adr.test_set")
OTHER = {"set_a": "set_b", "set_b": "set_a"}


def evaluate_subject(subject: str, test_set: str, kinds=None, snrs=None, sars=None) -> pd.DataFrame:
    section, inj = load_yaml("section_a"), load_yaml("injection")
    frozen = tuning.load_frozen(OTHER[test_set])
    recon = frozen.recon_cfg()
    base, lags, _ = d3.load_base_trfs()
    bank = T.load_bank(f"phantom_{test_set}_templates")
    source = d3.noise_source(subject)
    rows = []
    for snr, sar, kind in product(snrs or section["snr_db"], sars or inj["sar_db"], kinds or ("ocular", "muscle", "motion", "pop")):
        scenario = E.Scenario(subject, snr, sar, kind)
        outputs, recs = {}, {}
        for cond in d3.CONDITIONS:
            r = E.build(source, scenario, cond, bank, base, lags)
            recs[cond] = r
            names = list(r.rec.ch_names)
            scores = E.recording_scores(r.injected.eeg, names, frozen.variant)
            flags = E.flags_from_scores(scores, frozen.thresholds)
            proposed = E.run_proposed(r, scores, frozen.thresholds, recon)
            oracle = E.run_oracle_mwf(r, recon)
            ica = ica_iclabel.run(r.injected.eeg, E.FS, names)
            auto, auto_info = mwf_auto.run(r.injected.eeg, E.FS, names)
            pipelines = {
                "clean": (r.rec.eeg, 0.0, {}),
                "raw": (r.injected.eeg, 0.0, {}),
                "proposed": (proposed.eeg, proposed.info["fraction_modified"], E.detection_outcomes(r, flags)),
                "mwf_oracle": (oracle.eeg, oracle.info["fraction_modified"], {}),
                "auto_mwf": (auto.eeg if hasattr(auto, "eeg") else auto, 1.0,
                             E.detection_outcomes(r, {c: auto_info["flags"].get(c, np.zeros(len(r.starts), bool))
                                                      for c in ("ocular", "muscle", "motion", "pop")})),
            }
            for thr, res in ica.items():
                pipelines[f"ica_iclabel_p{thr}"] = (res.eeg, 1.0, {"ica_removed": str(res.removed_by_class()),
                                                                  "ica_n_removed": len(res.excluded)})
            for name, (eeg, frac, extra) in pipelines.items():
                outputs.setdefault(name, {})[cond] = eeg
                rows.append({"subject": subject, "snr_db": snr, "sar_db": sar, "type": kind, "condition": cond,
                             "pipeline": name, **E.evaluate(eeg, r, base[cond], frac), **extra})
        for name, per_cond in outputs.items():
            fwd = E.forward_prediction(per_cond, recs, base)
            rows.append({"subject": subject, "snr_db": snr, "sar_db": sar, "type": kind, "condition": "both",
                         "pipeline": name, **fwd})
        log.info("%s %s done", subject, scenario)
    return pd.DataFrame(rows)


def run(test_set: str, workers: int = 4, subjects=None) -> pd.DataFrame:
    subjects = subjects or d3.load_noise_split()[test_set]
    out_dir = load_paths()["results"] / "section_a" / f"test_{test_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    with ProcessPoolExecutor(workers) as pool:
        for subject, table in zip(subjects, pool.map(evaluate_subject, subjects, [test_set] * len(subjects))):
            table.to_csv(out_dir / f"{subject}.csv", index=False)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)
