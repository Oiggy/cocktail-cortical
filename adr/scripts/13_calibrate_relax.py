"""Calibrate the RELAX-style muscle slope threshold on Section A Set A (open item 14, option b).

Finds the slope threshold that marks 5 % of artifact-free Set A windows
(adr.baselines.mwf_auto.calibrate_muscle_threshold) and checks muscle
sensitivity on Set A recordings with injected phantom muscle templates.
Writes results/section_a/relax_muscle_calibration.{json,_check.csv}. The
frozen value lives in configs/baselines.yaml (copied there by hand after
this run, 2026-10-15); never re-run after Set B has been touched.

    python scripts/13_calibrate_relax.py
"""
import json
import logging

import numpy as np
import pandas as pd

from adr.baselines import mwf_auto
from adr.config import load_paths, package_versions
from adr.synth import dataset3 as d3
from adr.synth import inject as I
from adr.synth import templates as T

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("adr.mwf_auto")
    log.info("package versions: %s", package_versions())
    threshold, details = mwf_auto.calibrate_muscle_threshold(target_rate=0.05)
    log.info("calibrated threshold: %.3f", threshold)
    with open(out_dir / "relax_muscle_calibration.json", "w") as f:
        json.dump({"threshold": threshold, "target_rate": 0.05, **details}, f, indent=1)
    base, lags, _ = d3.load_base_trfs()
    bank = T.load_bank("phantom_set_a_templates")
    rows = []
    for subject in d3.load_noise_split()["set_a"]:
        rec = d3.synthesize(d3.noise_source(subject), "L", -10, base, lags)
        for sar in (0, -10):
            inj = I.inject(rec.eeg, 128.0, "muscle", sar, np.random.default_rng(1), bank=bank, ch_names=list(rec.ch_names))
            truth = I.window_truth(inj.log, len(rec.eeg), 128.0, 1.0, 0.5)["muscle"].to_numpy()
            slopes = mwf_auto.muscle_slopes(inj.eeg, 128.0)
            n = min(len(slopes), len(truth))
            for name, thr in (("calibrated", threshold), ("RELAX default", -0.59)):
                marked = slopes[:n] > thr
                rows.append({"subject": subject, "sar_db": sar, "threshold": name,
                             "sensitivity": marked[truth[:n]].mean(), "false_mark_rate": marked[~truth[:n]].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "relax_muscle_calibration_check.csv", index=False)
    log.info("\n%s", table.groupby(["sar_db", "threshold"])[["sensitivity", "false_mark_rate"]].mean().round(3))
