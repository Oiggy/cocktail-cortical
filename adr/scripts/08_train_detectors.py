"""Milestone 5b: train and cross-validate the detectors on PhysioMotion (PROJECT_PLAN.md §7).

Logic lives in adr.detection.train (also used by notebooks/04_detectors.ipynb).
Writes results/detectors/<rate>Hz/{cv_metrics.csv, <model>_<target>.joblib,
feature_names.txt}, cv_pr_auc.png and a log.

    python scripts/08_train_detectors.py [--rates 128 250] [--models lightgbm lda svm]
"""
import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from adr.config import load_paths, package_versions, seed_everything
from adr.detection import models as M
from adr.detection import train
from adr.detection.features import detection_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=float, default=[float(r) for r in detection_config()["rates"]])
    parser.add_argument("--models", nargs="+", default=list(M.MODELS))
    args = parser.parse_args()
    out_root = load_paths()["results"] / "detectors"
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_root / "train_detectors.log")])
    logging.getLogger("adr.train_detectors").info("package versions: %s", package_versions())
    seed_everything()
    tables = []
    for fs in args.rates:
        data = train.load_rate(fs)
        table = train.cross_validate(fs, args.models, data=data)
        table.to_csv(train.output_dir(fs) / "cv_metrics.csv", index=False)
        train.fit_final(fs, args.models, data=data)
        tables.append(table)
    train.plot_cv(pd.concat(tables)).savefig(out_root / "cv_pr_auc.png", dpi=150)
