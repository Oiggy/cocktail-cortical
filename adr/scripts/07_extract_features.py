"""Milestone 5a: detector features for the PhysioMotion detector-training group.

For every run of every detector-training participant
(splits/physiomotion_participants.yaml) and each detector rate in
configs/detection.yaml, computes window features, labels and pop examples
(adr.detection.dataset.run_features), cached under <cache>/features/.

    python scripts/07_extract_features.py [--workers 4]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from adr.config import ADR_ROOT, load_paths, package_versions
from adr.detection.dataset import run_features
from adr.detection.features import detection_config
from adr.io.physiomotion import RUNS

log = logging.getLogger("extract_features")


def job(subject: int, run: int, fs: float) -> str:
    out = run_features(subject, run, fs)
    n = len(out["X"])
    return (f"sub-{subject} run {run} {int(fs)} Hz: {n} windows, rest {out['rest'].sum()}, "
            f"ocular {out['ocular'].sum()}, muscle {out['muscle'].sum()}, motion {out['motion'].sum()}, "
            f"unlabelled {out['unlabelled'].sum()}, pop examples {len(out['X_pop'])}, bads {list(out['bads'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    log_dir = load_paths()["results"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "07_extract_features.log")])
    log.info("package versions: %s", package_versions())
    with open(ADR_ROOT / "splits" / "physiomotion_participants.yaml") as f:
        subjects = yaml.safe_load(f)["groups"]["detector_training"]
    jobs = [(s, r, float(fs)) for fs in detection_config()["rates"] for s in subjects for r in RUNS]
    log.info("%d jobs (%d participants x %d runs x %d rates)", len(jobs), len(subjects), len(RUNS),
             len(detection_config()["rates"]))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, *j) for j in jobs]
        for future in as_completed(futures):
            log.info(future.result())
    log.info("done")


if __name__ == "__main__":
    main()
