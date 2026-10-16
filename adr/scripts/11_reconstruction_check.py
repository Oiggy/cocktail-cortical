"""Milestone 7: reconstruction sanity check with oracle flags (PROJECT_PLAN.md §8).

Routed reconstruction vs a single MWF vs no correction on a Set A noise
subject, one artifact type at a time, at SAR 0 and -10 dB, given the true
(injection-log) flags. Logic lives in adr.reconstruction.check (also used by
notebooks/06_reconstruction.ipynb). Writes
results/section_a/reconstruction_check.csv and a log.

    python scripts/11_reconstruction_check.py [SUBJECT]
"""
import logging
import sys

import pandas as pd

from adr.config import load_paths, package_versions
from adr.reconstruction import check

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "reconstruction_check.log", mode="w")])
    logging.getLogger("adr.reconstruction_check").info("package versions: %s", package_versions())
    subject = sys.argv[1] if len(sys.argv) > 1 else None
    table = pd.concat([check.run(subject, sar_db=sar) for sar in (0, -10)])
    table.to_csv(out_dir / "reconstruction_check.csv", index=False)
