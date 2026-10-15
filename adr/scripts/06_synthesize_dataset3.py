"""Milestone 4: Dataset III synthesis check (PROJECT_PLAN.md §9.2-9.3 steps 1-5).

Logic lives in adr.synth.check (also used by notebooks/03_dataset3_synthesis.ipynb).
Writes results/section_a/base_trfs.npz, dataset3_check.{csv,png} and a log.

    python scripts/06_synthesize_dataset3.py [SUBJECT]
"""
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

from adr.config import load_paths, package_versions, seed_everything
from adr.synth import check
from adr.synth import dataset3 as d3

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "dataset3_check.log", mode="w")])
    logging.getLogger("adr.dataset3_check").info("package versions: %s", package_versions())
    seed_everything()
    base, lags, names = d3.load_base_trfs()
    np.savez(out_dir / "base_trfs.npz", base_left=base["L"], base_right=base["R"], lags=lags, ch_names=np.array(names))
    table = check.run(sys.argv[1] if len(sys.argv) > 1 else None)
    table.to_csv(out_dir / "dataset3_check.csv", index=False)
    check.plot(table).savefig(out_dir / "dataset3_check.png", dpi=150)
