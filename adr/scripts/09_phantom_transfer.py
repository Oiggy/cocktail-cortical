"""Milestone 5c: detector transfer to the phantom (PROJECT_PLAN.md §7).

Logic lives in adr.detection.transfer (also used by notebooks/04_detectors.ipynb).
Writes results/detectors/phantom_transfer.{csv,png} and a log.

    python scripts/09_phantom_transfer.py
"""
import logging

import matplotlib

matplotlib.use("Agg")

from adr.config import load_paths, package_versions
from adr.detection import transfer

if __name__ == "__main__":
    out_root = load_paths()["results"] / "detectors"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_root / "phantom_transfer.log", mode="w")])
    logging.getLogger("adr.phantom_transfer").info("package versions: %s", package_versions())
    table = transfer.run()
    table.to_csv(out_root / "phantom_transfer.csv", index=False)
    transfer.plot(table).savefig(out_root / "phantom_transfer.png", dpi=150)
