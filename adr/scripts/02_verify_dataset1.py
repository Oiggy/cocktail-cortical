"""Check the Dataset I v2.0 VERIFY items (PROJECT_PLAN.md §3.1, §9.1).

Logic lives in adr.verify.dataset1 (also used by notebooks/01_data_checks.ipynb).
Writes results/verify/dataset1_verify.json and a log.

    python scripts/02_verify_dataset1.py
"""
import json
import logging

from adr import verify
from adr.config import load_paths, package_versions

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "dataset1_verify.log", mode="w")])
    logging.getLogger("adr.verify").info("package versions: %s", package_versions())
    with open(out_dir / "dataset1_verify.json", "w") as f:
        json.dump(verify.dataset1(), f, indent=1)
