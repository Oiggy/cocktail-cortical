"""Milestone 3: long-term TRFs on Dataset I v2.0 (PROJECT_PLAN.md §9.2 steps 1-2).

Logic lives in adr.trf.long_term (also used by notebooks/02_long_term_trf.ipynb).
Writes results/section_a/long_term_trf.npz, long_term_trf.png and a log.

    python scripts/05_long_term_trf.py
"""
import logging

import matplotlib

matplotlib.use("Agg")

from adr.config import load_paths, package_versions, seed_everything
from adr.trf import long_term

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "section_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "long_term_trf.log", mode="w")])
    log = logging.getLogger("adr.long_term_trf")
    log.info("package versions: %s", package_versions())
    seed_everything()
    result = long_term.run()
    c5 = result.ch_names.index("C5")
    for ear in "LR":
        log.info("W_%s at C5 (lag ms: weight): %s", ear, ", ".join(
            f"{1000 * l / 32:.0f}: {v:+.3f}" for l, v in zip(result.lags, result.w[ear].weights[:, c5])))
    log.info("fraction of trials with mean r above their null: %.2f",
             (result.cv_r.mean(1) > result.cv_r_null.mean(1)).mean())
    long_term.save(result)
    long_term.plot(result).savefig(out_dir / "long_term_trf.png", dpi=150)
    log.info("saved %s", out_dir / "long_term_trf.png")
