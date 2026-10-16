"""Native-transient threshold sensitivity (open item 13, option b; PROJECT_PLAN.md §9.3).

For thresholds of 150, 200, 300, 500 uV peak-to-peak and no cut: how much of
each noise subject's MWF-cleaned base is cut (all 16 subjects; descriptive),
and how base-TRF recovery changes on a Set A subject. Logic in
adr.synth.check. Writes results/section_a/native_cut_table.csv,
native_recovery_sensitivity.csv and a log.

    python scripts/12_native_transients.py
"""
import logging

from adr.config import load_paths, package_versions
from adr.synth import check

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "native_transients.log", mode="w")])
    log = logging.getLogger("adr.dataset3_check")
    log.info("package versions: %s", package_versions())
    cuts = check.native_cut_table()
    cuts.to_csv(out_dir / "native_cut_table.csv", index=False)
    log.info("\n%s", cuts.pivot_table(index="subject", columns="threshold_uv", values="fraction_cut").round(3).to_string())
    check.native_recovery_sensitivity().to_csv(out_dir / "native_recovery_sensitivity.csv", index=False)
