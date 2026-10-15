"""Check the PhysioMotion (§3.2) and phantom (§3.3) VERIFY items in PROJECT_PLAN.md.

Logic lives in adr.verify.physiomotion / adr.verify.phantom (also used by
notebooks/01_data_checks.ipynb). Writes results/verify/openneuro_verify.json
and a log.

    python scripts/04_verify_openneuro.py
"""
import json
import logging

from adr import verify
from adr.config import load_paths, package_versions

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "openneuro_verify.log", mode="w")])
    log = logging.getLogger("adr.verify")
    log.info("package versions: %s", package_versions())
    pm = verify.physiomotion()
    log.info("physiomotion: %d subjects, %d ch at %.0f Hz, licence %s, %d shared with BioSemi 64",
             pm["n_subjects"], pm["n_channels"], pm["sfreq"], pm["licence"], len(pm["shared_with_biosemi64"]))
    log.info("physiomotion labels: %s", pm["labels"])
    log.info("physiomotion annotations on %d bipolar pairs over %d electrodes, plus %s; "
             "%d artifact annotations inside %d rest blocks",
             len(pm["annotation_bipolar_pairs"]), len(pm["annotation_electrodes"]),
             pm["annotation_channels_nonbipolar"], pm["artifact_annotations_inside_base_blocks"], pm["n_base_blocks"])
    with open(out_dir / "openneuro_verify.json", "w") as f:
        json.dump({"physiomotion": pm, "phantom": verify.phantom()}, f, indent=1)
