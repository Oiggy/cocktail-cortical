"""Milestone 6: build the artifact template banks (PROJECT_PLAN.md §9.3 step 6).

For Set A and Set B: phantom templates (primary; pools from
splits/phantom.yaml), PhysioMotion templates (secondary; participants from
splits/physiomotion_participants.yaml), and PhysioMotion with broadband
muscle (ablation). Banks are saved under <cache>/templates/<name>.npz;
logic lives in adr.synth.templates (also used by
notebooks/05_templates_injection.ipynb).

    python scripts/10_build_templates.py
"""
import json
import logging

from adr.config import load_paths, package_versions
from adr.synth import templates as T

if __name__ == "__main__":
    out_dir = load_paths()["results"] / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "build_templates.log")])
    log = logging.getLogger("adr.templates")
    log.info("package versions: %s", package_versions())
    summary = {}
    for i, pool in enumerate(("set_a_templates", "set_b_templates")):
        bank, stats = T.phantom_templates(pool, seed=i)
        summary[f"phantom_{pool}"] = {"path": str(T.save_bank(bank, f"phantom_{pool}")), **stats}
        for broadband in (False, True):
            name = f"physiomotion_{pool}" + ("_broadband" if broadband else "")
            bank = T.physiomotion_templates(pool, broadband_muscle=broadband, seed=i)
            summary[name] = {"path": str(T.save_bank(bank, name)),
                             "counts": {k: sum(t.kind == k for t in bank) for k in T.CLASSES}}
    with open(out_dir / "template_banks.json", "w") as f:
        json.dump(summary, f, indent=1)
    log.info("done: %s", list(summary))
