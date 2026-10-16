"""Milestone 9b: evaluate Section A on the held-out set with frozen settings (PROJECT_PLAN.md §9.4, §14).

Requires configs/frozen/section_a_tuned_on_<other set>.yaml (scripts/14).
Writes results/section_a/test_<set>/<subject>.csv and test_<set>.csv.
set_b is the main test fold; set_a is the cross-fit fold (after tuning on set_b).

    python scripts/15_evaluate_section_a.py [set_b|set_a] [--workers 4]
"""
import argparse
import logging

from adr.config import load_paths, package_versions
from adr.evaluation import test_set, tuning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_set", nargs="?", default="set_b", choices=["set_a", "set_b"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not tuning.frozen_path(test_set.OTHER[args.test_set]).exists():
        raise SystemExit(f"Frozen settings from {test_set.OTHER[args.test_set]} not found - tune first (scripts/14).")
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / f"test_{args.test_set}.log")])
    logging.getLogger("adr.test_set").info("package versions: %s", package_versions())
    table = test_set.run(args.test_set, args.workers)
    table.to_csv(out_dir / f"test_{args.test_set}.csv", index=False)
