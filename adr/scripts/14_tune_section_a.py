"""Milestone 9a: tune the proposed method on one Section A set and freeze it (PROJECT_PLAN.md §8, §14).

Staged coordinate descent (adr.evaluation.tuning) on the tuning sub-grid of
configs/section_a.yaml; writes results/section_a/tuning_<set>/stage*.csv and
configs/frozen/section_a_tuned_on_<set>.yaml. Run with set_a first; the
cross-fit later runs it with set_b.

    python scripts/14_tune_section_a.py [set_a|set_b] [--workers 5] [--stages 1 2 3 4 5]
"""
import argparse
import logging

from adr.config import load_paths, package_versions
from adr.evaluation import tuning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tuning_set", nargs="?", default="set_a", choices=["set_a", "set_b"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    out_dir = load_paths()["results"] / "section_a"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / f"tuning_{args.tuning_set}.log")])
    logging.getLogger("adr.tuning").info("package versions: %s", package_versions())
    tuning.tune(args.tuning_set, args.workers, tuple(args.stages))
