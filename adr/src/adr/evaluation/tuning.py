"""Section A tuning on one noise-subject set, then freezing (PROJECT_PLAN.md §8, §9.3, §14; milestone 9).

Staged coordinate descent over the proposed method's settings, on the
tuning sub-grid of configs/section_a.yaml (`tuning`), using only the tuning
set's noise subjects and template pool:

1. detector variant (raw / norm) x one threshold shared by all classes;
2. per-class thresholds, one class at a time (others fixed);
3. MWF rank x ocular method (MWF / MWF + frontal ridge);
4. interpolation neighbours x contamination z, and the motion method;
5. boundary taper x over-cleaning cap.

Each configuration is scored by the mean over recordings of the median
short-term TRF relative MSE (primary); configurations within 1 % of the
best are ranked by the 1-30 Hz relative RMS error (secondary). Detector
scores are computed once per recording and variant. The winner is written
to configs/frozen/section_a_tuned_on_<set>.yaml and never changed after
the test set is touched.
"""
from __future__ import annotations

import copy
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import yaml

from adr.config import ADR_ROOT, load_paths, load_yaml
from adr.evaluation import section_a as E
from adr.synth import dataset3 as d3
from adr.synth import templates as T

log = logging.getLogger("adr.tuning")
CLASSES = ("ocular", "muscle", "motion", "pop")
TOLERANCE = 0.01


@dataclass
class Config:
    name: str
    variant: str
    thresholds: dict
    recon: dict = field(default_factory=dict)   # overrides of configs/reconstruction.yaml (dotted keys)

    def recon_cfg(self) -> dict:
        cfg = copy.deepcopy(load_yaml("reconstruction"))
        for key, value in self.recon.items():
            node = cfg
            *path, leaf = key.split(".")
            for p in path:
                node = node[p]
            node[leaf] = value
        return cfg

    def as_dict(self) -> dict:
        return {"variant": self.variant, "thresholds": dict(self.thresholds), "reconstruction": dict(self.recon)}


def bank_name(tuning_set: str) -> str:
    return f"phantom_{tuning_set}_templates"


def scenarios(subject: str, grid: dict) -> list[E.Scenario]:
    return [E.Scenario(subject, snr, sar, kind) for snr, sar, kind in product(grid["snr_db"], grid["sar_db"], grid["types"])]


def evaluate_subject(subject: str, configs: list[Config], tuning_set: str, grid: dict) -> pd.DataFrame:
    base, lags, _ = d3.load_base_trfs()
    bank = T.load_bank(bank_name(tuning_set))
    source = d3.noise_source(subject)
    variants = sorted({c.variant for c in configs})
    recon_cfgs = {c.name: c.recon_cfg() for c in configs}
    rows = []
    for scenario in scenarios(subject, grid):
        for condition in d3.CONDITIONS:
            r = E.build(source, scenario, condition, bank, base, lags)
            scores = {v: E.recording_scores(r.injected.eeg, r.rec.ch_names, v) for v in variants}
            for cfg in configs:
                result = E.run_proposed(r, scores[cfg.variant], cfg.thresholds, recon_cfgs[cfg.name])
                metrics = E.evaluate(result.eeg, r, base[condition], result.info["fraction_modified"], light=True)
                rows.append({"subject": subject, "snr_db": scenario.snr_db, "sar_db": scenario.sar_db,
                             "type": scenario.kind, "condition": condition, "config": cfg.name, **metrics})
    return pd.DataFrame(rows)


def evaluate_configs(configs: list[Config], subjects: list[str], tuning_set: str, grid: dict, workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as pool:
        tables = list(pool.map(evaluate_subject, subjects, [configs] * len(subjects), [tuning_set] * len(subjects),
                               [grid] * len(subjects)))
    return pd.concat(tables, ignore_index=True)


def select(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Best config: lowest mean TRF relMSE; within 1 % of it, lowest mean 1-30 Hz error."""
    summary = table.groupby("config")[["trf_relmse", "rrmse_1_30", "fraction_modified"]].mean().sort_values("trf_relmse")
    best_primary = summary["trf_relmse"].min()
    near = summary[summary["trf_relmse"] <= best_primary * (1 + TOLERANCE)]
    return near["rrmse_1_30"].idxmin(), summary


def stage_configs(stage: int, best: Config) -> list[Config]:
    def with_(name, **kw):
        cfg = Config(name, kw.get("variant", best.variant), dict(kw.get("thresholds", best.thresholds)),
                     {**best.recon, **kw.get("recon", {})})
        return cfg

    if stage == 1:
        return [with_(f"s1_{v}_t{t}", variant=v, thresholds={c: t for c in CLASSES})
                for v, t in product(("raw", "norm"), (0.5, 0.7, 0.85, 0.95))]
    if stage == 2:
        out = [with_("s2_current")]
        for c, t in product(CLASSES, (0.5, 0.7, 0.85, 0.95, 0.99)):
            if t != best.thresholds[c]:
                out.append(with_(f"s2_{c}_t{t}", thresholds={**best.thresholds, c: t}))
        return out
    if stage == 3:
        return [with_(f"s3_rank{r}_{m}", recon={"mwf.rank": r, "ocular_method": m})
                for r, m in product(("eig>2", "eig>4", "eig>8"), ("mwf", "mwf+ridge"))]
    if stage == 4:
        return ([with_(f"s4_k{k}_z{z}", recon={"spatial_interp.n_neighbours": k, "spatial_interp.bad_z": z})
                 for k, z in product((4, 8, 12), (3.0, 5.0))]
                + [with_("s4_motion_interp", recon={"motion_method": "interp"})])
    if stage == 5:
        return [with_(f"s5_taper{t}_cap{c}", recon={"taper_s": t, "max_modified_fraction": c})
                for t, c in product((0.0, 0.25, 0.5), (0.3, 0.5, 1.0))]
    raise ValueError(stage)


def tune(tuning_set: str = "set_a", workers: int = 5, stages=(1, 2, 3, 4, 5)) -> Config:
    grid = load_yaml("section_a")["tuning"]
    subjects = d3.load_noise_split()[tuning_set]
    out_dir = load_paths()["results"] / "section_a" / f"tuning_{tuning_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = Config("start", "raw", {c: 0.85 for c in CLASSES})
    for stage in stages:
        configs = stage_configs(stage, best)
        log.info("stage %d: %d configs on %s (%d subjects)", stage, len(configs), tuning_set, len(subjects))
        table = evaluate_configs(configs, subjects, tuning_set, grid, workers)
        table.to_csv(out_dir / f"stage{stage}.csv", index=False)
        name, summary = select(table)
        summary.to_csv(out_dir / f"stage{stage}_summary.csv")
        best = next(c for c in configs if c.name == name)
        log.info("stage %d best: %s\n%s", stage, name, summary.round(4).to_string())
    freeze(best, tuning_set)
    return best


def frozen_path(tuning_set: str):
    return ADR_ROOT / "configs" / "frozen" / f"section_a_tuned_on_{tuning_set}.yaml"


def freeze(best: Config, tuning_set: str) -> None:
    path = frozen_path(tuning_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Section A settings tuned on {tuning_set} (adr.evaluation.tuning), frozen.\n"
              f"# Never change after the other set has been touched (PROJECT_PLAN.md §14).\n")
    with open(path, "w") as f:
        f.write(header + yaml.safe_dump({"tuned_on": tuning_set, "config_name": best.name, **best.as_dict()},
                                        sort_keys=False))
    log.info("froze %s -> %s", best.name, path)


def load_frozen(tuning_set: str) -> Config:
    with open(frozen_path(tuning_set)) as f:
        d = yaml.safe_load(f)
    return Config(d["config_name"], d["variant"], d["thresholds"], d.get("reconstruction", {}))
