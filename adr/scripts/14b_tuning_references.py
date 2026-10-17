"""Reference scores on a tuning grid (milestone 9 review): no correction, artifact-free, oracle single MWF.

Uses the frozen reconstruction settings for the oracle MWF. Writes
results/section_a/tuning_set_a/references.csv.

    python scripts/14b_tuning_references.py [workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from adr.config import load_paths, load_yaml
from adr.evaluation import section_a as E
from adr.evaluation import tuning as TU
from adr.synth import dataset3 as d3
from adr.synth import templates as T


def run(subject):
    grid = load_yaml("section_a")["tuning"]
    base, lags, _ = d3.load_base_trfs()
    bank = T.load_bank(TU.bank_name("set_a"))
    source = d3.noise_source(subject)
    recon = TU.load_frozen("set_a").recon_cfg()
    rows = []
    for sc in TU.scenarios(subject, grid):
        for cond in d3.CONDITIONS:
            r = E.build(source, sc, cond, bank, base, lags)
            oracle = E.run_oracle_mwf(r, recon)
            for name, eeg, frac in (("no correction", r.injected.eeg, 0.0), ("artifact-free", r.rec.eeg, 0.0),
                                    ("oracle single MWF", oracle.eeg, oracle.info["fraction_modified"])):
                rows.append({"subject": subject, "snr_db": sc.snr_db, "sar_db": sc.sar_db, "type": sc.kind,
                             "condition": cond, "config": name, **E.evaluate(eeg, r, base[cond], frac, light=True)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    subjects = d3.load_noise_split()["set_a"]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool:
        table = pd.concat(pool.map(run, subjects), ignore_index=True)
    out = load_paths()["results"] / "section_a" / "tuning_set_a" / "references.csv"
    table.to_csv(out, index=False)
    print(table.groupby("config")[["trf_relmse", "rrmse_1_30", "fraction_modified"]].mean().round(4))
    print("saved", out)
