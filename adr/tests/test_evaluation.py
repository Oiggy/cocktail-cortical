import pandas as pd

from adr.evaluation import section_a as E
from adr.evaluation import tuning


def test_scenario_seed_is_deterministic_and_distinct():
    a = E.Scenario("S4", -15, -10, "ocular")
    assert a.seed("L") == E.Scenario("S4", -15, -10, "ocular").seed("L")
    assert a.seed("L") != a.seed("R") and a.seed("L") != E.Scenario("S5", -15, -10, "ocular").seed("L")


def test_config_overrides_reconstruction():
    cfg = tuning.Config("x", "raw", {c: 0.5 for c in tuning.CLASSES},
                        {"mwf.rank": "eig>8", "taper_s": 0.0, "spatial_interp.bad_z": 5.0})
    rc = cfg.recon_cfg()
    assert rc["mwf"]["rank"] == "eig>8" and rc["taper_s"] == 0.0 and rc["spatial_interp"]["bad_z"] == 5.0


def test_select_uses_secondary_within_tolerance():
    table = pd.DataFrame({
        "config": ["a", "a", "b", "b", "c", "c"],
        "trf_relmse": [0.200, 0.200, 0.201, 0.201, 0.25, 0.25],   # b within 1 % of a
        "rrmse_1_30": [0.60, 0.60, 0.40, 0.40, 0.10, 0.10],       # c best secondary but too far on primary
        "fraction_modified": [0.1] * 6,
    })
    best, summary = tuning.select(table, "trf_relmse", "rrmse_1_30")
    assert best == "b" and list(summary.index)[0] == "a"


def test_select_default_objective_is_fidelity_first():
    table = pd.DataFrame({
        "config": ["a", "b", "c"],
        "rrmse_1_9": [0.300, 0.302, 0.40],    # b within 1 % of a on the primary
        "trf_relmse": [0.27, 0.26, 0.20],     # c best TRF but too far on fidelity
        "rrmse_1_30": [0.5, 0.5, 0.5],
        "fraction_modified": [0.2] * 3,
    })
    best, summary = tuning.select(table)
    assert best == "b" and list(summary.index)[0] == "a"


def test_stage_grids_cover_plan_hyperparameters():
    start = tuning.Config("start", "raw", {c: 0.85 for c in tuning.CLASSES})
    names = [c.name for s in (1, 2, 3, 4, 5) for c in tuning.stage_configs(s, start)]
    assert any("norm" in n for n in names) and any("rank" in n for n in names)
    assert any("taper" in n for n in names) and any("_k" in n for n in names)
    assert len(tuning.stage_configs(1, start)) == 8
