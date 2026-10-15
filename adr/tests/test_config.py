import numpy as np

from adr.config import (
    ADR_ROOT, WORKSPACE_ROOT, load_paths, load_project, package_versions,
    seed_everything,
)


def test_roots():
    assert (ADR_ROOT / "PROJECT_PLAN.md").exists()
    assert (WORKSPACE_ROOT / "cocktail-cortical").is_dir()


def test_paths_are_absolute():
    paths = load_paths()
    assert {"dataset1", "tuar", "own", "cache", "results", "splits"} <= paths.keys()
    assert all(p.is_absolute() for p in paths.values())
    assert paths["splits"] == ADR_ROOT / "splits"


def test_shared_channels():
    project = load_project()
    channels = project["shared_channels"]
    assert len(channels) == len(set(channels)) == 19
    # Renamed TUAR labels must land on shared channels; dropped ones must not.
    assert set(project["tuar_rename"].values()) <= set(channels)
    assert not set(project["tuar_drop"]) & set(channels)


def test_seed_is_deterministic():
    a = seed_everything(1).standard_normal(5)
    b = seed_everything(1).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_package_versions():
    versions = package_versions()
    assert versions["numpy"] != "not installed"
