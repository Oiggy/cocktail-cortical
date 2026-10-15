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
    assert {"dataset1", "physiomotion", "phantom", "own", "cache", "results", "splits"} <= paths.keys()
    assert all(p.is_absolute() for p in paths.values())
    assert paths["splits"] == ADR_ROOT / "splits"


def test_shared_channels():
    project = load_project()
    channels = project["fallback_channels"]
    assert len(channels) == len(set(channels)) == 19
    # Renamed legacy labels must land on fallback channels.
    assert set(project["legacy_rename"].values()) <= set(channels)
    assert not set(project["legacy_rename"]) & set(channels)


def test_seed_is_deterministic():
    a = seed_everything(1).standard_normal(5)
    b = seed_everything(1).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_package_versions():
    versions = package_versions()
    assert versions["numpy"] != "not installed"


def test_detector_channels():
    import mne
    project = load_project()
    detector = project["detector_channels"]
    biosemi = mne.channels.make_standard_montage("biosemi64").ch_names
    assert len(detector) == len(set(detector)) == 57
    assert detector == [c for c in biosemi if c in detector]  # BioSemi order
    assert set(project["label_electrodes"]) <= set(detector)


def test_phantom_split():
    import yaml
    with open(ADR_ROOT / "splits" / "phantom.yaml") as f:
        split = yaml.safe_load(f)
    blocks = sorted(split["pools"].values())
    assert len(blocks) == 3
    # Contiguous, disjoint, and covering the whole trial.
    assert blocks[0][0] == 0 and blocks[-1][1] == split["trial_seconds"]
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
