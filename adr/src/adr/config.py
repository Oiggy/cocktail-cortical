"""Config loading, seeding and environment logging shared by every script.

Every script reads paths and parameters from adr/configs/ (PROJECT_PLAN.md
§15), so nothing depends on the machine or on where the repo is cloned.
"""
from __future__ import annotations

import importlib.metadata
import platform
import random
from pathlib import Path

import numpy as np
import yaml

ADR_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ADR_ROOT / "configs"
# The folder holding cocktail-cortical/ and its sibling dataset/ folder.
WORKSPACE_ROOT = ADR_ROOT.parents[1]

LOGGED_PACKAGES = (
    "numpy", "scipy", "pandas", "mne", "scikit-learn", "lightgbm",
    "xgboost", "eelbrain", "matplotlib", "mne-icalabel",
)


def load_yaml(name: str) -> dict:
    """Read configs/<name>.yaml, overlaid with configs/<name>.local.yaml if present."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        config = yaml.safe_load(f) or {}
    local = CONFIG_DIR / f"{name}.local.yaml"
    if local.exists():
        with open(local) as f:
            for key, value in (yaml.safe_load(f) or {}).items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key] = {**config[key], **value}
                else:
                    config[key] = value
    return config


def load_paths() -> dict[str, Path]:
    """Dataset and output locations as absolute paths.

    Relative dataset entries resolve against WORKSPACE_ROOT, relative output
    entries against ADR_ROOT; absolute entries are used as given.
    """
    config = load_yaml("paths")
    paths = {}
    for section, base in (("datasets", WORKSPACE_ROOT), ("outputs", ADR_ROOT)):
        for key, value in config[section].items():
            paths[key] = (base / Path(value).expanduser()).resolve()
    return paths


def load_project() -> dict:
    return load_yaml("project")


def seed_everything(seed: int | None = None) -> np.random.Generator:
    """Seed the global RNGs and return a Generator for explicit use.

    Defaults to the project seed in configs/project.yaml.
    """
    if seed is None:
        seed = load_project()["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def package_versions() -> dict[str, str]:
    """Versions of the core stack, for writing into every run's log."""
    versions = {"python": platform.python_version()}
    for name in LOGGED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
