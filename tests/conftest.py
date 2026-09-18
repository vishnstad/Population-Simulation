from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_CONFIG = REPO_ROOT / "configs" / "gss_main.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def main_config_path() -> Path:
    return MAIN_CONFIG


@pytest.fixture(scope="session")
def cfg():
    from popsim.config import load_config
    return load_config(MAIN_CONFIG)


@pytest.fixture(scope="session")
def data_root(cfg) -> Path:
    return cfg.data_root


def pytest_collection_modifyitems(config, items):
    """Skip microdata-dependent tests when ../data is not present."""
    if (REPO_ROOT / ".." / "data" / "gss").resolve().exists():
        return
    skip = pytest.mark.skip(reason="../data not available")
    for item in items:
        if "slow" in item.keywords or "layer0" in item.keywords:
            item.add_marker(skip)
