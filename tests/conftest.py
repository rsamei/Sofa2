import pytest

from sofa2.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def s2(cfg):
    return cfg.sofa2


@pytest.fixture(scope="session")
def s1(cfg):
    return cfg.sofa1
