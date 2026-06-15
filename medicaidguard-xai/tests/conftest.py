import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from medicaidguard.config import GeneratorConfig, Settings  # noqa: E402
from medicaidguard.data.generator import SyntheticMedicaidEVVGenerator  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg():
    """Small enough to keep CI under a minute, large enough that every scenario
    injects at least a few rows."""
    return GeneratorConfig(n_claims=4000, n_caregivers=120, n_patients=300,
                           n_providers=12, seed=42)


@pytest.fixture(scope="session")
def tables(small_cfg):
    return SyntheticMedicaidEVVGenerator(small_cfg).generate()


@pytest.fixture(scope="session")
def settings(small_cfg):
    s = Settings()
    s.generator = small_cfg
    return s
