import numpy as np
import pytest

from scout.config import load_config
from scout.synthetic import SyntheticStar, Transit, generate


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def transit():
    return Transit(t0_btjd=1686.0, period_days=5.5, depth=2e-3, duration_hours=3.0)


@pytest.fixture
def raw_lc(transit):
    star = SyntheticStar(
        sectors=(14, 15),
        noise_ppm=300.0,
        rotation_period_days=8.0,
        rotation_amplitude=0.003,
        transits=(transit,),
        seed=3,
    )
    return generate(star)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
