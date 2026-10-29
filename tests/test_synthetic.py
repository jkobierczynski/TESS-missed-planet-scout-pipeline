import numpy as np
import pytest

from scout.synthetic import (
    MOMENTUM_DUMP_BIT,
    PRESETS,
    SyntheticStar,
    Transit,
    generate,
    preset,
    sector_window,
    transit_model,
    trapezoid_model,
)


def test_reproducible():
    s = preset("sub-neptune", seed=7)
    a, b = generate(s), generate(s)
    np.testing.assert_array_equal(a.flux, b.flux)
    c = generate(preset("sub-neptune", seed=8))
    assert not np.array_equal(a.flux, c.flux)


def test_structure():
    lc = generate(SyntheticStar(sectors=(14, 16)))
    assert lc.sectors == [14, 16]
    assert lc.tic == 999_000_001
    assert lc.meta["star"]["radius_rsun"] == 1.0
    start, end = sector_window(14)
    t14 = lc.time[lc.sector == 14]
    assert start <= t14.min() and t14.max() < end
    # Orbit gap in the middle of the sector.
    assert np.max(np.diff(t14)) > 0.9
    assert np.any(lc.quality & MOMENTUM_DUMP_BIT)
    # Un-normalised: different sector levels around 1e4.
    meds = [np.median(lc.flux[lc.sector == s]) for s in lc.sectors]
    assert all(9000 < m < 11000 for m in meds)


@pytest.mark.parametrize("model", [trapezoid_model, transit_model])
def test_transit_shape(model):
    tr = Transit(t0_btjd=10.0, period_days=3.0, depth=1e-3, duration_hours=2.4, impact=0.2)
    t = np.linspace(5, 15, 200_001)
    f = model(t, tr)
    assert f.max() == pytest.approx(1.0)
    # Centre depth close to nominal (limb darkening deepens it a little).
    assert 0.9e-3 < 1 - f[np.argmin(np.abs(t - 10.0))] < 1.35e-3
    # Same depth one period later.
    assert f[np.argmin(np.abs(t - 13.0))] == pytest.approx(f[np.argmin(np.abs(t - 10.0))], rel=1e-4)
    # Total duration roughly as requested.
    intransit = t[(f < 1 - 1e-6) & (np.abs(t - 10) < 1)]
    assert (intransit.max() - intransit.min()) * 24 == pytest.approx(2.4, rel=0.1)


def test_single_event():
    tr = Transit(t0_btjd=10.0, depth=1e-3, duration_hours=5.0)
    t = np.linspace(0, 100, 100_001)
    f = transit_model(t, tr)
    dips = t[f < 1 - 1e-6]
    assert dips.min() > 9.5 and dips.max() < 10.5


@pytest.mark.parametrize("name", PRESETS)
def test_presets(name):
    lc = generate(preset(name, sectors=(14,)))
    assert len(lc) > 10_000


def test_unknown_preset():
    with pytest.raises(KeyError):
        preset("nope")
