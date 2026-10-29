import numpy as np
import pytest

from scout.lightcurve import LightCurve, bin_per_sector


def make(n=100, sector=1, cadence=120):
    t = np.arange(n) * cadence / 86400
    return LightCurve(
        time=t,
        flux=np.ones(n),
        flux_err=np.full(n, 1e-3),
        quality=np.zeros(n),
        sector=np.full(n, sector),
        cadence_s=np.full(n, cadence),
        meta={"tic": 5},
    )


def test_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        LightCurve(
            time=[1, 2],
            flux=[1],
            flux_err=[1, 1],
            quality=[0, 0],
            sector=[1, 1],
            cadence_s=[120, 120],
        )


def test_parquet_roundtrip(tmp_path):
    lc = make()
    lc.trend = np.linspace(1, 2, len(lc))
    lc.meta["nested"] = {"a": [1, 2]}
    p = lc.write(tmp_path / "x" / "lc.parquet")
    back = LightCurve.read(p)
    np.testing.assert_array_equal(back.time, lc.time)
    np.testing.assert_array_equal(back.trend, lc.trend)
    assert back.meta == lc.meta
    assert back.tic == 5
    assert not list((tmp_path / "x").glob("*.tmp"))


def test_concatenate_sorts():
    a, b = make(sector=2), make(sector=1)
    b.time = b.time - 10
    lc = LightCurve.concatenate([a, b])
    assert np.all(np.diff(lc.time) > 0)
    assert lc.sectors == [1, 2]


def test_binning_per_sector():
    late = make(n=30, sector=2, cadence=1800)
    late.time = late.time + 100
    lc = LightCurve.concatenate([make(n=300, sector=1), late])
    b = bin_per_sector(lc, 10.0)
    assert set(np.unique(b.cadence_s[b.sector == 1])) == {600}
    # 30-min sector is already coarser than the bin: untouched.
    assert np.sum(b.sector == 2) == 30
    assert np.sum(b.sector == 1) == 300 // 5
    np.testing.assert_allclose(b.flux_err[b.sector == 1], 1e-3 / np.sqrt(5))
    assert b.meta["binned_minutes"] == 10.0
    assert bin_per_sector(lc, 0) is lc
