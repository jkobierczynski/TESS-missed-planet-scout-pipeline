import dataclasses

import numpy as np
import pytest

from scout.detrend import detrend
from scout.search.periodic import period_range, search_periodic
from scout.stitch import stitch
from scout.synthetic import generate, preset

pytestmark = pytest.mark.slow


def run(name, cfg, sectors=(14, 15), seed=0):
    lc = generate(preset(name, sectors=sectors, seed=seed))
    flat = detrend(stitch(lc, cfg.stitch), cfg.detrend)
    pcfg = dataclasses.replace(cfg.search.periodic, max_signals=2)
    return search_periodic(flat, pcfg), lc


def test_period_range(raw_lc, cfg):
    pmin, pmax = period_range(raw_lc, cfg.search.periodic)
    assert pmin == 0.5
    assert pmax == pytest.approx(0.5 * raw_lc.baseline_days)
    capped = dataclasses.replace(cfg.search.periodic, period_max_days=10.0)
    assert period_range(raw_lc, capped)[1] == 10.0


def test_recovers_sub_neptune(cfg):
    res, lc = run("sub-neptune", cfg)
    injected = lc.meta["synthetic"]["transits"][0]
    best = res.detections[0]
    assert best.passes_threshold
    assert best.period_days == pytest.approx(injected["period_days"], rel=2e-3)
    assert best.duration_hours == pytest.approx(injected["duration_hours"], rel=0.35)
    assert best.depth_ppm == pytest.approx(injected["depth"] * 1e6, rel=0.35)
    assert res.inverted_sde is not None and res.inverted_sde < best.sde
    assert "inverted_sde_ge_signal" not in best.flags


def test_quiet_star_has_no_passing_signal(cfg):
    res, _ = run("quiet", cfg)
    assert not any(d.passes_threshold for d in res.detections)


def test_period_may_be_double_uses_epoch_parity():
    from types import SimpleNamespace

    from scout.search.periodic import period_may_be_double

    def r(epochs_with_data, n=10):
        times = 100.0 + 2.0 * np.arange(n)
        counts = np.zeros(n)
        counts[list(epochs_with_data)] = 30
        return SimpleNamespace(period=2.0, T0=100.0, transit_times=times, per_transit_count=counts)

    # Most predicted transits fall in gaps, but odd and even epochs both have data.
    assert not period_may_be_double(r([0, 1, 7]))
    assert period_may_be_double(r([0, 2, 6]))
    assert period_may_be_double(r([3, 9]))
    assert not period_may_be_double(r([4]))  # one transit says nothing


def _box_lc(period, depth=2e-3, dur_h=4.0, n_days=54.0, noise=3e-4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1700.0, 1700.0 + n_days, 10 / 1440)
    phase = (t - 1700.5 + 0.5 * period) % period - 0.5 * period
    y = 1.0 - depth * (np.abs(phase) < dur_h / 48) + rng.normal(0, noise, len(t))
    return t, y, np.full(len(t), noise)


def test_resolve_submultiple_finds_true_period():
    from scout.search.periodic import resolve_submultiple

    t, y, dy = _box_lc(1.3)
    # TLS reporting 3× or 2× the true period: the in-between epochs give it away.
    assert resolve_submultiple(t, y, dy, 3.9, 1700.5, 4.0, 5, 0.5) == 3
    assert resolve_submultiple(t, y, dy, 2.6, 1700.5, 4.0, 5, 0.5) == 2
    # Right period: nothing in between, keep it.
    assert resolve_submultiple(t, y, dy, 1.3, 1700.5, 4.0, 5, 0.5) == 1
    # Never below the search's own minimum period.
    assert resolve_submultiple(t, y, dy, 3.9, 1700.5, 4.0, 5, 1.5) == 1


def test_resolve_submultiple_ignores_eclipsing_binary_secondary():
    from scout.search.periodic import resolve_submultiple

    t, y, dy = _box_lc(4.0, depth=5e-3)
    # A shallow secondary at phase 0.5 is not "the same dip".
    phase = (t - 1700.5 - 2.0 + 2.0) % 4.0 - 2.0
    y = y - 5e-4 * (np.abs(phase) < 4.0 / 48)
    assert resolve_submultiple(t, y, dy, 4.0, 1700.5, 4.0, 5, 0.5) == 1


def test_short_period_planet_on_large_star_not_aliased(cfg):
    from scout.synthetic import SyntheticStar, Transit, sector_window

    # 1.1 d around a 2.2 R☉ star: T14 ≈ 4.7 h, 18 % of the period, beyond TLS's 12 % cap.
    tr = Transit(
        t0_btjd=sector_window(14)[0] + 0.7, period_days=1.1, depth=2e-3, duration_hours=4.7
    )
    star = SyntheticStar(
        sectors=(14, 15), noise_ppm=600.0, radius_rsun=2.2, mass_msun=1.1, transits=(tr,)
    )
    flat = detrend(stitch(generate(star), cfg.stitch), cfg.detrend)
    pcfg = dataclasses.replace(cfg.search.periodic, max_signals=1, run_inverted=False)
    best = search_periodic(flat, pcfg).detections[0]
    assert best.period_days == pytest.approx(1.1, rel=2e-3)
    assert "period_divided_by_2" in best.flags
    # Without the check TLS reports twice the period.
    off = dataclasses.replace(pcfg, resolve_submultiples=False)
    assert search_periodic(flat, off).detections[0].period_days == pytest.approx(2.2, rel=2e-3)
