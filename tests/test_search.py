import dataclasses

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
