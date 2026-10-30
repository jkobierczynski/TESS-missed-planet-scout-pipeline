import json

import numpy as np
import pytest

from scout.cli import main
from scout.config import InjectConfig
from scout.inject import (
    Injection,
    choose_thresholds,
    draw_injection,
    match,
    scramble,
    transit_duration_hours,
)
from scout.search.periodic import Detection


def det(period, t0, sde=20.0, snr=15.0, n=5, index=1):
    return Detection(
        tic=1,
        kind="periodic",
        index=index,
        period_days=period,
        period_err_days=1e-4,
        t0_btjd=t0,
        duration_hours=3.0,
        depth_ppm=1000.0,
        snr=snr,
        sde=sde,
        n_transits=n,
        n_transits_expected=n,
        odd_even_sigma=0.0,
        rp_rs=0.03,
        passes_threshold=True,
    )


def inj(period=5.0, t0=1700.0, dur=3.0):
    return Injection(
        index=0,
        period_days=period,
        t0_btjd=t0,
        radius_rearth=2.0,
        impact=0.3,
        rp_rs=0.02,
        depth=4e-4,
        duration_hours=dur,
    )


def test_transit_duration_physics():
    # Earth around the Sun: ~13 h central transit.
    assert transit_duration_hours(365.25, 0.009, 0.0, 1.0, 1.0) == pytest.approx(13.0, rel=0.03)
    # Shorter for a grazing chord, zero for a miss.
    assert transit_duration_hours(5.0, 0.02, 0.8, 1.0, 1.0) < transit_duration_hours(
        5.0, 0.02, 0.0, 1.0, 1.0
    )
    assert transit_duration_hours(1.0, 0.02, 10.0, 1.0, 1.0) == 0.0


def test_draws_reproducible_and_in_range(raw_lc):
    icfg = InjectConfig(radius_min_rearth=1.0, radius_max_rearth=4.0, impact_max=0.5)
    a = [draw_injection(i, 42, raw_lc, icfg, 20.0) for i in range(30)]
    b = [draw_injection(i, 42, raw_lc, icfg, 20.0) for i in range(30)]
    assert a == b
    # Draw i does not depend on how many draws there are (runs resume cleanly).
    assert draw_injection(7, 42, raw_lc, icfg, 20.0) == a[7]
    assert draw_injection(7, 43, raw_lc, icfg, 20.0) != a[7]
    for x in a:
        assert 0.5 <= x.period_days <= 20.0
        assert 1.0 <= x.radius_rearth <= 4.0
        assert 0.0 <= x.impact <= 0.5
        assert raw_lc.time.min() <= x.t0_btjd <= raw_lc.time.min() + x.period_days
        assert x.duration_hours > 0


def test_match_recovered_alias_missed():
    i = inj()
    assert match(i, [det(5.0005, 1700.0 + 3 * 5.0)], 0.01)[0] == "recovered"
    # Right period, wrong epoch.
    assert match(i, [det(5.0, 1702.0)], 0.01)[0] == "missed"
    assert match(i, [det(10.0, 1700.0)], 0.01)[0] == "alias"
    assert match(i, [det(2.5, 1702.5)], 0.01)[0] == "alias"
    assert match(i, [det(3.7, 1700.0)], 0.01)[0] == "missed"
    # A later signal can be the one that matches.
    status, d = match(i, [det(3.7, 1700.0), det(5.0, 1700.0, index=2)], 0.01)
    assert status == "recovered" and d.index == 2


def test_scramble_keeps_times_and_values_per_sector(raw_lc, rng):
    sc = scramble(raw_lc, rng)
    assert np.array_equal(sc.time, raw_lc.time)
    for s in raw_lc.sectors:
        m = raw_lc.sector == s
        assert np.array_equal(np.sort(sc.flux[m]), np.sort(raw_lc.flux[m]))
        assert not np.array_equal(sc.flux[m], raw_lc.flux[m])


def _null(sde, snr):
    return {
        "tic": 1,
        "kind": "scrambled",
        "i": 0,
        "best": {"sde": sde, "snr": snr, "n_transits": 4},
    }


def _inj(sde, snr, status="recovered"):
    return {
        "tic": 1,
        "status": status,
        "period_days": 5.0,
        "radius_rearth": 2.0,
        "detection": {"sde": sde, "snr": snr, "n_transits": 4},
    }


def test_choose_thresholds_sits_above_nulls():
    nulls = [_null(s, 6.0) for s in np.linspace(6, 10, 400)]
    injections = [_inj(s, 12.0) for s in (8, 9, 11, 15, 30)] + [_inj(20, 20, "missed")]
    best = choose_thresholds(injections, nulls, 0.01, 2)
    # ≤ 1 % of nulls may pass: either SDE sits at the top of the nulls or SNR above 6.
    assert best["false_alarm"] <= 0.01
    assert best["recovery"] == pytest.approx(5 / 6)
    assert "warning" not in best
    few = choose_thresholds(injections, nulls[:20], 0.01, 2)
    assert "null samples" in few["warning"]


@pytest.mark.slow
def test_inject_and_calibrate_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[run]\nname = "cal"\n'
        "[search.periodic]\nmax_signals = 1\n"
        "[inject]\nn_injections = 3\nn_scrambles = 1\nradius_min_rearth = 3.0\n"
        "radius_max_rearth = 6.0\nperiod_max_days = 6.0\n"
    )
    assert (
        main(["synth", "-c", str(cfg), "--preset", "quiet", "--tic", "77", "--sectors", "14-15"])
        == 0
    )
    assert main(["inject", "-c", str(cfg), "77"]) == 0
    d = tmp_path / "results/cal/inject/TIC77"
    rows = [json.loads(x) for x in (d / "injections.jsonl").read_text().splitlines()]
    nulls = [json.loads(x) for x in (d / "nulls.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert sorted(n["kind"] for n in nulls) == ["inverted", "scrambled", "scrambled_inverted"]
    # Big, short-period planets on a quiet star: all found.
    assert all(r["status"] == "recovered" for r in rows)

    # Resume: asking for more only runs the new ones.
    capsys.readouterr()
    assert main(["inject", "-c", str(cfg), "77", "--n", "4"]) == 0
    err = capsys.readouterr().err
    assert "injection 1/1" in err and "null" not in err
    assert len((d / "injections.jsonl").read_text().splitlines()) == 4

    assert main(["calibrate", "-c", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "chosen:" in out and "warning" in out
    report = tmp_path / "results/cal/calibration/calibration.html"
    assert "Recovery map" in report.read_text()
    res = json.loads((tmp_path / "results/cal/calibration/calibration.json").read_text())
    assert res["chosen"]["n_injections"] == 4 and res["hosts"][0]["tic"] == 77


def test_adopting_thresholds_keeps_results():
    import dataclasses

    from scout.config import load_config
    from scout.inject import config_key

    cfg = load_config()
    new = dataclasses.replace(
        cfg,
        search=dataclasses.replace(
            cfg.search,
            periodic=dataclasses.replace(cfg.search.periodic, sde_threshold=17.0),
        ),
    )
    assert config_key(cfg, "u") == config_key(new, "u")
    wider = dataclasses.replace(cfg, detrend=dataclasses.replace(cfg.detrend, window_factor=4.0))
    assert config_key(cfg, "u") != config_key(wider, "u")
