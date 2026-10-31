import numpy as np
import pytest

from scout.config import VetConfig
from scout.synthetic import Transit, transit_model
from scout.vet import checks as ck
from scout.vet.centroid import event_shifts
from scout.vet.gaia import check_gaia

CFG = VetConfig()
P, T0, DUR, DEPTH = 3.7, 1700.3, 3.0, 2e-3


def lc(noise=4e-4, days=80.0, seed=1, cadence_min=10.0):
    rng = np.random.default_rng(seed)
    t = np.arange(1700.0, 1700.0 + days, cadence_min / 1440)
    t = t[np.abs((t - 1700.0) % 13.7 - 13.2) > 0.5]  # orbit gaps
    return t, 1.0 + rng.normal(0, noise, len(t)), np.where(t < 1700 + days / 2, 14, 15)


def planet(t, depth=DEPTH, period=P, t0=T0):
    return transit_model(
        t, Transit(t0_btjd=t0, depth=depth, duration_hours=DUR, period_days=period, impact=0.3)
    )


def fold(t, f, s, period=P, depth=DEPTH):
    return ck.Folded.build(t, f, s, ck.Ephemeris(period, T0, DUR, depth * 1e6), CFG)


def test_planet_passes_light_curve_checks():
    pytest.importorskip("batman")
    t, f, s = lc()
    f = f * planet(t)
    fo = fold(t, f, s)
    ev = fo.event_depths()
    fit = ck.fit_transit(fo, 1.0, 1.0)
    assert fit["ok"] and not fit["grazing"]
    assert fit["rp_rearth"] == pytest.approx(np.sqrt(DEPTH) * 109.08, rel=0.25)
    statuses = {
        "shape": ck.check_shape(fit).status,
        "var": ck.check_variability(fo, CFG).status,
        "oe": ck.check_odd_even(fo, ev, CFG).status,
        "sec": ck.check_secondary(fo, CFG).status,
        "depth": ck.check_depth_consistency(fo, ev, {14: "SPOC", 15: "SPOC"}, CFG).status,
        "radius": ck.check_radius(fit, CFG).status,
        "density": ck.check_density(fit, CFG).status,
        "dominance": ck.check_single_event_dominance(ev, CFG).status,
    }
    assert set(statuses.values()) == {"pass"}, statuses


def test_eb_at_twice_the_period_fails_odd_even():
    t, f, s = lc()
    # Primary and secondary of an EB with P = 2 × 3.7 d, found by the search at 3.7 d.
    f = f * planet(t, 3e-3, 2 * P, T0) * planet(t, 1e-3, 2 * P, T0 + P)
    fo = fold(t, f, s, depth=2e-3)
    assert ck.check_odd_even(fo, fo.event_depths(), CFG).status == "fail"


def test_secondary_eclipse_fails():
    t, f, s = lc()
    f = f * planet(t) * planet(t, 1.5e-3, P, T0 + 0.5 * P)
    c = ck.check_secondary(fold(t, f, s), CFG)
    assert c.status == "fail"
    assert c.values["phase"] == pytest.approx(0.5, abs=0.02)


def test_sinusoid_fails_variability():
    t, f, s = lc()
    f = f + 1e-3 * np.cos(2 * np.pi * (t - T0) / P + np.pi)  # minimum at T0
    c = ck.check_variability(fold(t, f, s, depth=1e-3), CFG)
    assert c.status == "fail"


def test_transit_too_long_for_star_fails_density():
    pytest.importorskip("batman")
    t, f, s = lc()
    f = f * transit_model(
        t, Transit(t0_btjd=T0, depth=DEPTH, duration_hours=12.0, period_days=P, impact=0.0)
    )
    fo = ck.Folded.build(t, f, s, ck.Ephemeris(P, T0, 12.0, DEPTH * 1e6), CFG)
    fit = ck.fit_transit(fo, 0.8, 0.8)
    assert ck.check_density(fit, CFG).status == "fail"


def test_shared_dates_and_partial_events_fail_sparse_signal():
    t, f, s = lc()
    period = 30.0
    f = f * planet(t, period=period)
    fo = fold(t, f, s, period=period)
    times = ck.event_times(fo)
    others = [(tic, tt + 0.05, 0.1) for tt in times for tic in (1, 2)]
    c = ck.check_spacecraft(fo, others, 99, ck.data_gaps(t, CFG.gap_min_days), CFG)
    assert c.status == "fail"
    c = ck.check_spacecraft(fo, [], 99, ck.data_gaps(t, CFG.gap_min_days), CFG)
    assert c.status == "pass"
    # Only one side of each dip observed: a ramp at a data edge.
    keep = ck.phase_days(t, period, T0) < 0.01
    fo = ck.Folded.build(t[keep], f[keep], s[keep], fo.eph, CFG)
    c = ck.check_spacecraft(fo, [], 99, [], CFG)
    assert c.status == "fail" and c.values["partial_events"]


def test_red_noise_inflates_event_errors():
    t, f, s = lc()
    rng = np.random.default_rng(3)
    # Correlated noise on transit time scales.
    walk = np.convolve(rng.normal(0, 4e-4, len(t)), np.ones(18) / 18, mode="same") * 4
    fo_white = fold(t, f, s)
    fo_red = fold(t, f + walk, s)
    assert fo_red.sigma_box > 2 * fo_white.sigma_box


def test_centroid_shift_per_event():
    t = np.arange(1700.0, 1720.0, 2 / 1440)
    eph = ck.Ephemeris(P, T0, DUR, 2000.0)
    intr = np.abs(ck.phase_days(t, P, T0)) < 0.4 * DUR / 24
    rng = np.random.default_rng(0)
    c = {
        "time": t,
        "col": 100 + rng.normal(0, 1e-3, len(t)) + 0.01 * (t - 1700) + np.where(intr, 2e-3, 0),
        "row": 50 + rng.normal(0, 1e-3, len(t)),
        "quality": np.zeros(len(t), int),
        "sector": 14,
    }
    sh = np.array(event_shifts(c, eph, 3.0, 175))
    assert len(sh) >= 4
    # The linear drift is removed; the in-transit step remains.
    assert np.mean(sh[:, 1]) == pytest.approx(2e-3, abs=3e-4)
    assert abs(np.mean(sh[:, 2])) < 3e-4


def test_gaia_neighbour_that_could_make_the_dip():
    star = {"ra_deg": 270.0, "dec_deg": 66.0, "gaia_dr2": "1", "tmag": 9.0}
    rows = [
        {"source_id": "1", "ra": 270.0, "dec": 66.0, "g": 9.5, "ruwe": 1.0},
        {"source_id": "2", "ra": 270.0, "dec": 66.0 + 30 / 3600, "g": 14.5, "ruwe": 1.0},
        {"source_id": "3", "ra": 270.0, "dec": 66.0 - 40 / 3600, "g": 19.0, "ruwe": 1.0},
    ]
    # 14.5 vs 9.5 mag: 1 % of the light, so it can fake a 2000 ppm dip but not 20000.
    c = check_gaia([dict(r) for r in rows], star, 2000.0, "fail", CFG)
    assert c.status == "warn" and c.values["n_capable"] == 1
    c = check_gaia([dict(r) for r in rows], star, 20000.0, "pass", CFG)
    assert c.status == "pass" and c.values["n_capable"] == 0
    rows[0]["ruwe"] = 2.5
    c = check_gaia([dict(r) for r in rows], star, 20000.0, "pass", CFG)
    assert c.status == "warn"
