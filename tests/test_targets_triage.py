import json

import numpy as np
import pytest

from scout.config import TargetsConfig, TriageConfig
from scout.targets import apply_cuts, ecliptic_latitude, read_targets, write_targets
from scout.triage import (
    KnownIndex,
    KnownObject,
    classify,
    parse_ctoi,
    parse_planets,
    parse_toi,
    period_match,
)


def star(tic, tmag, teff=5500.0, lat=85.0, contam=0.01):
    return {
        "tic": tic,
        "tmag": tmag,
        "teff_k": teff,
        "radius_rsun": 1.0,
        "mass_msun": 1.0,
        "ra_deg": 270.0,
        "dec_deg": 66.0,
        "ecliptic_lat_deg": lat,
        "contamination_ratio": contam,
    }


def test_ecliptic_pole():
    assert ecliptic_latitude(np.array([270.0]), np.array([66.560708]))[0] == pytest.approx(
        90.0, abs=0.01
    )


def test_cuts_sort_and_limit(tmp_path):
    rows = [
        star(1, 9.5),
        star(2, 8.0),
        star(3, 14.0),  # too faint
        star(4, 9.0, teff=8000),  # too hot
        star(5, 9.0, lat=70.0),  # outside the CVZ
        star(6, 9.0, contam=0.5),  # blended
        star(7, 9.2, contam=float("nan")),  # unknown contamination is kept
    ]
    cfg = TargetsConfig(tmag_max=13.0, contamination_max=0.1)
    assert [r["tic"] for r in apply_cuts(rows, cfg)] == [2, 7, 1]
    assert [r["tic"] for r in apply_cuts(rows, TargetsConfig(max_targets=2))] == [2, 7]
    p = write_targets(apply_cuts(rows, cfg), tmp_path / "t.csv")
    back = read_targets(p)
    assert [r["tic"] for r in back] == [2, 7, 1]
    assert back[0]["tmag"] == 8.0


TOI_CSV = (
    "TIC ID,TOI,TFOPWG Disposition,RA,Dec,Period (days)\n"
    "100,1.01,CP,18:00:00.0,+66:30:00.0,3.5\n"
    "100,1.02,PC,18:00:00.0,+66:30:00.0,10.0\n"
    "200,2.01,FP,18:00:05.0,+66:30:00.0,1.25\n"
)
CTOI_CSV = "TIC ID,CTOI,RA,Dec,Period (days)\n300,300.01,270.5,66.0,7.0\n"
PS_CSV = 'pl_name,tic_id,pl_orbper,ra,dec\n"Foo b","TIC 400",4.0,271.0,67.0\n'


def test_parse_catalogues():
    toi = parse_toi(TOI_CSV)
    assert [o.name for o in toi] == ["TOI-1.01", "TOI-1.02", "TOI-2.01"]
    assert toi[0].ra_deg == pytest.approx(270.0) and toi[0].dec_deg == pytest.approx(66.5)
    assert toi[2].disposition == "FP"
    assert parse_ctoi(CTOI_CSV)[0].period_days == 7.0
    p = parse_planets(PS_CSV)[0]
    assert (p.tic, p.name, p.kind) == (400, "Foo b", "planet")


def test_period_match_aliases():
    assert period_match(3.5, 3.5, 0.01) == 1.0
    assert period_match(7.0, 3.5, 0.01) == 2.0
    assert period_match(1.1667, 3.5, 0.01) == pytest.approx(1 / 3)
    assert period_match(4.2, 3.5, 0.01) is None
    assert period_match(3.5, float("nan"), 0.01) is None


def test_classify():
    objs = parse_toi(TOI_CSV) + parse_ctoi(CTOI_CSV) + parse_planets(PS_CSV)
    idx = KnownIndex(objs)
    tc = TriageConfig()
    ra, dec = 270.0, 66.5
    assert classify(100, 3.501, ra, dec, idx, tc)["triage"] == "known"
    assert classify(100, 20.0, ra, dec, idx, tc)["match"] == "TOI-1.02"  # 2× alias
    r = classify(100, 6.3, ra, dec, idx, tc)
    assert r["triage"] == "new_on_known_host" and "TOI-1.01" in r["match"]
    # TIC 200 is a known FP 30″ away: our signal at its period is that blend.
    r = classify(999, 1.25, ra, dec, idx, tc)
    assert r["triage"] == "neighbour" and r["match_sep_arcsec"] == pytest.approx(30, abs=1)
    assert classify(200, 1.25, 270.02, 66.5, idx, tc)["triage"] == "known_false_positive"
    assert classify(999, 6.3, ra, dec, idx, tc)["triage"] == "new"
    # Far away, same period: not a neighbour.
    assert classify(999, 7.0, 200.0, 10.0, idx, tc)["triage"] == "new"


def test_neighbour_ignores_unknown_positions():
    idx = KnownIndex([KnownObject(1, "x", "TOI", 2.0, float("nan"), float("nan"))])
    assert idx.near(270.0, 66.0, 120.0, exclude_tic=5) == []


@pytest.mark.slow
def test_batch_and_triage_on_synthetic_stars(tmp_path, monkeypatch, capsys):
    from scout import triage
    from scout.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[run]\nname = "b"\n[search.periodic]\nmax_signals = 1\nrun_inverted = false\n')
    for tic, preset in ((11, "hot-jupiter"), (12, "quiet")):
        assert (
            main(
                [
                    "synth",
                    "-c",
                    str(cfg),
                    "--preset",
                    preset,
                    "--tic",
                    str(tic),
                    "--sectors",
                    "14-15",
                ]
            )
            == 0
        )
    (tmp_path / "results/b/targets.csv").write_text("tic\n11\n12\n")
    assert main(["batch", "-c", str(cfg)]) == 0
    log = [json.loads(x) for x in (tmp_path / "results/b/batch.jsonl").read_text().splitlines()]
    assert {r["tic"]: r["status"] for r in log} == {11: "ok", 12: "ok"}
    assert {r["tic"]: r["n_passing"] for r in log}[11] == 1

    # Pretend TIC 11's planet is a known TOI.
    known = [KnownObject(11, "TOI-9.01", "TOI", 3.21, float("nan"), float("nan"), "PC")]
    monkeypatch.setattr(triage, "load_known", lambda c: (known, {"test": {"n": 1}}))
    capsys.readouterr()
    assert main(["triage", "-c", str(cfg)]) == 0
    assert "known 1" in capsys.readouterr().out
    html = (tmp_path / "results/b/summary.html").read_text()
    assert "TOI-9.01" in html and "reports/TIC11.html" in html

    import pyarrow.parquet as pq

    tab = pq.read_table(tmp_path / "results/b/candidates.parquet").to_pylist()
    assert {r["tic"]: r["triage"] for r in tab}[11] == "known"
