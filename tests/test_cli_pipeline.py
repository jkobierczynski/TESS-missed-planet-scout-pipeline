import json
import re

import pytest

from scout.cli import main


def test_config_commands(capsys):
    assert main(["config", "check"]) == 0
    assert "ok: run 'default'" in capsys.readouterr().out
    assert main(["config", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["name"] == "default"


def test_bad_config_exit_code(tmp_path, capsys):
    p = tmp_path / "bad.toml"
    p.write_text("[nope]\n")
    assert main(["config", "check", "-c", str(p)]) == 2
    assert "config error" in capsys.readouterr().err


def test_planned_commands_exit_2(capsys):
    assert main(["ctoi", "123"]) == 2
    assert "M6" in capsys.readouterr().err


def test_vet_needs_triage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["vet"]) == 2
    assert "scout triage" in capsys.readouterr().err


def test_bad_tic():
    with pytest.raises(SystemExit):
        main(["run", "not-a-tic"])


@pytest.mark.slow
def test_synth_then_run_is_resumable(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "synth",
                "--run",
                "t",
                "--preset",
                "hot-jupiter",
                "--sectors",
                "14-15",
                "--tic",
                "TIC 1234",
            ]
        )
        == 0
    )
    assert main(["run", "--run", "t", "1234"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"\* #1 P=3\.2[01]\d+ d", out)
    star = tmp_path / "results/t/stars/TIC1234"
    report = tmp_path / "results/t/reports/TIC1234.html"
    assert report.exists() and "Vetting checklist" in report.read_text()
    run_json = json.loads((tmp_path / "results/t/run.json").read_text())
    assert run_json["invocations"][-1]["status"] == {"1234": "ok"}

    mtimes = {p.name: p.stat().st_mtime_ns for p in star.iterdir()}
    assert main(["run", "--run", "t", "1234"]) == 0
    assert {p.name: p.stat().st_mtime_ns for p in star.iterdir()} == mtimes

    # Changing a detrend setting reruns detrend and later stages, not stitch.
    cfg = tmp_path / "c.toml"
    cfg.write_text('[run]\nname = "t"\n[detrend]\nwindow_factor = 3.5\n')
    assert main(["run", "-c", str(cfg), "1234"]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in star.iterdir()}
    assert after["stitched.parquet"] == mtimes["stitched.parquet"]
    assert after["flat.parquet"] != mtimes["flat.parquet"]
    assert after["search.json"] != mtimes["search.json"]


def test_failed_star_recorded(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # No raw data and fetch needs MAST: make fetch fail fast and offline.
    import scout.fetch as f

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(f, "fetch", boom)
    assert main(["run", "--run", "t", "77"]) == 1
    # The message names the stage that failed, not the last requested one.
    assert "FAILED in fetch: offline" in capsys.readouterr().err
    run_json = json.loads((tmp_path / "results/t/run.json").read_text())
    assert run_json["invocations"][-1]["status"]["77"].startswith("failed")


def test_prune_after_stitch_and_refetch(tmp_path, monkeypatch):
    import dataclasses

    from scout import pipeline
    from scout.config import load_config
    from scout.synthetic import generate, preset

    cfg = load_config(overrides={"run": {"name": "p", "results_dir": str(tmp_path)}})
    cfg = dataclasses.replace(cfg, fetch=dataclasses.replace(cfg.fetch, prune_after_stitch=True))
    fits = tmp_path / "x.fits"
    lc = generate(preset("quiet", sectors=(14,)))
    calls = []

    def fake_fetch(tic, cache_dir, fcfg):
        calls.append(tic)
        fits.write_bytes(b"x")
        return {"tic": tic, "star": {}, "products": [{"local_path": str(fits), "source": "SPOC"}]}

    monkeypatch.setattr(pipeline.fetch_mod, "fetch", fake_fetch)
    monkeypatch.setattr(pipeline.fetch_mod, "load_from_manifest", lambda m, c: [lc])
    key = pipeline.stage_fetch(cfg, 5)
    pipeline.stage_stitch(cfg, 5, key)
    assert not fits.exists() and calls == [5]
    # Stitch again (forced): the FITS are gone, so they are fetched again first.
    pipeline.stage_stitch(cfg, 5, key, force=True)
    assert calls == [5, 5]
    # Toggling pruning is not an input: the fetch key stays the same.
    off = dataclasses.replace(cfg, fetch=dataclasses.replace(cfg.fetch, prune_after_stitch=False))
    assert pipeline.stage_fetch(off, 5) == key


def test_thread_count_is_not_part_of_the_search_key(tmp_path, monkeypatch):
    import dataclasses

    from scout import pipeline
    from scout.config import load_config

    cfg = load_config(overrides={"run": {"name": "k", "results_dir": str(tmp_path)}})
    keys = []
    monkeypatch.setattr(pipeline, "_run_stage", lambda name, out, key, force, produce: (key, 0))
    for threads in (0, 2):
        c = dataclasses.replace(
            cfg,
            search=dataclasses.replace(
                cfg.search, periodic=dataclasses.replace(cfg.search.periodic, threads=threads)
            ),
        )
        keys.append(pipeline.stage_search(c, 1, "up"))
    assert keys[0] == keys[1]
