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
    assert main(["vet", "123"]) == 2
    assert "M5" in capsys.readouterr().err


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
    assert "FAILED" in capsys.readouterr().err
    run_json = json.loads((tmp_path / "results/t/run.json").read_text())
    assert run_json["invocations"][-1]["status"]["77"].startswith("failed")
