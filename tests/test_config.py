from pathlib import Path

import pytest

from scout.config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]


def test_defaults_load():
    cfg = load_config()
    assert cfg.run.name == "default"
    assert cfg.search.periodic.min_transits == 2
    assert cfg.detrend.window_factor == 3.0


def test_shipped_cvz_config_loads():
    cfg = load_config(REPO / "config" / "cvz-north.toml")
    assert cfg.run.name == "cvz-north"
    assert cfg.fetch.sources == ("SPOC", "TESS-SPOC", "QLP")
    assert cfg.fetch.flux_columns["QLP"] == "SAP_FLUX"
    assert cfg.run_dir == Path("results/cvz-north")


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[detrend]\nwindow_factr = 3\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[detrnd]\nwindow_factor = 3\n"))


def test_wrong_type_rejected(tmp_path):
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(write(tmp_path, '[detrend]\nwindow_factor = "three"\n'))
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(write(tmp_path, "[search.periodic]\nmin_transits = 2.5\n"))


def test_int_accepted_for_float(tmp_path):
    cfg = load_config(write(tmp_path, "[detrend]\nwindow_factor = 4\n"))
    assert cfg.detrend.window_factor == 4.0
    assert isinstance(cfg.detrend.window_factor, float)


def test_validation(tmp_path):
    with pytest.raises(ConfigError, match="window_factor"):
        load_config(write(tmp_path, "[detrend]\nwindow_factor = 0.5\n"))
    with pytest.raises(ConfigError, match="min_transits"):
        load_config(write(tmp_path, "[search.periodic]\nmin_transits = 1\n"))
    with pytest.raises(ConfigError, match="flux_columns"):
        load_config(write(tmp_path, '[fetch]\nsources = ["SPOC", "TGLC"]\n'))
    with pytest.raises(ConfigError, match=r"fetch\.sectors"):
        load_config(write(tmp_path, '[fetch]\nsectors = [14, "x"]\n'))
    with pytest.raises(ConfigError, match="radius"):
        load_config(write(tmp_path, "[inject]\nradius_min_rearth = 5.0\nradius_max_rearth = 2.0\n"))
    with pytest.raises(ConfigError, match="false_alarm_target"):
        load_config(write(tmp_path, "[inject]\nfalse_alarm_target = 0\n"))


def test_shipped_calibration_config_matches_science_search():
    sci = load_config(REPO / "config" / "cvz-north.toml")
    cal = load_config(REPO / "config" / "m2-calibration.toml")
    # Thresholds only transfer if everything upstream of them is the same.
    assert (cal.stitch, cal.detrend, cal.search) == (sci.stitch, sci.detrend, sci.search)
    assert cal.fetch.sectors == tuple(range(14, 27))


def test_bad_toml_and_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "[detrend\n"))
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_overrides_and_digest(tmp_path):
    base = load_config()
    over = load_config(overrides={"run": {"name": "x"}, "detrend": {"window_factor": 5}})
    assert over.run.name == "x"
    assert over.detrend.window_factor == 5.0
    assert over.digest() != base.digest()
    assert load_config().digest() == base.digest()
