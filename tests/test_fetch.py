import numpy as np
import pytest
from astropy.io import fits

from scout.config import FetchConfig
from scout.fetch import Candidate, _sector_from_filename, read_lightcurve_fits, select_products


def cand(sector, source, cadence, name=None):
    return Candidate(
        sector=sector,
        source=source,
        cadence_s=cadence,
        obs_id="x",
        filename=name or f"{source}-s{sector}-{cadence}.fits",
        url="mast:x",
    )


def test_select_prefers_source_then_cadence():
    cfg = FetchConfig()
    chosen = select_products(
        [
            cand(14, "QLP", 1800),
            cand(14, "SPOC", 120),
            cand(15, "QLP", 600),
            cand(15, "TESS-SPOC", 600),
            cand(16, "QLP", 200),
            cand(16, "QLP", 1800),
            cand(17, "TGLC", 1800),  # not configured
            cand(18, "SPOC", 20),  # 20 s not in allowed cadences
        ],
        cfg,
    )
    assert [(p.sector, p.source, p.cadence_s) for p in chosen] == [
        (14, "SPOC", 120),
        (15, "TESS-SPOC", 600),
        (16, "QLP", 200),
    ]


def test_sector_from_filename():
    assert _sector_from_filename("tess2019198215352-s0014-0000000123456789-0150-s_lc.fits") == 14
    assert _sector_from_filename("hlsp_qlp_tess_ffi_s0040-0000000123456789_tess_v01_llc.fits") == 40
    assert _sector_from_filename("nothing.fits") is None


def _write_fits(path, cols, header0, timedel):
    h0 = fits.PrimaryHDU()
    for k, v in header0.items():
        h0.header[k] = v
    t = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=k, format="D" if v.dtype.kind == "f" else "J", array=v)
            for k, v in cols.items()
        ]
    )
    t.header["TIMEDEL"] = timedel
    fits.HDUList([h0, t]).writeto(path)


def test_read_spoc_like(tmp_path):
    n = 50
    t = 1700 + np.arange(n) * 120 / 86400
    q = np.zeros(n, dtype=np.int32)
    q[3] = 32
    _write_fits(
        tmp_path / "spoc.fits",
        {
            "TIME": t,
            "PDCSAP_FLUX": np.full(n, 5e3),
            "PDCSAP_FLUX_ERR": np.full(n, 5.0),
            "SAP_FLUX": np.full(n, 6e3),
            "QUALITY": q,
        },
        {"SECTOR": 21, "TICID": 123456789, "PROCVER": "spoc-5.0"},
        120 / 86400,
    )
    lc = read_lightcurve_fits(tmp_path / "spoc.fits", "SPOC", FetchConfig())
    assert lc.sectors == [21]
    assert lc.tic == 123456789
    assert set(lc.cadence_s) == {120}
    assert lc.flux[0] == 5e3 and lc.flux_err[0] == 5.0
    assert lc.quality[3] == 32
    assert lc.meta["flux_column"] == "PDCSAP_FLUX"


def test_read_qlp_like(tmp_path, rng):
    n = 200
    t = 1700 + np.arange(n) * 600 / 86400
    q = np.zeros(n, dtype=np.int32)
    q[5] = 4096
    _write_fits(
        tmp_path / "qlp.fits",
        {
            "TIME": t,
            "SAP_FLUX": 1 + rng.normal(0, 1e-3, n),
            "KSPSAP_FLUX": np.ones(n),
            "QUALITY": q,
        },
        {"SECTOR": 41, "TICID": 42},
        600 / 86400,
    )
    lc = read_lightcurve_fits(tmp_path / "qlp.fits", "QLP", FetchConfig())
    assert set(lc.cadence_s) == {600}
    # Non-zero QLP flags become SPOC manual-exclude.
    assert lc.quality[5] == 128 and lc.quality[6] == 0
    # No error column: estimated from point-to-point scatter.
    assert lc.flux_err[0] == pytest.approx(1e-3, rel=0.25)


def test_with_retries_recovers_from_dropped_connection():
    from scout.fetch import _with_retries

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("proxy closed the connection")
        return "ok"

    assert _with_retries("test", flaky, delays=(0.0, 0.0, 0.0)) == "ok"
    assert len(calls) == 3


def test_with_retries_gives_up_and_reraises():
    from scout.fetch import _with_retries

    def broken():
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError, match="still down"):
        _with_retries("test", broken, delays=(0.0,))


def test_with_retries_does_not_retry_other_errors():
    from scout.fetch import _with_retries

    calls = []

    def bad():
        calls.append(1)
        raise LookupError("TIC not found")

    with pytest.raises(LookupError):
        _with_retries("test", bad, delays=(0.0, 0.0))
    assert len(calls) == 1
