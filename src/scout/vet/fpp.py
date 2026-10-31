"""False-positive probability with TRICERATOPS (design §6 item 10).

Runs after ``scout vet`` on the signals that survived it (``vet.fpp_verdicts``)
and fills in checklist item 10 of their vetting results. TRICERATOPS
(Giacalone & Dressing 2020) needs lightkurve, which does not install on
Python 3.13, so this step runs in its own environment, e.g.::

    uv venv -p 3.12 .venv-fpp && . .venv-fpp/bin/activate
    pip install -e ".[mast,models,fpp]"
    scout fpp -c config/cvz-north.toml

TRICERATOPS draws the field-star population from Gaia DR3 at ESA. Where that
archive is unreachable, its query is sent to the Gaia DR3 copy at MAST
instead (:func:`_patch_gaia`), with the same columns.

Apertures are the SPOC optimal apertures from the light-curve files we
already use; sectors without one (QLP) are left out of the depth model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config
from scout.lightcurve import LightCurve
from scout.vet import checks as ck

log = logging.getLogger(__name__)


def _patch_gaia(url: str) -> None:
    """Point astroquery's Gaia (used inside TRICERATOPS) at the MAST Gaia DR3 TAP."""
    import warnings

    import astroquery.gaia
    import pyvo

    class _Job:
        def __init__(self, adql: str) -> None:
            self.adql = adql.replace("gaiadr3.gaia_source", "dbo.gaia_source")

        def get_results(self):
            svc = pyvo.dal.TAPService(url)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pyvo.dal.exceptions.DALOverflowWarning)
                return svc.search(self.adql, maxrec=1_000_000).to_table()

    class _Gaia:
        ROW_LIMIT = -1

        def launch_job(self, adql: str, **_: Any) -> _Job:
            return _Job(adql)

        launch_job_async = launch_job

    astroquery.gaia.Gaia = _Gaia()


def spoc_aperture(path: str | Path) -> np.ndarray | None:
    """[[col, row], ...] of the SPOC optimal aperture stored in a light-curve file."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        if "APERTURE" not in hdul:
            return None
        ap = hdul["APERTURE"].data.astype(int)
        h = hdul["APERTURE"].header
        if "CRVAL1P" not in h:
            return None
        rows, cols = np.nonzero(ap & 2)
        if len(rows) == 0:
            return None
        return np.column_stack([cols + h["CRVAL1P"], rows + h["CRVAL2P"]])


def folded_for_fpp(f: ck.Folded, bin_minutes: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Time from mid-transit (days), binned flux and its per-bin error, ±3 durations."""
    w = 3 * f.eph.dur_days
    m = np.abs(f.phase) < w
    x, y = f.phase[m], f.flux[m]
    width = bin_minutes / 1440
    idx = np.floor((x + w) / width).astype(int)
    n = np.bincount(idx)
    keep = n > 0
    xb = np.bincount(idx, x)[keep] / n[keep]
    yb = np.bincount(idx, y)[keep] / n[keep]
    out = np.abs(xb) > f.eph.dur_days
    err = ck.robust_std(yb[out]) if out.sum() > 5 else f.sigma / np.sqrt(np.median(n[keep]))
    return xb, yb, float(err)


def check_fpp(fpp: float, nfpp: float, cfg_vet) -> ck.Check:
    v = {"fpp": fpp, "nfpp": nfpp}
    text = f"FPP = {fpp:.3g}, NFPP = {nfpp:.3g}"
    if nfpp >= cfg_vet.nfpp_fail or fpp >= cfg_vet.fpp_fail:
        return ck.Check(10, "TRICERATOPS FPP", "fail", text, v)
    if fpp < cfg_vet.fpp_validated and nfpp < cfg_vet.nfpp_max:
        return ck.Check(10, "TRICERATOPS FPP", "pass", text + " (validation level)", v)
    if fpp < cfg_vet.fpp_likely and nfpp < cfg_vet.nfpp_max:
        return ck.Check(10, "TRICERATOPS FPP", "pass", text + " (likely planet)", v)
    return ck.Check(10, "TRICERATOPS FPP", "warn", text, v)


def run_fpp_signal(cfg: Config, res: dict[str, Any]) -> ck.Check:
    import pyarrow.parquet as pq
    import triceratops.triceratops as tr
    from astropy.io import fits

    from scout.pipeline import star_dir
    from scout.vet.run import _fits_files, _mask_others

    tic, index = int(res["tic"]), int(res["index"])
    sd = star_dir(cfg, tic)
    all_rows = pq.read_table(cfg.run_dir / "candidates.parquet").to_pylist()
    row = next(r for r in all_rows if int(r["tic"]) == tic and int(r["index"]) == index)
    flat = _mask_others(LightCurve.read(sd / "flat.parquet"), row, all_rows)
    e = res["ephemeris"]
    eph = ck.Ephemeris(e["period_days"], e["t0_btjd"], e["duration_hours"], e["depth_ppm"])
    f = ck.Folded.build(flat.time, flat.flux, flat.sector, eph, cfg.vet)
    files, _sources, _ = _fits_files(cfg, tic)
    # Sectors with a SPOC aperture, those with the most in-transit data first.
    n_in = {int(s): int((f.in_transit & (f.sector == s)).sum()) for s in np.unique(f.sector)}
    aps = []
    for path, _src in files:
        sector = int(fits.getheader(path, 0).get("SECTOR", -1))
        ap = spoc_aperture(path)
        if ap is not None and n_in.get(sector, 0) > 0:
            aps.append((n_in[sector], sector, ap))
    if not aps:
        return ck.Check(10, "TRICERATOPS FPP", "n/a", "no SPOC aperture in a sector with a transit")
    aps.sort(key=lambda a: -a[0])
    aps = aps[: cfg.vet.fpp_max_sectors]
    sectors = np.array([a[1] for a in aps])
    target = tr.target(ID=tic, sectors=sectors)
    target.calc_depths(tdepth=eph.depth_ppm * 1e-6, all_ap_pixels=[a[2] for a in aps])
    x, y, err = folded_for_fpp(f, cfg.vet.fpp_bin_minutes)
    target.calc_probs(
        time=x, flux_0=y, flux_err_0=err, P_orb=eph.period_days, N=cfg.vet.fpp_n, verbose=0
    )
    c = check_fpp(float(target.FPP), float(target.NFPP), cfg.vet)
    c.values["sectors"] = [int(s) for s in sectors]
    probs = target.probs
    c.values["scenarios"] = {
        str(s): float(p) for s, p in probs.groupby("scenario")["prob"].sum().items()
    }
    return c


def fpp_run(cfg: Config, tics: list[int] | None = None, progress=None) -> dict[str, Any]:
    from scout.report.vet import update_vet_report
    from scout.vet.run import _verdict_from_dicts, refresh_summary, vet_dir

    _patch_gaia(cfg.vet.gaia_tap_url)
    done, errors = [], []
    paths = sorted(vet_dir(cfg).glob("TIC*.json"))
    todo = []
    for p in paths:
        res = json.loads(p.read_text())
        if tics and int(res["tic"]) not in tics:
            continue
        if not tics and res["verdict"] not in cfg.vet.fpp_verdicts:
            continue
        todo.append((p, res))
    for i, (p, res) in enumerate(todo, 1):
        try:
            c = run_fpp_signal(cfg, res)
        except Exception as exc:
            log.exception("TIC %s: TRICERATOPS failed", res["tic"])
            errors.append({"tic": res["tic"], "index": res["index"], "error": str(exc)})
            if progress:
                progress(res, None, i, len(todo))
            continue
        res["checks"] = [x for x in res["checks"] if x["item"] != 10] + [c.to_dict()]
        res["checks"].sort(key=lambda x: x["item"])
        res["verdict"] = _verdict_from_dicts(res["checks"])
        res["n_fail"] = sum(x["status"] == "fail" for x in res["checks"])
        res["n_warn"] = sum(x["status"] == "warn" for x in res["checks"])
        p.write_text(json.dumps(res, indent=2, default=str))
        html_path = cfg.run_dir / res["report"]
        if html_path.exists():
            update_vet_report(html_path, res)
        done.append(res)
        if progress:
            progress(res, c, i, len(todo))
    summary = refresh_summary(cfg)
    return {"done": done, "errors": errors, "summary": str(summary)}
