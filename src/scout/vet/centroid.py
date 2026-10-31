"""Centroid shift during transit (design §6 item 5), from SPOC light-curve files.

SPOC and TESS-SPOC light curves carry the flux-weighted centroid of the
aperture (MOM_CENTR1/2, in pixels). If the dip comes from a neighbour, the
centroid moves towards the target during transit. Each transit is compared
with its own local baseline, so slow pointing drifts cancel; the pointing
model (POS_CORR1/2) is removed first.

The shift ``Δc`` and the depth ``d`` give the offset of the dimming source
from the out-of-transit centroid: ``|Δc| (1 − d) / d`` pixels. This is the
light-curve version of the check; TPF difference images are the follow-up
when it is ambiguous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scout.config import VetConfig
from scout.vet.checks import Check, Ephemeris, phase_days, robust_std


def read_centroids(path: str | Path) -> dict[str, np.ndarray] | None:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        d = hdul[1].data
        cols = {c.upper() for c in d.columns.names}
        if not {"MOM_CENTR1", "MOM_CENTR2"} <= cols:
            return None
        sector = int(hdul[0].header.get("SECTOR", -1))
        out = {
            "time": np.asarray(d["TIME"], float),
            "col": np.asarray(d["MOM_CENTR1"], float),
            "row": np.asarray(d["MOM_CENTR2"], float),
            "quality": np.asarray(d["QUALITY"], int) if "QUALITY" in cols else None,
            "sector": sector,
        }
        if {"POS_CORR1", "POS_CORR2"} <= cols:
            pc1, pc2 = np.asarray(d["POS_CORR1"], float), np.asarray(d["POS_CORR2"], float)
            if np.isfinite(pc1).any() and np.isfinite(pc2).any():
                out["col"] = out["col"] - np.nan_to_num(pc1)
                out["row"] = out["row"] - np.nan_to_num(pc2)
    return out


def event_shifts(
    c: dict[str, Any], eph: Ephemeris, baseline_durations: float, bitmask: int
) -> list[tuple[int, float, float, float, float]]:
    """(epoch, Δcol, Δrow, σcol, σrow) per transit in one sector."""
    t, col, row = c["time"], c["col"], c["row"]
    good = np.isfinite(t) & np.isfinite(col) & np.isfinite(row)
    if c.get("quality") is not None:
        good &= (c["quality"] & bitmask) == 0
    t, col, row = t[good], col[good], row[good]
    if len(t) < 20:
        return []
    ph = phase_days(t, eph.period_days, eph.t0_btjd)
    ep = np.round((t - eph.t0_btjd) / eph.period_days).astype(int)
    dur = eph.dur_days
    in_tr = np.abs(ph) < 0.4 * dur
    outer = min(baseline_durations * dur, 0.25 * eph.period_days)
    near = (np.abs(ph) > dur) & (np.abs(ph) < max(outer, 1.5 * dur))
    out = []
    for e in np.unique(ep[in_tr]):
        mi, mo = in_tr & (ep == e), near & (ep == e)
        if mi.sum() < 3 or mo.sum() < 6:
            continue
        # Local linear baseline through the out-of-transit centroids.
        res = []
        for y in (col, row):
            a = np.polyfit(t[mo] - t[mo].mean(), y[mo], 1)
            base = np.polyval(a, t - t[mo].mean())
            r_out = y[mo] - base[mo]
            s = robust_std(r_out)
            res.append((float(np.mean(y[mi] - base[mi])), s / np.sqrt(mi.sum())))
        out.append((int(e), res[0][0], res[1][0], res[0][1], res[1][1]))
    return out


def _combine(vals: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    w = 1 / errs**2
    mean = float(np.sum(w * vals) / np.sum(w))
    err = float(np.sqrt(1 / np.sum(w)))
    if len(vals) >= 4:
        # Red-noise floor from the scatter of the per-transit shifts.
        err = max(err, robust_std(vals) / np.sqrt(len(vals)))
    return mean, err


def check_centroid(
    files: list[tuple[str, str]], eph: Ephemeris, tmag: float | None, cfg: VetConfig, bitmask: int
) -> Check:
    """``files``: (path, source) of the light-curve FITS files used for the search."""
    shifts = []
    per_sector: dict[int, list] = {}
    for path, _source in files:
        try:
            c = read_centroids(path)
        except (OSError, KeyError, ValueError):
            c = None
        if c is None:
            continue
        ev = event_shifts(c, eph, cfg.baseline_durations, bitmask)
        shifts += ev
        if ev:
            a = np.array(ev)
            per_sector[c["sector"]] = [
                *_combine(a[:, 1], a[:, 3]),
                *_combine(a[:, 2], a[:, 4]),
            ]
    if not shifts:
        return Check(5, "No centroid shift", "n/a", "no SPOC centroids in transit")
    a = np.array(shifts)
    dc, ec = _combine(a[:, 1], a[:, 3])
    dr, er = _combine(a[:, 2], a[:, 4])
    shift = float(np.hypot(dc, dr))
    sig = float(np.hypot(dc / ec, dr / er))
    d = eph.depth_ppm * 1e-6
    offset = shift * (1 - d) / d * cfg.pixel_arcsec if d > 0 else float("nan")
    offset_err = float(np.hypot(ec, er)) * (1 - d) / d * cfg.pixel_arcsec if d > 0 else float("nan")
    v = {
        "dcol_px": dc,
        "drow_px": dr,
        "dcol_err_px": ec,
        "drow_err_px": er,
        "sigma": sig,
        "source_offset_arcsec": offset,
        "source_offset_err_arcsec": offset_err,
        "n_transits": len(shifts),
        "per_sector": per_sector,
    }
    text = (
        f"in-transit shift {shift * 1e3:.2f} mpx ({sig:.1f}σ, {len(shifts)} transits): "
        f"source {offset:.1f} ± {offset_err:.1f}″ from the target"
    )
    # With hundreds of transits even the target's own dimming moves the centroid
    # measurably (other light in the aperture); what matters is where the source is.
    far = offset >= max(cfg.centroid_min_offset_arcsec, cfg.centroid_fail_sigma * offset_err)
    if sig >= cfg.centroid_fail_sigma and far:
        status = "fail"
    elif sig >= cfg.warn_sigma and far:
        status = "warn"
    elif offset_err > 2 * cfg.pixel_arcsec:
        status = "warn"
        text += "; too imprecise to place the source within the aperture"
    else:
        status = "pass"
    if tmag is not None and tmag < 7.0 and status != "pass":
        text += f"; Tmag {tmag:.1f} is near saturation, where moment centroids are unreliable"
        status = "warn" if status == "fail" else status
    return Check(5, "No centroid shift", status, text, v)
