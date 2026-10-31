"""Gaia DR3 neighbours and RUWE (design §6 item 6).

Queried from the Gaia DR3 copy at MAST (TAP), since the ESA archive and
VizieR are not reachable from every environment. A neighbour "could make the
dip" when, fully eclipsed, it would dim the blended light by at least the
observed depth: ``f / (1 + f) ≥ depth`` with ``f`` its flux relative to the
target. Results are cached per star under ``<cache>/gaia/``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import VetConfig
from scout.vet.checks import Check

_QUERY = (
    "SELECT source_id, ra, dec, phot_g_mean_mag, ruwe FROM dbo.gaia_source "
    "WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1"
)


def query_neighbours(
    ra: float, dec: float, radius_arcsec: float, url: str, cache: Path | None
) -> list[dict[str, Any]]:
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    import pyvo

    svc = pyvo.dal.TAPService(url)
    q = _QUERY.format(ra=ra, dec=dec, r=radius_arcsec / 3600.0)
    last: Exception | None = None
    for attempt in range(4):
        try:
            with warnings.catch_warnings():
                # MAST flags every result as truncated at its row limit, even small ones.
                warnings.simplefilter("ignore", pyvo.dal.exceptions.DALOverflowWarning)
                tab = svc.search(q, maxrec=100000).to_table()
            break
        except Exception as exc:  # network and service errors look alike through pyvo
            last = exc
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"Gaia DR3 query failed: {last}") from last
    rows = []
    for r in tab:
        rows.append(
            {
                "source_id": str(r["source_id"]),
                "ra": float(r["ra"]),
                "dec": float(r["dec"]),
                "g": float(r["phot_g_mean_mag"]) if np.isfinite(r["phot_g_mean_mag"]) else None,
                "ruwe": float(r["ruwe"]) if np.isfinite(r["ruwe"]) else None,
            }
        )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    return rows


def separations(rows: list[dict[str, Any]], ra: float, dec: float) -> None:
    """Add sep_arcsec and the (dx, dy) offsets in arcsec (east, north) to each row."""
    for r in rows:
        dx = (r["ra"] - ra) * np.cos(np.radians(dec)) * 3600
        dy = (r["dec"] - dec) * 3600
        r["dx"], r["dy"], r["sep_arcsec"] = float(dx), float(dy), float(np.hypot(dx, dy))


def identify_target(
    rows: list[dict[str, Any]], gaia_dr2: str | None, tmag: float | None
) -> int | None:
    """Index of the target: same source_id as the TIC's Gaia DR2 id, else brightest within 5″."""
    if gaia_dr2:
        for i, r in enumerate(rows):
            if r["source_id"] == str(gaia_dr2):
                return i
    close = [(r["g"] or 99, i) for i, r in enumerate(rows) if r["sep_arcsec"] < 5.0]
    if close:
        return min(close)[1]
    return None


def check_gaia(
    rows: list[dict[str, Any]],
    star: dict[str, Any],
    depth_ppm: float,
    centroid_status: str,
    cfg: VetConfig,
) -> Check:
    ra, dec = star.get("ra_deg"), star.get("dec_deg")
    if ra is None or dec is None:
        return Check(6, "Gaia neighbours / RUWE", "n/a", "no coordinates")
    separations(rows, ra, dec)
    it = identify_target(rows, star.get("gaia_dr2"), star.get("tmag"))
    if it is None:
        return Check(
            6, "Gaia neighbours / RUWE", "warn", "target not found in Gaia DR3", {"n": len(rows)}
        )
    target = rows[it]
    g0 = target["g"]
    depth = depth_ppm * 1e-6
    capable = []
    for i, r in enumerate(rows):
        r["is_target"] = i == it
        r["capable"] = False
        if i == it or r["g"] is None or g0 is None:
            continue
        f = 10 ** (-0.4 * (r["g"] - g0))
        r["max_depth_ppm"] = f / (1 + f) * 1e6
        if f / (1 + f) >= depth:
            r["capable"] = True
            capable.append(r)
    ruwe = target["ruwe"]
    v = {
        "target_source_id": target["source_id"],
        "target_g": g0,
        "ruwe": ruwe,
        "n_neighbours": len(rows) - 1,
        "n_capable": len(capable),
        "capable": [
            {k: c[k] for k in ("source_id", "g", "sep_arcsec", "max_depth_ppm")} for c in capable
        ],
    }
    parts = [f"RUWE {ruwe:.2f}" if ruwe is not None else "RUWE unknown"]
    if capable:
        c = min(capable, key=lambda r: r["sep_arcsec"])
        parts.append(
            f"{len(capable)} neighbour(s) within {cfg.neighbour_radius_arcsec:.0f}″ bright "
            f"enough to make the dip (closest G = {c['g']:.1f} at {c['sep_arcsec']:.0f}″)"
        )
    else:
        parts.append(
            f"no neighbour within {cfg.neighbour_radius_arcsec:.0f}″ bright enough to make the dip"
        )
    text = "; ".join(parts)
    status = "pass"
    if ruwe is not None and ruwe > cfg.ruwe_max:
        status = "warn"
        text += f" (RUWE > {cfg.ruwe_max:g}: possible unresolved binary)"
    if capable:
        # A centroid that clearly stays on target clears neighbours outside ~1 pixel.
        close = [c for c in capable if c["sep_arcsec"] < cfg.pixel_arcsec]
        if centroid_status != "pass" or close:
            status = "warn"
            text += "; needs a difference image to rule them out"
    return Check(6, "Gaia neighbours / RUWE", status, text, v)
