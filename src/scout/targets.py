"""Target selection from the TIC (design §4, ``targets`` stage).

The northern CVZ is a cone around the north ecliptic pole; the TIC is queried
on MAST for that cone with the magnitude, temperature and luminosity-class
cuts from the config, then trimmed to the exact ecliptic-latitude limit and
the contamination cut, and sorted brightest first.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config, TargetsConfig

log = logging.getLogger(__name__)

# North ecliptic pole (J2000).
NEP_RA_DEG = 270.0
NEP_DEC_DEG = 66.560708

COLUMNS = (
    "tic",
    "tmag",
    "teff_k",
    "radius_rsun",
    "mass_msun",
    "ra_deg",
    "dec_deg",
    "ecliptic_lat_deg",
    "contamination_ratio",
)


def ecliptic_latitude(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

    c = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg)
    return c.transform_to(BarycentricMeanEcliptic()).lat.deg


def _f(row: Any, name: str) -> float:
    try:
        v = float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def apply_cuts(rows: list[dict[str, Any]], cfg: TargetsConfig) -> list[dict[str, Any]]:
    """Exact cuts on already-fetched rows; brightest first."""
    out = []
    for r in rows:
        if not cfg.tmag_min <= r["tmag"] <= cfg.tmag_max:
            continue
        if not cfg.teff_min_k <= r["teff_k"] <= cfg.teff_max_k:
            continue
        if r["ecliptic_lat_deg"] < cfg.ecliptic_lat_min_deg:
            continue
        c = r["contamination_ratio"]
        if cfg.contamination_max > 0 and np.isfinite(c) and c > cfg.contamination_max:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["tmag"], r["tic"]))
    if cfg.max_targets > 0:
        out = out[: cfg.max_targets]
    return out


def query_targets(cfg: TargetsConfig, timeout_s: float | None = None) -> list[dict[str, Any]]:
    import astropy.units as u

    from scout.fetch import _require_astroquery, _with_retries

    catalogs, _ = _require_astroquery(timeout_s)
    radius = 90.0 - cfg.ecliptic_lat_min_deg
    criteria: dict[str, Any] = {
        "Tmag": [cfg.tmag_min, cfg.tmag_max],
        "Teff": [cfg.teff_min_k, cfg.teff_max_k],
        "objType": "STAR",
    }
    if cfg.lumclass:
        criteria["lumclass"] = cfg.lumclass
    table = _with_retries(
        "TIC cone query",
        lambda: catalogs.query_criteria(
            catalog="Tic",
            coordinates=f"{NEP_RA_DEG} {NEP_DEC_DEG}",
            radius=radius * u.deg,
            **criteria,
        ),
    )
    ra = np.array([_f(r, "ra") for r in table])
    dec = np.array([_f(r, "dec") for r in table])
    lat = ecliptic_latitude(ra, dec) if len(table) else np.array([])
    rows = [
        {
            "tic": int(r["ID"]),
            "tmag": _f(r, "Tmag"),
            "teff_k": _f(r, "Teff"),
            "radius_rsun": _f(r, "rad"),
            "mass_msun": _f(r, "mass"),
            "ra_deg": ra[i],
            "dec_deg": dec[i],
            "ecliptic_lat_deg": float(lat[i]),
            "contamination_ratio": _f(r, "contratio"),
        }
        for i, r in enumerate(table)
    ]
    log.info("TIC cone: %d rows before exact cuts", len(rows))
    return apply_cuts(rows, cfg)


def targets_path(cfg: Config) -> Path:
    return cfg.run_dir / "targets.csv"


def write_targets(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})
    tmp.replace(path)
    return path


def read_targets(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "tic" not in rows[0]:
        raise ValueError(f"{path}: needs a 'tic' column")
    out = []
    for r in rows:
        d: dict[str, Any] = {"tic": int(float(r["tic"]))}
        for k in COLUMNS[1:]:
            if k in r:
                try:
                    d[k] = float(r[k])
                except ValueError:
                    d[k] = float("nan")
        out.append(d)
    return out
