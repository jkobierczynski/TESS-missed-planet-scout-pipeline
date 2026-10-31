"""Triage: is a detection already known? (design §4, ``triage`` stage)

Every detection that passes the search thresholds is checked against the
ExoFOP TOI and CTOI lists and the NASA Exoplanet Archive (plus any local
catalogues, e.g. the TESS EB catalogue or a VSX extract):

- ``known``: the same star has a known object at this period (or 1/2, 2, 1/3,
  3 × it). Expected for TOI hosts; recovering them is the sanity check.
- ``neighbour``: a known object on another star within
  ``triage.neighbour_radius_arcsec`` has this period, so the signal is
  probably that object blended into our aperture.
- ``new_on_known_host``: the star has known objects, but not at this period.
- ``new``: nothing known on or near the star.

Signals that are not known but fail a cheap check are then demoted to
``duplicate``, ``too_large`` or ``inverted`` (see :func:`cheap_kills`).

A TOI whose TFOPWG disposition is FP (false positive) or EB is reported as
such, since a known false positive at our period is as good as a kill.

Catalogues are downloaded into ``<cache>/catalogs/`` and reused for
``triage.max_age_days``. The result is ``results/<run>/candidates.parquet``
with every detection of every star, and ``summary.html`` for review.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config, TriageConfig

log = logging.getLogger(__name__)

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
PS_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+pl_name,tic_id,pl_orbper,ra,dec+from+pscomppars+where+tic_id+is+not+null"
    "&format=csv"
)
RATIOS = (1.0, 0.5, 2.0, 1 / 3, 3.0)
BAD_DISPOSITIONS = {"FP", "FA", "EB", "NEB", "BEB", "V"}


@dataclass(frozen=True)
class KnownObject:
    tic: int
    name: str
    kind: str  # "TOI", "CTOI", "planet", or a local catalogue's kind (e.g. "EB")
    period_days: float  # nan when unknown (single transits)
    ra_deg: float
    dec_deg: float
    disposition: str = ""


# --- catalogues ------------------------------------------------------------------


def _download(url: str, path: Path, max_age_days: float, timeout_s: float = 120.0) -> Path:
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        return path
    import requests

    from scout.fetch import _with_retries

    log.info("downloading %s", url)
    resp = _with_retries(f"download of {path.name}", lambda: requests.get(url, timeout=timeout_s))
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _tic(x: Any) -> int | None:
    digits = "".join(c for c in str(x) if c.isdigit() or c == ".").split(".")[0]
    return int(digits) if digits else None


def _coords(ra: str, dec: str) -> tuple[float, float]:
    """Decimal degrees from either decimal or sexagesimal strings."""
    r, d = _num(ra), _num(dec)
    if np.isfinite(r) and np.isfinite(d):
        return r, d
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return float("nan"), float("nan")


def parse_toi(text: str) -> list[KnownObject]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        tic = _tic(r.get("TIC ID"))
        if tic is None:
            continue
        ra, dec = _coords(r.get("RA", ""), r.get("Dec", ""))
        out.append(
            KnownObject(
                tic=tic,
                name=f"TOI-{r.get('TOI', '').strip()}",
                kind="TOI",
                period_days=_num(r.get("Period (days)")),
                ra_deg=ra,
                dec_deg=dec,
                disposition=(
                    r.get("TFOPWG Disposition") or r.get("TESS Disposition") or ""
                ).strip(),
            )
        )
    return out


def parse_ctoi(text: str) -> list[KnownObject]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        tic = _tic(r.get("TIC ID"))
        if tic is None:
            continue
        ra, dec = _coords(r.get("RA", ""), r.get("Dec", ""))
        out.append(
            KnownObject(
                tic=tic,
                name=f"CTOI-{r.get('CTOI', '').strip()}",
                kind="CTOI",
                period_days=_num(r.get("Period (days)")),
                ra_deg=ra,
                dec_deg=dec,
                disposition=(
                    r.get("TFOPWG Disposition") or r.get("User Disposition") or ""
                ).strip(),
            )
        )
    return out


def parse_planets(text: str) -> list[KnownObject]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        tic = _tic(r.get("tic_id"))
        if tic is None:
            continue
        out.append(
            KnownObject(
                tic=tic,
                name=str(r.get("pl_name", "")).strip(),
                kind="planet",
                period_days=_num(r.get("pl_orbper")),
                ra_deg=_num(r.get("ra")),
                dec_deg=_num(r.get("dec")),
                disposition="CP",
            )
        )
    return out


def parse_local(text: str) -> list[KnownObject]:
    """Local CSV: tic, period_days, name, kind (+ optional ra_deg, dec_deg)."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        tic = _tic(r.get("tic"))
        if tic is None:
            continue
        out.append(
            KnownObject(
                tic=tic,
                name=str(r.get("name", "")).strip(),
                kind=str(r.get("kind", "local")).strip() or "local",
                period_days=_num(r.get("period_days")),
                ra_deg=_num(r.get("ra_deg")),
                dec_deg=_num(r.get("dec_deg")),
                disposition=str(r.get("kind", "")).strip(),
            )
        )
    return out


def load_known(cfg: Config) -> tuple[list[KnownObject], dict[str, Any]]:
    """All known objects, and where each list came from."""
    tc = cfg.triage
    d = cfg.run.cache_dir / "catalogs"
    sources: dict[str, Any] = {}
    objs: list[KnownObject] = []
    for name, url, parse in (
        ("toi.csv", TOI_URL, parse_toi),
        ("ctoi.csv", CTOI_URL, parse_ctoi),
        ("pscomppars.csv", PS_URL, parse_planets),
    ):
        path = _download(url, d / name, tc.max_age_days)
        got = parse(path.read_text(encoding="utf-8", errors="replace"))
        objs += got
        sources[name] = {
            "url": url,
            "n": len(got),
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
        }
    for extra in tc.extra_catalogs:
        p = Path(extra)
        got = parse_local(p.read_text(encoding="utf-8", errors="replace"))
        objs += got
        sources[p.name] = {"path": str(p), "n": len(got)}
    return objs, sources


# --- matching --------------------------------------------------------------------


def period_match(p: float, known: float, tol: float) -> float | None:
    """The ratio p/known matches (1, 1/2, 2, 1/3, 3), or None."""
    if not (np.isfinite(p) and np.isfinite(known)) or known <= 0:
        return None
    for k in RATIOS:
        if abs(p / (k * known) - 1) < tol:
            return k
    return None


def _sep_arcsec(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


class KnownIndex:
    def __init__(self, objs: list[KnownObject]) -> None:
        self.by_tic: dict[int, list[KnownObject]] = {}
        for o in objs:
            self.by_tic.setdefault(o.tic, []).append(o)
        pos = [o for o in objs if np.isfinite(o.ra_deg) and np.isfinite(o.dec_deg)]
        self.pos = pos
        self.ra = np.array([o.ra_deg for o in pos])
        self.dec = np.array([o.dec_deg for o in pos])

    def near(self, ra: float, dec: float, radius_arcsec: float, exclude_tic: int) -> list:
        if not len(self.pos) or not (np.isfinite(ra) and np.isfinite(dec)):
            return []
        # Cheap box first, exact separation after.
        box = (np.abs(self.dec - dec) < radius_arcsec / 3600) & (
            np.abs((self.ra - ra + 180) % 360 - 180) * np.cos(np.radians(dec))
            < radius_arcsec / 3600
        )
        idx = np.flatnonzero(box)
        if not len(idx):
            return []
        sep = _sep_arcsec(ra, dec, self.ra[idx], self.dec[idx])
        return [
            (self.pos[i], float(s))
            for i, s in zip(idx, sep, strict=True)
            if s <= radius_arcsec and self.pos[i].tic != exclude_tic
        ]


def classify(
    tic: int,
    period: float,
    ra: float,
    dec: float,
    index: KnownIndex,
    tc: TriageConfig,
) -> dict[str, Any]:
    """Triage verdict for one detection."""
    on_star = index.by_tic.get(tic, [])
    for o in on_star:
        k = period_match(period, o.period_days, tc.period_tolerance)
        if k is not None:
            bad = o.disposition.upper() in BAD_DISPOSITIONS
            return {
                "triage": "known_false_positive" if bad else "known",
                "match": o.name,
                "match_kind": o.kind,
                "match_disposition": o.disposition,
                "match_ratio": k,
                "match_sep_arcsec": 0.0,
            }
    for o, sep in sorted(index.near(ra, dec, tc.neighbour_radius_arcsec, tic), key=lambda x: x[1]):
        k = period_match(period, o.period_days, tc.period_tolerance)
        if k is not None:
            return {
                "triage": "neighbour",
                "match": o.name,
                "match_kind": o.kind,
                "match_disposition": o.disposition,
                "match_ratio": k,
                "match_sep_arcsec": round(sep, 1),
            }
    names = ", ".join(sorted({o.name for o in on_star}))
    return {
        "triage": "new_on_known_host" if on_star else "new",
        "match": names,
        "match_kind": "",
        "match_disposition": "",
        "match_ratio": float("nan"),
        "match_sep_arcsec": float("nan"),
    }


def _too_long(r: dict[str, Any], factor: float) -> bool:
    from scout.inject import transit_duration_hours

    rad, mass = _num(r.get("radius_rsun")), _num(r.get("mass_msun"))
    p, dur = _num(r.get("period_days")), _num(r.get("duration_hours"))
    if factor <= 0 or not all(np.isfinite(x) and x > 0 for x in (rad, mass, p, dur)):
        return False
    rp_rs = _num(r.get("rp_rs"))
    k = rp_rs if np.isfinite(rp_rs) else 0.1
    return dur > factor * transit_duration_hours(p, k, 0.0, rad, mass)


def cheap_kills(rows: list[dict[str, Any]], tc: TriageConfig) -> None:
    """Demote "new" signals that cheap checks already explain (design §3.7).

    - ``duplicate``: the same star has an earlier detection at this period
      (or an alias of it): the search found the same signal again after masking.
    - ``too_large``: implied companion radius above ``max_radius_rearth``
      (about 2 R_Jup): an eclipsing binary, not a planet (checklist item 7).
    - ``too_long``: longer than ``max_duration_factor`` × the longest
      central transit a circular orbit allows at this period around this star
      (checklist item 8): stellar variability or a blend, not a transit.
    - ``inverted``: the flux-inverted search did at least as well, so the
      dips are as likely systematics or variability as transits.

    The full §6 checklist still runs on whatever stays "new".
    """
    seen: dict[int, list[float]] = {}
    for r in sorted(rows, key=lambda r: (r["tic"], r["index"])):
        if r["triage"] == "below_threshold":
            continue
        earlier = seen.setdefault(r["tic"], [])
        dup = any(period_match(r["period_days"], p, tc.period_tolerance) for p in earlier)
        earlier.append(r["period_days"])
        if r["triage"] not in ("new", "new_on_known_host"):
            continue
        rp, inv = _num(r.get("rp_rearth")), _num(r.get("inverted_sde"))
        if dup:
            r["triage"] = "duplicate"
        elif np.isfinite(rp) and rp > tc.max_radius_rearth:
            r["triage"] = "too_large"
        elif _too_long(r, tc.max_duration_factor):
            r["triage"] = "too_long"
        elif np.isfinite(inv) and inv >= r["sde"]:
            r["triage"] = "inverted"


# --- driver ----------------------------------------------------------------------


def collect_detections(cfg: Config) -> list[dict[str, Any]]:
    """One row per detection of every searched star in the run."""
    from scout.pipeline import load_search, report_path

    rows = []
    root = cfg.run_dir / "stars"
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        sp = d / "search.json"
        if not sp.exists():
            continue
        try:
            star = json.loads((d / "fetch.json").read_text()).get("star") or {}
        except (OSError, ValueError):
            star = {}
        res = load_search(sp)
        tic = int(d.name.removeprefix("TIC"))
        for det in res.detections:
            row = {
                k: v
                for k, v in asdict(det).items()
                if k not in {"transit_times_btjd", "per_transit_depth_ppm", "warnings"}
            }
            row["tic"] = tic
            row["flags"] = ",".join(det.flags)
            row["inverted_sde"] = res.inverted_sde
            row["tmag"] = star.get("tmag")
            row["teff_k"] = star.get("teff_k")
            row["radius_rsun"] = star.get("radius_rsun")
            row["mass_msun"] = star.get("mass_msun")
            row["ra_deg"] = star.get("ra_deg")
            row["dec_deg"] = star.get("dec_deg")
            r = star.get("radius_rsun")
            row["rp_rearth"] = det.rp_rs * r * 109.08 if r and np.isfinite(det.rp_rs) else None
            rp = report_path(cfg, tic)
            row["report"] = str(rp.relative_to(cfg.run_dir)) if rp.exists() else ""
            rows.append(row)
    return rows


def triage_run(cfg: Config) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scout.provenance import provenance
    from scout.report.summary import write_summary

    objs, sources = load_known(cfg)
    index = KnownIndex(objs)
    rows = collect_detections(cfg)
    for r in rows:
        if r["passes_threshold"]:
            r.update(
                classify(
                    r["tic"],
                    r["period_days"],
                    _num(r["ra_deg"]),
                    _num(r["dec_deg"]),
                    index,
                    cfg.triage,
                )
            )
        else:
            r.update(
                triage="below_threshold",
                match="",
                match_kind="",
                match_disposition="",
                match_ratio=float("nan"),
                match_sep_arcsec=float("nan"),
            )
    cheap_kills(rows, cfg.triage)
    out = cfg.run_dir / "candidates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({"tic": pa.array([], pa.int64())})
    meta = {"catalogs": sources, "provenance": provenance({"config_digest": cfg.digest()})}
    table = table.replace_schema_metadata({b"scout": json.dumps(meta, default=str)})
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["triage"]] = counts.get(r["triage"], 0) + 1
    n_stars = len({r["tic"] for r in rows})
    summary = write_summary(cfg.run_dir / "summary.html", cfg, rows, counts, meta, n_stars)
    return {
        "candidates": str(out),
        "summary": str(summary),
        "counts": counts,
        "n_stars": n_stars,
        "n_known": len(objs),
    }
