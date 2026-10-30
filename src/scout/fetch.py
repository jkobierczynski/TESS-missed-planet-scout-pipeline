"""Download TESS light curves from MAST into the immutable cache, and read them.

Raw FITS files land under ``<cache>/mast/`` exactly as MAST serves them and are
never modified. Each fetch writes a manifest (which file was chosen for which
sector, plus TIC stellar parameters) into the run directory; later stages read
the FITS files through :func:`read_lightcurve_fits` using that manifest.

MAST access needs the optional ``astroquery`` dependency (``pip install
scout[mast]``). Everything else in this module works offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import FetchConfig
from scout.lightcurve import LightCurve

log = logging.getLogger(__name__)

# QLP quality flags are not bit-compatible with SPOC's. Any non-zero QLP flag
# is mapped onto SPOC's "manual exclude" bit, which the default bitmask rejects.
_SPOC_MANUAL_EXCLUDE = 128


@dataclass(frozen=True)
class Product:
    """One selected light-curve file for one sector."""

    sector: int
    source: str  # SPOC | TESS-SPOC | QLP
    cadence_s: int
    obs_id: str
    filename: str
    url: str
    local_path: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A light-curve product as listed by MAST, before selection."""

    sector: int
    source: str
    cadence_s: int
    obs_id: str
    filename: str
    url: str


def select_products(cands: list[Candidate], cfg: FetchConfig) -> list[Product]:
    """Pick one product per sector: preferred source first, then shortest cadence."""
    rank = {s: i for i, s in enumerate(cfg.sources)}
    allowed = set(cfg.cadences_s)
    sectors = set(cfg.sectors)
    best: dict[int, Candidate] = {}
    for c in cands:
        if c.source not in rank or c.cadence_s not in allowed:
            continue
        if sectors and c.sector not in sectors:
            continue
        cur = best.get(c.sector)
        key = (rank[c.source], c.cadence_s, c.filename)
        if cur is None or key < (rank[cur.source], cur.cadence_s, cur.filename):
            best[c.sector] = c
    return [Product(**asdict(best[s])) for s in sorted(best)]


# --- MAST ----------------------------------------------------------------------


def _require_astroquery(timeout_s: float | None = None) -> Any:
    try:
        from astroquery.mast import Catalogs, Observations
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("MAST access needs astroquery: pip install 'scout[mast]'") from exc
    if timeout_s:
        # astroquery copies its timeout into each connection object at import,
        # so changing astroquery.mast.conf afterwards has no effect.
        for obj in (Catalogs, Observations):
            for target in (
                obj,
                getattr(obj, "_portal_api_connection", None),
                getattr(obj, "_service_api_connection", None),
            ):
                if target is not None and hasattr(target, "TIMEOUT"):
                    target.TIMEOUT = timeout_s
    return Catalogs, Observations


# Delays before the 2nd, 3rd, ... attempt of a MAST call. Connections through
# an HTTPS proxy are occasionally dropped mid-request; a retry usually works.
_RETRY_DELAYS_S = (2.0, 8.0, 30.0)


def _network_errors() -> tuple[type[BaseException], ...]:
    errors: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    try:
        import requests
    except ImportError:  # pragma: no cover - requests comes with astroquery
        return errors
    return (*errors, requests.exceptions.RequestException)


def _with_retries(
    what: str, call: Callable[[], Any], delays: tuple[float, ...] = _RETRY_DELAYS_S
) -> Any:
    """Run ``call``, retrying on network errors with the given backoff."""
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return call()
        except _network_errors() as exc:
            if delay is None:
                raise
            log.warning("%s failed (attempt %d): %s; retrying in %.0f s", what, attempt, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def query_tic(tic: int, timeout_s: float | None = None) -> dict[str, Any]:
    """Stellar parameters from the TIC (v8.2 via MAST)."""
    catalogs, _ = _require_astroquery(timeout_s)
    rows = _with_retries(
        f"TIC query for {tic}", lambda: catalogs.query_criteria(catalog="Tic", ID=int(tic))
    )
    if len(rows) == 0:
        raise LookupError(f"TIC {tic} not found")
    r = rows[0]

    def get(name: str) -> float | None:
        try:
            v = float(r[name])
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    return {
        "tic": int(tic),
        "ra_deg": get("ra"),
        "dec_deg": get("dec"),
        "ecliptic_lat_deg": get("eclat"),
        "tmag": get("Tmag"),
        "teff_k": get("Teff"),
        "logg": get("logg"),
        "radius_rsun": get("rad"),
        "mass_msun": get("mass"),
        "gaia_dr2": str(r["GAIA"]) if "GAIA" in r.colnames else None,
        "contamination_ratio": get("contratio"),
    }


_SECTOR_RE = re.compile(r"(?:-s|_s)0*(\d{1,4})[-_]")


def _sector_from_filename(name: str) -> int | None:
    m = _SECTOR_RE.search(name)
    return int(m.group(1)) if m else None


def list_products(tic: int, cfg: FetchConfig) -> list[Candidate]:
    """All SPOC/TESS-SPOC/QLP light-curve files MAST has for a TIC ID."""
    _, observations = _require_astroquery(cfg.timeout_s)
    obs = _with_retries(
        f"MAST observation query for TIC {tic}",
        lambda: observations.query_criteria(
            target_name=str(int(tic)),
            dataproduct_type="timeseries",
            provenance_name=list(cfg.sources),
        ),
    )
    if len(obs) == 0:
        return []
    prods = _with_retries(
        f"MAST product list for TIC {tic}", lambda: observations.get_product_list(obs)
    )
    by_obsid = {str(o["obsid"]): o for o in obs}
    cands: list[Candidate] = []
    for p in prods:
        fn = str(p["productFilename"])
        if not fn.endswith(".fits"):
            continue
        # SPOC: *_lc.fits; TESS-SPOC HLSP: *_lc.fits; QLP HLSP: *_llc.fits.
        if not (fn.endswith("_lc.fits") or fn.endswith("_llc.fits")):
            continue
        o = by_obsid.get(str(p["parent_obsid"])) or by_obsid.get(str(p["obsID"]))
        if o is None:
            continue
        source = str(o["provenance_name"])
        sector = _int_or_none(o["sequence_number"]) or _sector_from_filename(fn)
        if sector is None:
            continue
        cadence = _cadence_from_obs(o, fn)
        cands.append(
            Candidate(
                sector=sector,
                source=source,
                cadence_s=cadence,
                obs_id=str(o["obs_id"]),
                filename=fn,
                url=str(p["dataURI"]),
            )
        )
    return cands


def _int_or_none(x: Any) -> int | None:
    try:
        v = int(x)
    except (TypeError, ValueError):
        return None
    return v


def _cadence_from_obs(o: Any, filename: str) -> int:
    try:
        t = float(o["t_exptime"])
        if np.isfinite(t) and t > 0:
            # Round to the nominal TESS cadences.
            return min((20, 120, 200, 600, 1800), key=lambda c: abs(c - t))
    except (KeyError, TypeError, ValueError):
        pass
    return 120 if "fast" not in filename else 20


def fetch(tic: int, cache_dir: Path, cfg: FetchConfig) -> dict[str, Any]:
    """Download the selected products for ``tic`` and return a manifest."""
    _, observations = _require_astroquery(cfg.timeout_s)
    star = query_tic(tic, cfg.timeout_s)
    products = select_products(list_products(tic, cfg), cfg)
    dest = Path(cache_dir) / "mast"
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Product] = []
    for p in products:
        res = _with_retries(
            f"download of {p.filename}",
            lambda p=p: observations.download_file(
                p.url, local_path=str(dest / p.filename), cache=True
            ),
        )
        if res[0] != "COMPLETE":
            raise RuntimeError(f"download of {p.filename} failed: {res}")
        out.append(Product(**{**asdict(p), "local_path": str(dest / p.filename)}))
    return {"tic": int(tic), "star": star, "products": [asdict(p) for p in out]}


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# --- Reading FITS ---------------------------------------------------------------


def read_lightcurve_fits(path: str | Path, source: str, fetch_cfg: FetchConfig) -> LightCurve:
    """Read one SPOC / TESS-SPOC / QLP light-curve FITS file."""
    from astropy.io import fits

    flux_col = fetch_cfg.flux_columns[source]
    with fits.open(path, memmap=False) as hdul:
        h0, h1 = hdul[0].header, hdul[1].header
        d = hdul[1].data
        cols = {c.upper() for c in d.columns.names}
        time = np.asarray(d["TIME"], dtype=float)
        flux = np.asarray(d[flux_col], dtype=float)
        err_col = next(
            (c for c in (f"{flux_col}_ERR", "KSPSAP_FLUX_ERR", "SAP_FLUX_ERR") if c in cols), None
        )
        flux_err = np.asarray(d[err_col], dtype=float) if err_col else np.full_like(flux, np.nan)
        qcol = "QUALITY" if "QUALITY" in cols else None
        quality = np.asarray(d[qcol], dtype=np.int64) if qcol else np.zeros(len(time), np.int64)
        if source == "QLP":
            quality = np.where(quality != 0, _SPOC_MANUAL_EXCLUDE, 0)
        sector = int(h0.get("SECTOR", -1))
        tic = h0.get("TICID", h0.get("OBJECT"))
        timedel = h1.get("TIMEDEL")
        cadence_s = round(float(timedel) * 86400) if timedel else 0

    if not np.isfinite(flux_err).any():
        # QLP files lacking an error column: use the point-to-point scatter.
        good = np.isfinite(flux)
        sig = 1.4826 * np.nanmedian(np.abs(np.diff(flux[good]))) / np.sqrt(2)
        flux_err = np.full_like(flux, sig)

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality.astype(np.int32),
        sector=np.full(len(time), sector),
        cadence_s=np.full(len(time), cadence_s),
        meta={
            "tic": _tic_int(tic),
            "source": source,
            "file": Path(path).name,
            "flux_column": flux_col,
            "procver": h0.get("PROCVER"),
            "data_rel": h0.get("DATA_REL"),
        },
    )


def _tic_int(x: Any) -> int | None:
    if x is None:
        return None
    m = re.search(r"\d+", str(x))
    return int(m.group(0)) if m else None


def load_from_manifest(manifest: dict[str, Any], fetch_cfg: FetchConfig) -> list[LightCurve]:
    return [
        read_lightcurve_fits(p["local_path"], p["source"], fetch_cfg) for p in manifest["products"]
    ]
