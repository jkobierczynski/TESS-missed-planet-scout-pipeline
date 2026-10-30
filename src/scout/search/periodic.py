"""Periodic transit search with Transit Least Squares.

Signals are found iteratively: search, record the strongest peak, mask its
transits, search again, up to ``max_signals`` or until the peak falls below
threshold. The same search is run on the flux-inverted light curve; its best
SDE is a per-star false-alarm yardstick (design §5), since systematics that
look like dips look just as much like bumps.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from scout.config import PeriodicSearchConfig
from scout.lightcurve import LightCurve, bin_per_sector


@dataclass
class Detection:
    tic: int | None
    kind: str  # "periodic" | "single" (M4)
    index: int  # 1-based order found within the star
    period_days: float
    period_err_days: float
    t0_btjd: float
    duration_hours: float
    depth_ppm: float
    snr: float
    sde: float
    n_transits: int  # transits covered by data
    n_transits_expected: int
    odd_even_sigma: float
    rp_rs: float
    passes_threshold: bool
    transit_times_btjd: list[float] = field(default_factory=list)
    per_transit_depth_ppm: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PeriodicSearchResult:
    detections: list[Detection]
    inverted_sde: float | None
    period_grid_days: tuple[float, float]
    n_points: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "inverted_sde": self.inverted_sde,
            "period_grid_days": list(self.period_grid_days),
            "n_points": self.n_points,
            "meta": self.meta,
        }


def period_range(lc: LightCurve, cfg: PeriodicSearchConfig) -> tuple[float, float]:
    pmax = cfg.period_max_baseline_fraction * lc.baseline_days
    if cfg.period_max_days > 0:
        pmax = min(pmax, cfg.period_max_days)
    if pmax <= cfg.period_min_days:
        raise ValueError(
            f"baseline {lc.baseline_days:.1f} d too short for period_min {cfg.period_min_days} d"
        )
    return cfg.period_min_days, pmax


def _run_tls(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    pmin: float,
    pmax: float,
    cfg: PeriodicSearchConfig,
    stellar: dict[str, float],
) -> Any:
    from transitleastsquares import transitleastsquares

    threads = cfg.threads if cfg.threads > 0 else multiprocessing.cpu_count()
    model = transitleastsquares(t, y, dy, verbose=False)
    # TLS prints banners even with verbose=False in some versions, and its
    # warnings (e.g. "true period may be twice") belong in the result, not stderr.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.power(
            period_min=pmin,
            period_max=pmax,
            n_transits_min=cfg.min_transits,
            oversampling_factor=cfg.oversampling_factor,
            duration_grid_step=cfg.duration_grid_step,
            use_threads=threads,
            show_progress_bar=False,
            verbose=False,
            **stellar,
        )
    res.scout_warnings = sorted({str(w.message) for w in caught})
    return res


def _stellar_kwargs(meta: dict[str, Any]) -> dict[str, float]:
    """Use TIC radius/mass when known to set TLS's duration grid sensibly."""
    star = meta.get("star") or {}
    out: dict[str, float] = {}
    r, m = star.get("radius_rsun"), star.get("mass_msun")
    if r and np.isfinite(r):
        out.update(R_star=float(r), R_star_min=max(0.07, 0.5 * r), R_star_max=2.0 * r)
    if m and np.isfinite(m):
        out.update(M_star=float(m), M_star_min=max(0.07, 0.5 * m), M_star_max=2.0 * m)
    return out


def _finite(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def period_may_be_double(r: Any) -> bool:
    """True when every transit with data falls on epochs of the same parity.

    Then the fold cannot tell P from 2P: the "missing" half of the transits was
    simply never observed. TLS's own warning ("true period may be twice") fires
    whenever a third of the predicted transits have no data, which on sparse
    multi-year sector coverage is nearly always, so it is kept in ``warnings``
    but not used as a flag.
    """
    times = np.atleast_1d(np.asarray(r.transit_times, dtype=float))
    counts = np.atleast_1d(np.asarray(r.per_transit_count, dtype=float))
    period, t0 = _finite(r.period), _finite(r.T0)
    if len(times) != len(counts) or not (np.isfinite(period) and np.isfinite(t0)):
        return False
    epochs = np.rint((times[counts > 0] - t0) / period).astype(np.int64)
    if len(epochs) < 2:
        return False
    return len(np.unique(epochs % 2)) == 1


def search_periodic(lc: LightCurve, cfg: PeriodicSearchConfig) -> PeriodicSearchResult:
    from transitleastsquares import transit_mask

    work = bin_per_sector(lc, cfg.bin_minutes)
    pmin, pmax = period_range(work, cfg)
    stellar = _stellar_kwargs(lc.meta)
    t, y, dy = work.time.copy(), work.flux.copy(), work.flux_err.copy()

    detections: list[Detection] = []
    for i in range(1, cfg.max_signals + 1):
        if len(t) < 100:
            break
        r = _run_tls(t, y, dy, pmin, pmax, cfg, stellar)
        sde, snr = _finite(r.SDE, 0.0), _finite(r.snr, 0.0)
        n_tr = int(_finite(r.distinct_transit_count, 0))
        passes = sde >= cfg.sde_threshold and snr >= cfg.snr_threshold and n_tr >= cfg.min_transits
        flags = []
        if n_tr < cfg.min_transits:
            flags.append("too_few_transits")
        if period_may_be_double(r):
            flags.append("period_may_be_double")
        per_depth = [
            (1.0 - _finite(d)) * 1e6 for d in np.atleast_1d(r.transit_depths) if np.isfinite(d)
        ]
        det = Detection(
            tic=lc.tic,
            kind="periodic",
            index=i,
            period_days=_finite(r.period),
            period_err_days=_finite(r.period_uncertainty),
            t0_btjd=_finite(r.T0),
            duration_hours=_finite(r.duration) * 24.0,
            depth_ppm=(1.0 - _finite(r.depth)) * 1e6,
            snr=snr,
            sde=sde,
            n_transits=n_tr,
            n_transits_expected=int(_finite(r.transit_count, 0)),
            odd_even_sigma=_finite(r.odd_even_mismatch),
            rp_rs=_finite(r.rp_rs),
            passes_threshold=passes,
            transit_times_btjd=[float(x) for x in np.atleast_1d(r.transit_times)],
            per_transit_depth_ppm=per_depth,
            flags=flags,
            warnings=list(r.scout_warnings),
        )
        detections.append(det)
        if not passes:
            break
        # Mask 2× the duration around each transit before looking again.
        intransit = transit_mask(t, det.period_days, 2 * det.duration_hours / 24.0, det.t0_btjd)
        t, y, dy = t[~intransit], y[~intransit], dy[~intransit]

    inverted_sde = None
    if cfg.run_inverted:
        r = _run_tls(work.time, 2.0 - work.flux, work.flux_err, pmin, pmax, cfg, stellar)
        inverted_sde = _finite(r.SDE, 0.0)
        for d in detections:
            if inverted_sde >= d.sde:
                d.flags.append("inverted_sde_ge_signal")

    meta = {
        "method": "tls",
        "bin_minutes": cfg.bin_minutes,
        "stellar_priors": stellar,
        "thresholds": {"sde": cfg.sde_threshold, "snr": cfg.snr_threshold},
        "min_transits": cfg.min_transits,
        "oversampling_factor": cfg.oversampling_factor,
        "duration_grid_step": cfg.duration_grid_step,
    }
    return PeriodicSearchResult(
        detections=detections,
        inverted_sde=inverted_sde,
        period_grid_days=(pmin, pmax),
        n_points=len(work),
        meta=meta,
    )
