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


def _box_depth(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, period: float, t0: float, dur_days: float, k: int
) -> tuple[float, float, float, float, np.ndarray]:
    """Box depths at period ``period`` split by epoch: n % k == 0 (old) vs the rest (new).

    Returns (depth_old, err_old, depth_new, err_new, epochs_with_data).
    """
    phase = (t - t0) / period
    n = np.rint(phase).astype(np.int64)
    intr = np.abs(phase - n) * period < 0.5 * dur_days
    w = 1.0 / np.square(dy)
    out = []
    for sel in (intr & (n % k == 0), intr & (n % k != 0)):
        if not sel.any():
            out += [float("nan"), float("inf")]
            continue
        wsum = float(np.sum(w[sel]))
        out += [1.0 - float(np.sum(w[sel] * y[sel]) / wsum), 1.0 / np.sqrt(wsum)]
    return out[0], out[1], out[2], out[3], np.unique(n[intr])


def resolve_submultiple(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    kmax: int,
    period_min: float,
) -> int:
    """Largest k ≤ kmax for which the true period looks like period / k.

    TLS cannot fit a transit longer than 12 % of the trial period (a constant
    compiled into it). Short-period planets around large stars have longer
    transits than that, so TLS reports 2× or 3× the period instead, fitting
    every second or third transit. Here the epochs that P/k adds are checked:
    if they carry a significant dip of comparable depth, the period is P/k.
    """
    dur = duration_hours / 24.0
    best = 1
    for k in range(2, kmax + 1):
        pk = period / k
        if pk < period_min or dur >= 0.5 * pk:
            continue
        d_old, e_old, d_new, e_new, _ = _box_depth(t, y, dy, pk, t0, dur, k)
        if not (np.isfinite(d_old) and np.isfinite(d_new)) or d_old <= 0:
            continue
        significant = d_new / e_new >= 5.0
        comparable = abs(d_new - d_old) <= max(3.0 * np.hypot(e_new, e_old), 0.3 * d_old)
        if significant and comparable:
            best = k
    return best


def _apply_submultiple(
    det: Detection, k: int, t: np.ndarray, y: np.ndarray, dy: np.ndarray
) -> Detection:
    """Rewrite a detection for period P/k, recomputing what depends on the period."""
    p = det.period_days / k
    dur = det.duration_hours / 24.0
    phase = (t - det.t0_btjd) / p
    n = np.rint(phase).astype(np.int64)
    intr = np.abs(phase - n) * p < 0.5 * dur
    epochs = np.unique(n[intr])
    w = 1.0 / np.square(dy)
    per_depth, odd, even = [], [], []
    for e in epochs:
        sel = intr & (n == e)
        d = 1.0 - float(np.sum(w[sel] * y[sel]) / np.sum(w[sel]))
        per_depth.append(d * 1e6)
        (odd if e % 2 else even).append((d, 1.0 / np.sqrt(np.sum(w[sel]))))

    def mean(xs: list[tuple[float, float]]) -> tuple[float, float]:
        if not xs:
            return float("nan"), float("inf")
        v = np.array(xs)
        ww = 1.0 / np.square(v[:, 1])
        return float(np.sum(ww * v[:, 0]) / np.sum(ww)), float(1.0 / np.sqrt(np.sum(ww)))

    (do, eo), (de, ee) = mean(odd), mean(even)
    depth = 1.0 - float(np.mean(y[intr]))
    # Same SNR definition as TLS: mean depth / out-of-transit scatter × √N_in.
    oot = np.abs(phase - n) * p > dur
    snr = depth / float(np.std(y[oot])) * np.sqrt(int(intr.sum())) if oot.any() else det.snr
    tmin, tmax = float(np.min(t)), float(np.max(t))
    first = int(np.ceil((tmin - det.t0_btjd) / p))
    last = int(np.floor((tmax - det.t0_btjd) / p))
    det.period_days = p
    det.period_err_days = det.period_err_days / k
    det.depth_ppm = depth * 1e6
    det.snr = float(snr)
    det.n_transits = len(epochs)
    det.n_transits_expected = max(last - first + 1, len(epochs))
    det.odd_even_sigma = (
        abs(do - de) / (eo + ee) if np.isfinite(do) and np.isfinite(de) else float("nan")
    )
    det.transit_times_btjd = [det.t0_btjd + e * p for e in epochs]
    det.per_transit_depth_ppm = per_depth
    det.flags = [f for f in det.flags if f != "period_may_be_double"]
    det.flags.append(f"period_divided_by_{k}")
    det.warnings.append(
        f"TLS reported {det.period_days * k:.5f} d; the epochs in between show the same dip, "
        f"so the period is 1/{k} of that (TLS caps duration at 12 % of the period)."
    )
    return det


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
        if cfg.resolve_submultiples and np.isfinite(det.period_days):
            k = resolve_submultiple(
                t,
                y,
                dy,
                det.period_days,
                det.t0_btjd,
                det.duration_hours,
                cfg.submultiple_max,
                pmin,
            )
            if k > 1:
                det = _apply_submultiple(det, k, t, y, dy)
                passes = (
                    sde >= cfg.sde_threshold
                    and det.snr >= cfg.snr_threshold
                    and det.n_transits >= cfg.min_transits
                )
                det.passes_threshold = passes
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
        "submultiple_max": cfg.submultiple_max if cfg.resolve_submultiples else 1,
    }
    return PeriodicSearchResult(
        detections=detections,
        inverted_sde=inverted_sde,
        period_grid_days=(pmin, pmax),
        n_points=len(work),
        meta=meta,
    )
