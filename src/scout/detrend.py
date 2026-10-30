"""Remove stellar variability and residual systematics with wotan.

The window is tied to the longest transit we search for (window ≈ 3× max
duration, design §4) so the filter cannot eat the transits it is meant to keep.

That window cannot follow spot modulation on fast rotators (P_rot of a few
days or less), which then leaves dips and bumps that TLS happily fits. So when
Lomb-Scargle finds a strong short rotation period, the rotation signal is
fitted first (a few frequencies with harmonics, amplitudes free per orbit,
dips clipped so transits do not pull the fit) and divided out; the biweight
then removes what is left.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scout.config import DetrendConfig
from scout.lightcurve import LightCurve, bin_per_sector


def window_days(cfg: DetrendConfig) -> float:
    return cfg.window_factor * cfg.max_transit_duration_hours / 24.0


def _segments(time: np.ndarray, gap_days: float) -> list[np.ndarray]:
    """Index arrays of contiguous stretches (TESS orbits, in practice)."""
    if len(time) == 0:
        return []
    cuts = np.flatnonzero(np.diff(time) > gap_days) + 1
    return np.split(np.arange(len(time)), cuts)


def _design(t: np.ndarray, freqs: list[float], harmonics: int) -> np.ndarray:
    tc = t - t.mean()
    cols = [np.ones_like(t), tc]
    for f in freqs:
        for k in range(1, harmonics + 1):
            arg = 2 * np.pi * k * f * tc
            cols += [np.sin(arg), np.cos(arg)]
    return np.column_stack(cols)


def _robust_fit(t: np.ndarray, y: np.ndarray, freqs: list[float], harmonics: int) -> np.ndarray:
    """Least-squares fit, iteratively ignoring dips (transits) and flares."""
    a = _design(t, freqs, harmonics)
    keep = np.ones(len(t), dtype=bool)
    model = np.full_like(y, np.median(y))
    for _ in range(5):
        if keep.sum() <= a.shape[1] + 5:
            break
        coef, *_ = np.linalg.lstsq(a[keep], y[keep], rcond=None)
        model = a @ coef
        r = y - model
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        if not np.isfinite(s) or s == 0:
            break
        new = (r > -3 * s) & (r < 5 * s)
        if np.array_equal(new, keep):
            break
        keep = new
    return model


def _rotation_model(lc: LightCurve, cfg: DetrendConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Multiplicative rotation model (≈1) and what was found, or ones if none."""
    from astropy.timeseries import LombScargle

    info: dict[str, Any] = {"applied": False, "frequencies_per_day": [], "amplitudes": []}
    ones = np.ones(len(lc))
    if len(lc) < 200:
        return ones, info
    # Lomb-Scargle on a 30-min binned copy: fast, and transits barely register.
    b = bin_per_sector(lc, 30.0)
    t, y = b.time, b.flux.copy()
    fmin, fmax = 1.0 / cfg.rotation_max_period_days, 1.0 / 0.1
    segs = _segments(t, cfg.break_tolerance_days)
    freqs: list[float] = []
    resid = y.copy()
    for _ in range(cfg.rotation_max_terms):
        grid = np.linspace(fmin, fmax, 20000)
        power = LombScargle(t, resid).power(grid)
        f = float(grid[int(np.argmax(power))])
        # Semi-amplitude of that frequency, fitted per segment then averaged.
        amps = []
        for idx in segs:
            if len(idx) < 20:
                continue
            m = _robust_fit(t[idx], resid[idx], [f], 1)
            amps.append(0.5 * (np.percentile(m, 95) - np.percentile(m, 5)))
        amp = float(np.median(amps)) if amps else 0.0
        if amp < (cfg.rotation_term_min_amplitude if freqs else cfg.rotation_min_amplitude):
            break
        freqs.append(f)
        info["amplitudes"].append(amp)
        resid = y.copy()
        for idx in segs:
            if len(idx) > 2 * len(freqs) * cfg.rotation_harmonics + 10:
                resid[idx] = y[idx] - _robust_fit(t[idx], y[idx], freqs, cfg.rotation_harmonics) + 1
    if not freqs:
        return ones, info

    model = np.ones(len(lc))
    for idx in _segments(lc.time, cfg.break_tolerance_days):
        if len(idx) > 2 * len(freqs) * cfg.rotation_harmonics + 10:
            model[idx] = _robust_fit(lc.time[idx], lc.flux[idx], freqs, cfg.rotation_harmonics)
    model = np.where(np.isfinite(model) & (model > 0), model, 1.0)
    info.update(applied=True, frequencies_per_day=freqs)
    info["periods_days"] = [1.0 / f for f in freqs]
    return model, info


def detrend(lc: LightCurve, cfg: DetrendConfig) -> LightCurve:
    """Return a flattened copy of ``lc`` with the removed trend in ``.trend``."""
    from wotan import flatten

    rot_info: dict[str, Any] = {"applied": False}
    rot = np.ones(len(lc))
    if cfg.rotation_prewhiten:
        rot, rot_info = _rotation_model(lc, cfg)

    window = window_days(cfg)
    flat, trend = flatten(
        lc.time,
        lc.flux / rot,
        method=cfg.method,
        window_length=window,
        edge_cutoff=cfg.edge_cutoff_days,
        break_tolerance=cfg.break_tolerance_days,
        return_trend=True,
    )
    out = lc.select(np.ones(len(lc), dtype=bool))
    out.flux = np.asarray(flat, dtype=float)
    out.trend = np.asarray(trend, dtype=float) * rot
    out.flux_err = lc.flux_err / np.where(out.trend > 0, out.trend, np.nan)

    keep = np.isfinite(out.flux) & np.isfinite(out.flux_err)
    n_clipped = 0
    if cfg.upper_sigma_clip > 0:
        resid = out.flux[keep] - 1.0
        mad_sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        upper = out.flux > 1.0 + cfg.upper_sigma_clip * mad_sigma
        n_clipped = int(np.sum(upper & keep))
        keep &= ~upper
    out = out.select(keep)
    out.meta["detrend"] = {
        "method": cfg.method,
        "window_days": window,
        "edge_cutoff_days": cfg.edge_cutoff_days,
        "break_tolerance_days": cfg.break_tolerance_days,
        "upper_sigma_clip": cfg.upper_sigma_clip,
        "n_clipped_upper": n_clipped,
        "n_dropped_nonfinite": int(len(lc) - len(out) - n_clipped),
        "rotation_prewhiten": rot_info,
    }
    return out
