"""Remove stellar variability and residual systematics with wotan.

The window is tied to the longest transit we search for (window ≈ 3× max
duration, design §4) so the filter cannot eat the transits it is meant to keep.
"""

from __future__ import annotations

import numpy as np

from scout.config import DetrendConfig
from scout.lightcurve import LightCurve


def window_days(cfg: DetrendConfig) -> float:
    return cfg.window_factor * cfg.max_transit_duration_hours / 24.0


def detrend(lc: LightCurve, cfg: DetrendConfig) -> LightCurve:
    """Return a flattened copy of ``lc`` with the removed trend in ``.trend``."""
    from wotan import flatten

    window = window_days(cfg)
    flat, trend = flatten(
        lc.time,
        lc.flux,
        method=cfg.method,
        window_length=window,
        edge_cutoff=cfg.edge_cutoff_days,
        break_tolerance=cfg.break_tolerance_days,
        return_trend=True,
    )
    out = lc.select(np.ones(len(lc), dtype=bool))
    out.flux = np.asarray(flat, dtype=float)
    out.trend = np.asarray(trend, dtype=float)
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
    }
    return out
