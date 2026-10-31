"""Stitch per-sector light curves into one normalised multi-sector light curve.

Per sector: drop flagged cadences, mask (padded) momentum dumps, drop
non-finite points, then normalise to median 1.0 (design §4, §10). Cadence is
kept per row; sectors with different cadences are never resampled.
"""

from __future__ import annotations

import numpy as np

from scout.config import StitchConfig
from scout.lightcurve import LightCurve


def quality_mask(lc: LightCurve, cfg: StitchConfig) -> np.ndarray:
    """True for cadences to keep."""
    keep = (lc.quality & cfg.quality_bitmask) == 0
    keep &= np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
    if cfg.mask_momentum_dumps:
        dumps = lc.time[(lc.quality & cfg.momentum_dump_bit) != 0]
        pad = cfg.momentum_dump_pad_hours / 24.0
        if len(dumps) and pad > 0:
            # Distance from each cadence to its nearest dump.
            dumps = np.sort(dumps)
            idx = np.searchsorted(dumps, lc.time)
            left = dumps[np.clip(idx - 1, 0, len(dumps) - 1)]
            right = dumps[np.clip(idx, 0, len(dumps) - 1)]
            near = np.minimum(np.abs(lc.time - left), np.abs(lc.time - right))
            keep &= near > pad
        else:
            keep &= (lc.quality & cfg.momentum_dump_bit) == 0
    return keep


def normalise(lc: LightCurve) -> tuple[LightCurve, dict[int, float]]:
    """Divide each sector by its median flux.

    A sector whose median is not positive (it happens: background
    over-subtraction in PDCSAP for faint or blended stars) cannot be
    normalised and is dropped; the dropped sectors and their medians are
    returned so the stitch metadata records them.
    """
    out = lc.select(np.ones(len(lc), dtype=bool))
    dropped: dict[int, float] = {}
    for s in out.sectors:
        m = out.sector == s
        med = np.nanmedian(out.flux[m])
        if not np.isfinite(med) or med <= 0:
            dropped[int(s)] = float(med)
            continue
        out.flux[m] = out.flux[m] / med
        out.flux_err[m] = out.flux_err[m] / med
    if dropped:
        out = out.select(~np.isin(out.sector, list(dropped)))
    if len(out) == 0:
        raise ValueError(f"no sector could be normalised (medians {dropped})")
    return out, dropped


def stitch(parts: list[LightCurve] | LightCurve, cfg: StitchConfig) -> LightCurve:
    """Clean, normalise per sector and concatenate."""
    lc = parts if isinstance(parts, LightCurve) else LightCurve.concatenate(parts)
    n_in = len(lc)
    keep = quality_mask(lc, cfg)
    lc = lc.select(keep)
    if len(lc) == 0:
        raise ValueError("no cadences left after quality masking")
    lc, dropped = normalise(lc)
    lc.meta["stitch"] = {
        "sectors_dropped_bad_median": dropped,
        "n_in": n_in,
        "n_kept": len(lc),
        "sectors": lc.sectors,
        "cadences_s": sorted({int(c) for c in np.unique(lc.cadence_s)}),
        "quality_bitmask": cfg.quality_bitmask,
        "momentum_dump_pad_hours": cfg.momentum_dump_pad_hours if cfg.mask_momentum_dumps else None,
    }
    return lc
