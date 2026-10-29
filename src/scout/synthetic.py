"""Synthetic TESS-like light curves for tests, CI and injection-recovery.

The generator mimics the things that trip up a real search: per-sector flux
offsets, a mid-sector orbit gap, momentum dumps with flagged cadences and small
flux jumps, red noise from stellar rotation, and white noise. Transits are
drawn with batman (limb-darkened) when it is installed, otherwise with a
trapezoid. Everything is seeded, so a given call is reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from scout.lightcurve import LightCurve

SECTOR_DAYS = 27.4
SECTOR1_START_BTJD = 1325.3
ORBIT_GAP_DAYS = 1.0
MOMENTUM_DUMP_BIT = 32


@dataclass(frozen=True)
class Transit:
    """A transit signal. ``period_days=None`` means a single event at ``t0``."""

    t0_btjd: float
    depth: float  # fractional, e.g. 1e-3 = 1000 ppm
    duration_hours: float
    period_days: float | None = None
    impact: float = 0.3
    limb_darkening: tuple[float, float] = (0.4, 0.25)  # quadratic u1, u2


@dataclass(frozen=True)
class SyntheticStar:
    tic: int = 999_000_001
    sectors: tuple[int, ...] = (14, 15, 16)
    cadence_s: int = 120
    noise_ppm: float = 500.0
    # Quasi-periodic rotational modulation.
    rotation_period_days: float = 0.0  # 0 disables variability
    rotation_amplitude: float = 0.0
    momentum_dump_interval_days: float = 3.125
    momentum_dump_jump_ppm: float = 100.0
    sector_offset_scatter: float = 0.02  # fractional spread of per-sector medians
    transits: tuple[Transit, ...] = field(default_factory=tuple)
    # Stellar parameters, handed to the search as if they came from the TIC.
    radius_rsun: float = 1.0
    mass_msun: float = 1.0
    seed: int = 0


def sector_window(sector: int) -> tuple[float, float]:
    start = SECTOR1_START_BTJD + (sector - 1) * SECTOR_DAYS
    return start, start + SECTOR_DAYS


def transit_model(time: np.ndarray, transit: Transit) -> np.ndarray:
    """Relative flux (1 out of transit) for one transit signal."""
    try:
        return _batman_model(time, transit)
    except ImportError:
        return trapezoid_model(time, transit)


def trapezoid_model(time: np.ndarray, transit: Transit) -> np.ndarray:
    dur = transit.duration_hours / 24.0
    k = np.sqrt(transit.depth)
    b = min(transit.impact, 1.0 + k - 1e-6)
    # Ingress fraction of the total duration, from the chord geometry.
    outer = np.sqrt(max((1 + k) ** 2 - b**2, 1e-12))
    inner = np.sqrt(max((1 - k) ** 2 - b**2, 0.0))
    ingress = 0.5 * dur * (1.0 - inner / outer)
    dt = _phase_offset(time, transit)
    x = np.abs(dt)
    model = np.ones_like(time)
    full = x <= dur / 2 - ingress
    model[full] -= transit.depth
    ramp = (x > dur / 2 - ingress) & (x < dur / 2)
    if ingress > 0:
        model[ramp] -= transit.depth * (dur / 2 - x[ramp]) / ingress
    return model


def _phase_offset(time: np.ndarray, transit: Transit) -> np.ndarray:
    if transit.period_days is None:
        return time - transit.t0_btjd
    p = transit.period_days
    return (time - transit.t0_btjd + 0.5 * p) % p - 0.5 * p


def _batman_model(time: np.ndarray, transit: Transit) -> np.ndarray:
    import batman  # optional dependency

    k = np.sqrt(transit.depth)
    b = transit.impact
    # Orbital period only sets a/R* here; a single event uses a nominal long period.
    period = transit.period_days if transit.period_days is not None else 1000.0
    dur = transit.duration_hours / 24.0
    chord = np.sqrt(max((1 + k) ** 2 - b**2, 1e-6))
    a_rs = max(period / (np.pi * dur) * chord, 1.5)
    inc = np.degrees(np.arccos(np.clip(b / a_rs, 0, 1)))

    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = period
    params.rp = k
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = list(transit.limb_darkening)

    dt = _phase_offset(time, transit)
    model = np.ones_like(time)
    near = np.abs(dt) < dur  # only evaluate near transit; batman is per-point anyway
    if near.any():
        m = batman.TransitModel(params, dt[near])
        model[near] = m.light_curve(params)
    if transit.period_days is None:
        model[np.abs(dt) > 0.5 * period] = 1.0
    return model


def generate(star: SyntheticStar) -> LightCurve:
    """Build a raw (un-normalised, un-detrended) multi-sector light curve."""
    rng = np.random.default_rng(star.seed)
    step = star.cadence_s / 86400.0
    parts: list[LightCurve] = []
    phase0 = rng.uniform(0, 2 * np.pi, size=3)

    for sector in star.sectors:
        start, end = sector_window(sector)
        t = np.arange(start, end, step)
        mid = 0.5 * (start + end)
        t = t[np.abs(t - mid) > ORBIT_GAP_DAYS / 2]
        n = len(t)
        quality = np.zeros(n, dtype=np.int32)

        flux = np.ones(n)
        if star.rotation_period_days > 0 and star.rotation_amplitude > 0:
            w = 2 * np.pi / star.rotation_period_days
            flux += star.rotation_amplitude * (
                np.sin(w * t + phase0[0])
                + 0.3 * np.sin(2 * w * t + phase0[1])
                + 0.15 * np.sin(0.5 * w * t + phase0[2])
            )
        for tr in star.transits:
            flux *= transit_model(t, tr)

        if star.momentum_dump_interval_days > 0:
            jump = star.momentum_dump_jump_ppm * 1e-6
            for orbit_start, orbit_end in ((start, mid), (mid, end)):
                dumps = np.arange(
                    orbit_start + star.momentum_dump_interval_days,
                    orbit_end,
                    star.momentum_dump_interval_days,
                )
                for td in dumps:
                    idx = np.argmin(np.abs(t - td))
                    if abs(t[idx] - td) < step:
                        quality[idx] |= MOMENTUM_DUMP_BIT
                    flux[(t > td) & (t < orbit_end)] += rng.normal(0, jump)

        sigma = star.noise_ppm * 1e-6
        flux += rng.normal(0, sigma, n)
        scale = 1.0 + rng.normal(0, star.sector_offset_scatter)
        # SPOC-like electrons/s level, so normalisation is actually exercised.
        level = 1.0e4 * scale
        parts.append(
            LightCurve(
                time=t,
                flux=flux * level,
                flux_err=np.full(n, sigma * level),
                quality=quality,
                sector=np.full(n, sector),
                cadence_s=np.full(n, star.cadence_s),
            )
        )

    meta = {
        "tic": star.tic,
        "source": "synthetic",
        "star": {"tic": star.tic, "radius_rsun": star.radius_rsun, "mass_msun": star.mass_msun},
        "synthetic": _star_meta(star),
    }
    return LightCurve.concatenate(parts, meta=meta)


def _star_meta(star: SyntheticStar) -> dict:
    d = asdict(star)
    d["sectors"] = list(star.sectors)
    d["transits"] = [asdict(t) for t in star.transits]
    return d


def preset(
    name: str, tic: int = 999_000_001, sectors: tuple[int, ...] = (14, 15, 16), seed: int = 0
) -> SyntheticStar:
    """Ready-made test stars. Transit epochs are placed inside the first sector."""
    t_first = sector_window(sectors[0])[0]
    common = {"tic": tic, "sectors": sectors, "seed": seed}
    presets = {
        # Flat star, no signal: the search must come back empty.
        "quiet": SyntheticStar(noise_ppm=400.0, **common),
        "hot-jupiter": SyntheticStar(
            noise_ppm=800.0,
            rotation_period_days=9.0,
            rotation_amplitude=0.003,
            transits=(
                Transit(
                    t0_btjd=t_first + 1.7,
                    period_days=3.21,
                    depth=8e-3,
                    duration_hours=2.6,
                    impact=0.4,
                ),
            ),
            **common,
        ),
        "sub-neptune": SyntheticStar(
            noise_ppm=500.0,
            rotation_period_days=12.0,
            rotation_amplitude=0.002,
            transits=(
                Transit(t0_btjd=t_first + 2.3, period_days=7.43, depth=9e-4, duration_hours=3.4),
            ),
            **common,
        ),
        # M dwarf: shallow, short transits under spotty variability.
        "m-dwarf": SyntheticStar(
            noise_ppm=1500.0,
            rotation_period_days=6.0,
            rotation_amplitude=0.006,
            radius_rsun=0.4,
            mass_msun=0.4,
            transits=(
                Transit(
                    t0_btjd=t_first + 0.9,
                    period_days=4.62,
                    depth=3.5e-3,
                    duration_hours=1.4,
                    impact=0.2,
                    limb_darkening=(0.3, 0.35),
                ),
            ),
            **common,
        ),
    }
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; choose from {', '.join(sorted(presets))}")
    return presets[name]


PRESETS = ("quiet", "hot-jupiter", "sub-neptune", "m-dwarf")
