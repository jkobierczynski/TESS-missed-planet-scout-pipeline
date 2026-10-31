"""Light-curve vetting checks (design §6, items 1–4 and 7–9).

Every check takes the flattened light curve and one ephemeris and returns a
:class:`Check`: a status (``pass``, ``warn``, ``fail`` or ``n/a``), the
numbers behind it, and one line a reviewer can read. Nothing here downloads
anything, so all of it is testable on synthetic light curves.

Depth errors are never just white noise: wherever there are enough events,
the scatter of per-event depths sets a floor, so red noise and spot crossings
cannot make a weak signal look precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scout.config import VetConfig

G_CGS = 6.674e-8
RHO_SUN = 1.408  # g cm^-3
R_EARTH_PER_R_SUN = 109.08


@dataclass
class Ephemeris:
    period_days: float
    t0_btjd: float
    duration_hours: float
    depth_ppm: float

    @property
    def dur_days(self) -> float:
        return self.duration_hours / 24.0


@dataclass
class Check:
    item: int  # number in the design §6 checklist
    name: str
    status: str  # "pass" | "warn" | "fail" | "n/a"
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "values": {k: _plain(v) for k, v in self.values.items()},
        }


def _plain(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


# --- folding and depths -------------------------------------------------------------


def phase_days(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from the nearest mid-transit, in [-P/2, P/2)."""
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def epochs(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    return np.round((t - t0) / period).astype(int)


def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@dataclass
class Depth:
    depth: float  # fractional
    err: float
    n_points: int

    @property
    def snr(self) -> float:
        return self.depth / self.err if self.err > 0 else float("nan")


def box_depth(
    flux: np.ndarray, in_mask: np.ndarray, out_mask: np.ndarray, sigma: float | None = None
) -> Depth:
    """Mean out-of-window flux minus mean in-window flux, with a white-noise error."""
    n_in, n_out = int(in_mask.sum()), int(out_mask.sum())
    if n_in < 3 or n_out < 3:
        return Depth(float("nan"), float("nan"), n_in)
    base = float(np.median(flux[out_mask]))
    s = robust_std(flux[out_mask]) if sigma is None else sigma
    d = base - float(np.mean(flux[in_mask]))
    return Depth(d, s / np.sqrt(n_in), n_in)


@dataclass
class Folded:
    """The light curve with its ephemeris applied, shared by every check."""

    time: np.ndarray
    flux: np.ndarray
    sector: np.ndarray
    eph: Ephemeris
    phase: np.ndarray  # days from mid-transit
    epoch: np.ndarray
    in_transit: np.ndarray  # inner 80 % of the transit
    near: np.ndarray  # out of transit, within the local baseline window
    sigma: float  # robust scatter per point out of transit
    # Scatter of transit-length averages out of transit: the noise one transit
    # really has once red noise is counted (Pont et al. 2006).
    sigma_box: float
    points_per_event: float
    # Per sector (sigma, sigma_box): noise differs a lot between sectors.
    sector_noise: dict[int, tuple[float, float]] = field(default_factory=dict)

    def err_for(self, n_points: int, n_events: float) -> float:
        white = self.sigma / np.sqrt(max(n_points, 1))
        red = self.sigma_box / np.sqrt(max(n_events, 1.0))
        return float(np.nanmax([white, red]))

    @classmethod
    def build(
        cls, time: np.ndarray, flux: np.ndarray, sector: np.ndarray, eph: Ephemeris, cfg: VetConfig
    ) -> Folded:
        good = np.isfinite(time) & np.isfinite(flux)
        time, flux, sector = time[good], flux[good], sector[good]
        ph = phase_days(time, eph.period_days, eph.t0_btjd)
        half = 0.5 * eph.dur_days
        in_tr = np.abs(ph) < 0.8 * half
        # Local baseline: from one to `baseline_durations` durations off mid-transit,
        # capped at a quarter period so short periods still have one.
        outer = min(cfg.baseline_durations * eph.dur_days, 0.25 * eph.period_days)
        near = (np.abs(ph) > 1.0 * eph.dur_days) & (np.abs(ph) < max(outer, 1.5 * eph.dur_days))
        oot = ~(np.abs(ph) < 1.2 * half)
        sigma = robust_std(flux[oot])
        ep = epochs(time, eph.period_days, eph.t0_btjd)
        n_ev = max(len(np.unique(ep[in_tr])), 1)
        noise = {}
        for sec in np.unique(sector):
            m = oot & (sector == sec)
            noise[int(sec)] = (
                robust_std(flux[m]),
                _box_scatter(time[m], flux[m], 0.8 * eph.dur_days),
            )
        return cls(
            time,
            flux,
            sector,
            eph,
            ph,
            ep,
            in_tr,
            near,
            sigma,
            _box_scatter(time[oot], flux[oot], 0.8 * eph.dur_days),
            max(int(in_tr.sum()) / n_ev, 1.0),
            noise,
        )

    def event_depths(self) -> dict[int, Depth]:
        """Depth of each individual transit against its own local baseline."""
        out: dict[int, Depth] = {}
        for e in np.unique(self.epoch[self.in_transit]):
            m = self.epoch == e
            sec = int(np.bincount(self.sector[m & self.in_transit].astype(int)).argmax())
            sig, box = self.sector_noise.get(sec, (self.sigma, self.sigma_box))
            d = box_depth(self.flux, m & self.in_transit, m & self.near, sig)
            if np.isfinite(d.depth):
                err = float(np.nanmax([sig / np.sqrt(max(d.n_points, 1)), box]))
                out[int(e)] = Depth(d.depth, err, d.n_points)
        return out

    def coverage(self) -> dict[int, bool]:
        """Whether each transit has data on both sides: before ingress and after egress."""
        out = {}
        half = 0.5 * self.eph.dur_days
        for e in np.unique(self.epoch[self.in_transit]):
            m = self.epoch == e
            ph = self.phase[m]
            before = np.any((ph < -half) & (ph > -half - self.eph.dur_days))
            after = np.any((ph > half) & (ph < half + self.eph.dur_days))
            out[int(e)] = bool(before and after)
        return out

    def depth_of(self, select: np.ndarray, events: dict[int, Depth] | None = None) -> Depth:
        """Depth of the transits picked by ``select`` (a cadence mask).

        The error is the larger of the white-noise error and the scatter of
        those transits' individual depths over sqrt(n) (red-noise floor).
        """
        d = box_depth(self.flux, select & self.in_transit, select & self.near, self.sigma)
        if not np.isfinite(d.depth):
            return d
        n_ev = len(np.unique(self.epoch[select & self.in_transit]))
        d = Depth(d.depth, self.err_for(d.n_points, n_ev), d.n_points)
        if events is None:
            return d
        picked = [events[e] for e in np.unique(self.epoch[select & self.in_transit]) if e in events]
        errs = np.array([p.err for p in picked if np.isfinite(p.err) and p.err > 0])
        if len(errs):
            # The transits' own (per-sector) errors combined.
            comb = float(1 / np.sqrt(np.sum(1 / errs**2)))
            if comb > d.err:
                d = Depth(d.depth, comb, d.n_points)
        if len(picked) >= 4:
            vals = np.array([p.depth for p in picked])
            floor = robust_std(vals) / np.sqrt(len(vals))
            if np.isfinite(floor) and floor > d.err:
                d = Depth(d.depth, floor, d.n_points)
        return d


def _box_scatter(t: np.ndarray, y: np.ndarray, width: float) -> float:
    """Robust scatter of means over consecutive windows of ``width`` days."""
    if len(t) < 10 or width <= 0:
        return float("nan")
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    _, inv, n = np.unique(idx, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, y)
    full = n >= max(3, 0.5 * np.median(n))
    return robust_std(sums[full] / n[full])


def _chi2_consistency(depths: list[Depth]) -> tuple[float, int, float]:
    """chi^2 of the depths about their weighted mean, its dof and p-value."""
    from scipy.stats import chi2 as chi2_dist

    v = np.array([d.depth for d in depths])
    e = np.array([d.err for d in depths])
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if len(v) < 2:
        return float("nan"), 0, float("nan")
    w = 1 / e**2
    mean = np.sum(w * v) / np.sum(w)
    c2 = float(np.sum(((v - mean) / e) ** 2))
    dof = len(v) - 1
    return c2, dof, float(chi2_dist.sf(c2, dof))


# --- 1. shape: transit model fit ----------------------------------------------------


def fit_transit(f: Folded, radius_rsun: float | None, mass_msun: float | None) -> dict[str, Any]:
    """Least-squares limb-darkened transit fit on the fold (P fixed).

    Free: mid-time shift, Rp/R*, a/R*, b. Returns the parameters, the
    implied stellar density and the chi^2 gain over a flat line.
    """
    from scipy.optimize import least_squares

    eph = f.eph
    win = np.abs(f.phase) < 2.5 * eph.dur_days
    x, y = f.phase[win], f.flux[win]
    if win.sum() < 20 or not np.isfinite(f.sigma) or f.sigma <= 0:
        return {"ok": False}
    # Bin the fold to ~1/20 of the duration: the fit only needs the shape.
    width = eph.dur_days / 20
    idx = np.floor((x + 2.5 * eph.dur_days) / width).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    xb = (np.bincount(idx, x)[keep]) / counts[keep]
    yb = (np.bincount(idx, y)[keep]) / counts[keep]
    eb = f.sigma / np.sqrt(counts[keep])
    base = float(np.median(y[np.abs(x) > eph.dur_days]))

    k0 = np.sqrt(max(eph.depth_ppm, 10.0) * 1e-6)
    # a/R* from the duration for b = 0
    a0 = max(eph.period_days / (np.pi * eph.dur_days) * (1 + k0), 1.6)

    def model(p: np.ndarray, xx: np.ndarray) -> np.ndarray:
        dt0, k, a, b = p
        return base * _batman(xx - dt0, eph.period_days, k, a, b)

    def resid(p: np.ndarray) -> np.ndarray:
        return (yb - model(p, xb)) / eb

    lo = [-0.5 * eph.dur_days, 1e-3, 1.5, 0.0]
    hi = [0.5 * eph.dur_days, 0.9, max(a0 * 30, 5.0), 1.9]
    best = None
    for b0 in (0.1, 0.5, 0.85):
        p0 = np.clip([0.0, k0, a0 / np.sqrt(max(1 - b0**2, 0.05)), b0], lo, hi)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return {"ok": False}
    dt0, k, a, b = best.x
    chi2_model = float(2 * best.cost)
    chi2_flat = float(np.sum(((yb - base) / eb) ** 2))
    p_s = eph.period_days * 86400.0
    rho_fit = 3 * np.pi * a**3 / (G_CGS * p_s**2)
    rho_tic = (
        RHO_SUN * mass_msun / radius_rsun**3
        if radius_rsun and mass_msun and radius_rsun > 0 and mass_msun > 0
        else float("nan")
    )
    inc = np.arccos(min(b / a, 1.0))
    t14 = _t14(eph.period_days, k, a, b)
    t23 = _t14(eph.period_days, -k, a, b)
    return {
        "ok": True,
        "period_days": eph.period_days,
        "dt0_days": float(dt0),
        "rp_rs": float(k),
        "a_rs": float(a),
        "b": float(b),
        "inc_deg": float(np.degrees(inc)),
        "t14_hours": t14 * 24,
        "t23_hours": t23 * 24,
        "flat_fraction": t23 / t14 if t14 > 0 else 0.0,
        "grazing": bool(b + k > 1.0),
        "rho_fit_cgs": float(rho_fit),
        "rho_tic_cgs": float(rho_tic),
        "rp_rearth": float(k * radius_rsun * R_EARTH_PER_R_SUN) if radius_rsun else float("nan"),
        "delta_chi2": chi2_flat - chi2_model,
        "base": base,
        "x_binned": xb,
        "y_binned": yb,
    }


def _t14(period: float, k: float, a: float, b: float) -> float:
    """Duration between contacts; k < 0 gives T23 (0 when grazing)."""
    arg = (1 + k) ** 2 - b**2
    if arg <= 0 or a <= 1:
        return 0.0
    sin_i = np.sqrt(max(1 - (b / a) ** 2, 1e-12))
    x = np.clip(np.sqrt(arg) / (a * sin_i), 0, 1)
    return float(period / np.pi * np.arcsin(x))


def _batman(dt: np.ndarray, period: float, k: float, a: float, b: float) -> np.ndarray:
    import batman

    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = period
    params.rp = k
    params.a = a
    params.inc = float(np.degrees(np.arccos(min(b / a, 1.0))))
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [0.4, 0.25]
    return batman.TransitModel(params, np.asarray(dt, dtype=float)).light_curve(params)


def transit_model_curve(fit: dict[str, Any], eph: Ephemeris, x: np.ndarray) -> np.ndarray:
    return fit["base"] * _batman(
        x - fit["dt0_days"], eph.period_days, fit["rp_rs"], fit["a_rs"], fit["b"]
    )


def check_shape(fit: dict[str, Any]) -> Check:
    if not fit.get("ok"):
        return Check(1, "Transit shape", "n/a", "transit fit failed")
    v = {k: fit[k] for k in ("rp_rs", "a_rs", "b", "t14_hours", "flat_fraction", "grazing")}
    if fit["grazing"]:
        return Check(
            1,
            "Transit shape",
            "warn",
            f"V-shaped: best fit is grazing (b = {fit['b']:.2f}, Rp/R* = {fit['rp_rs']:.3f}); "
            "an EB until shown otherwise",
            v,
        )
    return Check(
        1,
        "Transit shape",
        "pass",
        f"U-shaped: b = {fit['b']:.2f}, flat bottom {fit['flat_fraction']:.0%} of T14",
        v,
    )


# --- 1b. variability: are the dips just one side of a wave? ------------------------------


def _scan_boxes(
    f: Folded, centers: np.ndarray, width: float, sign: float
) -> tuple[np.ndarray, np.ndarray]:
    """Depth and SNR of a box of ``width`` days at each phase centre (in days).

    ``sign=+1`` measures dips, ``-1`` bumps. Uses the folded, all-epoch light
    curve with the global out-of-transit level as baseline.
    """
    p = f.eph.period_days
    order = np.argsort(f.phase)
    ph, fl = f.phase[order], f.flux[order]
    base = float(np.median(fl[np.abs(ph) > f.eph.dur_days]))
    # Wrap-around padding so boxes near ±P/2 are complete.
    ph = np.concatenate([ph - p, ph, ph + p])
    fl = np.concatenate([fl, fl, fl])
    cs = np.concatenate([[0.0], np.cumsum(fl)])
    lo = np.searchsorted(ph, centers - 0.5 * width)
    hi = np.searchsorted(ph, centers + 0.5 * width)
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cs[hi] - cs[lo]) / n
        depth = sign * (base - mean)
        white = f.sigma / np.sqrt(n)
        red = f.sigma_box / np.sqrt(np.maximum(n / f.points_per_event, 1.0))
        snr = depth / np.fmax(white, red)
    depth[n < 5] = np.nan
    snr[n < 5] = np.nan
    return depth, snr


def check_variability(f: Folded, cfg: VetConfig, repeats: list[float] | None = None) -> Check:
    """Bumps as strong as the dips, or a sinusoid that explains the dip, mean variability.

    ``repeats``: phase offsets, in durations, at which the search found this
    period again after masking the transits. Beyond one duration the dimming
    is not confined to the transits.
    """
    eph = f.eph
    w = 0.8 * eph.dur_days
    step = eph.dur_days / 4
    centers = np.arange(-0.5 * eph.period_days, 0.5 * eph.period_days, step)
    dip = f.depth_of(np.ones(len(f.time), bool), f.event_depths())
    _, bump_snr = _scan_boxes(f, centers, w, -1.0)
    off = np.abs(centers) > eph.dur_days
    max_bump = float(np.nanmax(bump_snr[off])) if np.isfinite(bump_snr[off]).any() else 0.0
    # Sinusoid at P and P/2 against the box model, both on the same cadences.
    sin_gain = _sine_gain(f)
    box_gain = (dip.depth / (f.sigma / np.sqrt(max(dip.n_points, 1)))) ** 2 if dip.n_points else 0.0
    # Both gains are white-noise chi^2, so their ratio is fair even when red noise dominates.
    v = {"dip_snr": dip.snr, "max_bump_snr": max_bump, "repeats_durations": repeats or []}
    if not dip.snr > 0:
        return Check(1, "Not variability", "fail", "no dip at this ephemeris once re-measured", v)
    off_phase = [r for r in repeats or [] if abs(r) > 1.0]
    if off_phase:
        return Check(
            1,
            "Not variability",
            "fail",
            f"the search finds this period again {len(off_phase)} time(s) after masking the "
            f"transits, offset by {', '.join(f'{r:+.1f}' for r in off_phase)} durations: "
            "the dimming is not confined to the transits",
            v,
        )
    ratio_bump = max_bump / dip.snr
    ratio_sine = sin_gain / box_gain if box_gain > 0 else float("inf")
    v.update(bump_ratio=ratio_bump, sine_ratio=ratio_sine)
    if ratio_bump >= cfg.max_bump_ratio or ratio_sine >= cfg.max_sine_ratio:
        why = []
        if ratio_bump >= cfg.max_bump_ratio:
            why.append(f"brightenings at this period reach {ratio_bump:.0%} of the dip SNR")
        if ratio_sine >= cfg.max_sine_ratio:
            why.append(f"a sinusoid at P or P/2 explains {ratio_sine:.0%} of the dip")
        return Check(1, "Not variability", "fail", "; ".join(why), v)
    return Check(
        1,
        "Not variability",
        "pass",
        f"strongest brightening {ratio_bump:.0%} of dip SNR; sinusoid explains {ratio_sine:.0%}",
        v,
    )


def _sine_gain(f: Folded) -> float:
    """chi^2 gain of the best sinusoid at P or P/2 over a constant."""
    y = f.flux - np.median(f.flux)
    best = 0.0
    for div in (1, 2):
        ang = 2 * np.pi * f.phase / (f.eph.period_days / div)
        a = np.column_stack([np.ones_like(ang), np.cos(ang), np.sin(ang)])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        resid = y - a @ coef
        gain = (np.sum((y - y.mean()) ** 2) - np.sum(resid**2)) / f.sigma**2
        best = max(best, float(gain))
    return best


# --- 2. odd/even -----------------------------------------------------------------------


def check_odd_even(f: Folded, events: dict[int, Depth], cfg: VetConfig) -> Check:
    odd = f.depth_of(f.epoch % 2 == 1, events)
    even = f.depth_of(f.epoch % 2 == 0, events)
    v = {
        "odd_ppm": odd.depth * 1e6,
        "odd_err_ppm": odd.err * 1e6,
        "even_ppm": even.depth * 1e6,
        "even_err_ppm": even.err * 1e6,
    }
    if not (np.isfinite(odd.depth) and np.isfinite(even.depth)):
        return Check(2, "Odd/even depths", "n/a", "needs at least one odd and one even transit", v)
    sig = abs(odd.depth - even.depth) / np.hypot(odd.err, even.err)
    v["sigma"] = sig
    text = (
        f"odd {odd.depth * 1e6:.0f} ± {odd.err * 1e6:.0f} ppm, "
        f"even {even.depth * 1e6:.0f} ± {even.err * 1e6:.0f} ppm ({sig:.1f}σ)"
    )
    status = (
        "fail" if sig >= cfg.odd_even_fail_sigma else "warn" if sig >= cfg.warn_sigma else "pass"
    )
    return Check(2, "Odd/even depths", status, text, v)


# --- 3. secondary eclipse ------------------------------------------------------------------


def check_secondary(f: Folded, cfg: VetConfig) -> Check:
    """Deepest dip at phase 0.5, and anywhere else away from the transit.

    Phase 0.5 fails at ``secondary_sigma``. Elsewhere (eccentric orbits) the
    look-elsewhere effect raises the bar to ``secondary_sigma_any_phase``,
    and with few transits the "orbit" is most of the light curve, so the
    deepest dip anywhere is just its worst systematic: it can only warn.
    """
    eph = f.eph
    p = eph.period_days
    w = 0.8 * eph.dur_days
    centers = np.arange(-0.5 * p, 0.5 * p, eph.dur_days / 4)
    dist = np.abs(phase_days(centers, p, 0.0))
    away = dist > 1.5 * eph.dur_days
    if not away.any():
        return Check(3, "No secondary eclipse", "n/a", "period too short to look for one")
    depth, snr = _scan_boxes(f, centers, w, +1.0)
    phases = (centers / p) % 1.0
    half = away & (np.abs(phases - 0.5) * p < eph.dur_days)
    other = away & ~half
    sparse = len(np.unique(f.epoch[f.in_transit])) <= cfg.sparse_events

    def best(mask: np.ndarray) -> int | None:
        s = np.where(mask, snr, np.nan)
        return int(np.nanargmax(s)) if np.isfinite(s).any() else None

    ih, io = best(half), best(other)
    if ih is None and io is None:
        return Check(3, "No secondary eclipse", "n/a", "no phase coverage away from transit")

    def describe(i: int) -> str:
        return f"{depth[i] * 1e6:.0f} ppm at phase {phases[i]:.3f} ({snr[i]:.1f}σ)"

    def ratio(i: int) -> float:
        return depth[i] * 1e6 / eph.depth_ppm if eph.depth_ppm else float("nan")

    v: dict[str, Any] = {}
    parts = []
    status = "pass"
    if ih is not None:
        v.update(
            phase=phases[ih], depth_ppm=depth[ih] * 1e6, snr=snr[ih], ratio_to_primary=ratio(ih)
        )
        parts.append("at phase 0.5: " + describe(ih))
        if snr[ih] >= cfg.secondary_sigma:
            status = "fail" if ratio(ih) >= cfg.occultation_max_ratio else "warn"
            if status == "warn":
                parts[-1] += ", small enough to be an occultation"
        elif snr[ih] >= cfg.warn_sigma:
            status = "warn"
    if io is not None:
        v.update(
            other_phase=phases[io],
            other_depth_ppm=depth[io] * 1e6,
            other_snr=snr[io],
            other_ratio_to_primary=ratio(io),
        )
        if ih is None or snr[io] > snr[ih]:
            v.update(phase=phases[io])
        parts.append("elsewhere: " + describe(io))
        if snr[io] >= cfg.secondary_sigma_any_phase:
            if sparse or ratio(io) < cfg.occultation_max_ratio:
                status = "warn" if status == "pass" else status
            else:
                status = "fail"
    return Check(3, "No secondary eclipse", status, "deepest dip " + "; ".join(parts), v)


# --- 4. depth consistency ------------------------------------------------------------------


def check_depth_consistency(
    f: Folded, events: dict[int, Depth], sources: dict[int, str], cfg: VetConfig
) -> Check:
    per_sector = {}
    for s in np.unique(f.sector):
        d = f.depth_of(f.sector == s, events)
        if np.isfinite(d.depth):
            per_sector[int(s)] = d
    c2s, dofs, ps = _chi2_consistency(list(per_sector.values()))
    c2e, dofe, pe = (
        _chi2_consistency(list(events.values()))
        if len(events) <= 50
        else (float("nan"), 0, float("nan"))
    )
    by_source: dict[str, list[Depth]] = {}
    for s, d in per_sector.items():
        by_source.setdefault(sources.get(s, "?"), []).append(d)
    v: dict[str, Any] = {
        "per_sector": {s: [d.depth * 1e6, d.err * 1e6] for s, d in per_sector.items()},
        "sector_chi2": c2s,
        "sector_dof": dofs,
        "sector_p": ps,
        "event_chi2": c2e,
        "event_dof": dofe,
        "event_p": pe,
        "sources": sorted(by_source),
    }
    parts = []
    if dofs:
        parts.append(f"{dofs + 1} sectors χ²/dof = {c2s / dofs:.1f} (p = {ps:.2g})")
    if dofe:
        parts.append(f"{dofe + 1} transits χ²/dof = {c2e / dofe:.1f} (p = {pe:.2g})")
    if not parts:
        return Check(4, "Depth consistent", "n/a", "only one sector and one transit", v)
    pmin = np.nanmin([x for x in (ps, pe) if np.isfinite(x)])
    status = "fail" if pmin < cfg.depth_consistency_p else "warn" if pmin < 0.05 else "pass"
    if len(by_source) > 1:
        parts.append("sources: " + ", ".join(sorted(by_source)))
    return Check(4, "Depth consistent", status, "; ".join(parts), v)


# --- 7, 8: size and density ---------------------------------------------------------


def check_radius(fit: dict[str, Any], cfg: VetConfig) -> Check:
    if not fit.get("ok") or not np.isfinite(fit.get("rp_rearth", np.nan)):
        return Check(7, "Radius < 2 R_Jup", "n/a", "no fit or no TIC radius")
    rp = fit["rp_rearth"]
    v = {"rp_rearth": rp}
    if rp > cfg.max_radius_rearth:
        return Check(7, "Radius < 2 R_Jup", "fail", f"Rp = {rp:.1f} R⊕: too big for a planet", v)
    if fit["grazing"]:
        return Check(
            7, "Radius < 2 R_Jup", "warn", f"Rp = {rp:.1f} R⊕ but grazing, so poorly bound", v
        )
    return Check(7, "Radius < 2 R_Jup", "pass", f"Rp = {rp:.1f} R⊕", v)


def check_density(fit: dict[str, Any], cfg: VetConfig) -> Check:
    """Fitted T14 against the longest transit the TIC star allows (b = 0, circular).

    The fitted density itself is reported but not judged: at low SNR the
    b-a/R* degeneracy leaves it poorly bound, while T14 is measured directly.
    """
    if not fit.get("ok") or not np.isfinite(fit.get("rho_tic_cgs", np.nan)):
        return Check(8, "Duration fits the star", "n/a", "no fit or no TIC mass/radius")
    p_s = fit["period_days"] * 86400.0
    a_tic = (G_CGS * fit["rho_tic_cgs"] * p_s**2 / (3 * np.pi)) ** (1 / 3)
    t14_max = _t14(fit["period_days"], fit["rp_rs"], a_tic, 0.0) * 24
    ratio = fit["t14_hours"] / t14_max if t14_max > 0 else float("inf")
    v = {
        "t14_hours": fit["t14_hours"],
        "t14_max_hours": t14_max,
        "ratio": ratio,
        "rho_fit_cgs": fit["rho_fit_cgs"],
        "rho_tic_cgs": fit["rho_tic_cgs"],
    }
    text = (
        f"T14 {fit['t14_hours']:.1f} h vs at most {t14_max:.1f} h for this star "
        f"(fitted density {fit['rho_fit_cgs']:.2f} g/cm³, TIC {fit['rho_tic_cgs']:.2f})"
    )
    if ratio > cfg.max_duration_factor:
        return Check(8, "Duration fits the star", "fail", text + ": too long", v)
    if ratio > 1.15:
        return Check(8, "Duration fits the star", "warn", text, v)
    return Check(8, "Duration fits the star", "pass", text, v)


# --- 9. spacecraft events --------------------------------------------------------------


def observed_events(f: Folded) -> list[int]:
    return sorted({int(e) for e in np.unique(f.epoch[f.in_transit])})


def event_times(f: Folded) -> list[float]:
    e = observed_events(f)
    return [f.eph.t0_btjd + k * f.eph.period_days for k in e]


def check_spacecraft(
    f: Folded,
    other_events: list[tuple[int, float, float]],
    tic: int,
    gaps: list[tuple[float, float]],
    cfg: VetConfig,
) -> Check:
    """Transits on the same dates as other stars' dips, at data gaps, or only half seen.

    ``other_events``: (tic, time, duration_days) for every event of every
    sparse detection in the run (systematics hit many stars at once).
    ``gaps``: (start, end) of data gaps longer than ``cfg.gap_min_days``.
    Shared dates are only counted for sparse signals: a short-period signal
    lands on other stars' dates by chance.
    """
    times = event_times(f)
    n = len(times)
    if n == 0:
        return Check(9, "Not spacecraft events", "n/a", "no observed transits")
    cov = f.coverage()
    partial = [f.eph.t0_btjd + e * f.eph.period_days for e, ok in cov.items() if not ok]
    sparse = n <= cfg.sparse_events
    check_shared = n <= 2 * cfg.sparse_events
    ot = np.array([x[1] for x in other_events]) if other_events else np.zeros(0)
    od = np.array([x[2] for x in other_events]) if other_events else np.zeros(0)
    otic = np.array([x[0] for x in other_events]) if other_events else np.zeros(0, int)
    shared, near_gap = [], []
    for t in times:
        if check_shared:
            tol = 0.5 * (f.eph.dur_days + od) + cfg.shared_epoch_pad_days
            hit = (np.abs(ot - t) < tol) & (otic != tic)
            if len(np.unique(otic[hit])) >= cfg.shared_epoch_min_stars:
                shared.append(t)
        if any(a - f.eph.dur_days < t < b + f.eph.dur_days for a, b in gaps):
            near_gap.append(t)
    v = {
        "n_events": n,
        "shared_events": shared,
        "near_gap_events": near_gap,
        "partial_events": partial,
    }
    parts = []
    if check_shared:
        parts.append(f"{len(shared)}/{n} transits on dates other stars also dip")
    parts.append(f"{len(near_gap)}/{n} next to data gaps")
    parts.append(f"{len(partial)}/{n} without data on both sides")
    text = "; ".join(parts)
    full = n - len(partial)
    if sparse and (len(shared) / n >= 0.5 or len(near_gap) / n >= 0.5 or full < 2):
        return Check(9, "Not spacecraft events", "fail", text, v)
    if shared or (sparse and (near_gap or partial)):
        return Check(9, "Not spacecraft events", "warn", text, v)
    return Check(9, "Not spacecraft events", "pass", text, v)


def data_gaps(time: np.ndarray, min_days: float) -> list[tuple[float, float]]:
    t = np.sort(time[np.isfinite(time)])
    if len(t) < 2:
        return []
    d = np.diff(t)
    i = np.nonzero(d > min_days)[0]
    return [(float(t[k]), float(t[k + 1])) for k in i]


# --- event dominance (part of item 4) -----------------------------------------------------


def check_single_event_dominance(events: dict[int, Depth], cfg: VetConfig) -> Check:
    if len(events) < 2:
        return Check(4, "No single event dominates", "n/a", "fewer than two transits")
    snr2 = np.array([max(d.snr, 0.0) ** 2 for d in events.values()])
    share = float(snr2.max() / snr2.sum()) if snr2.sum() > 0 else 1.0
    weak = [e for e, d in events.items() if not d.snr >= cfg.min_event_snr]
    v = {"max_share": share, "event_snr": {e: d.snr for e, d in events.items()}}
    n = len(events)
    text = f"strongest transit carries {share:.0%} of the signal over {n} transits"
    if n <= cfg.sparse_events and weak:
        text += f"; {len(weak)} transit(s) below {cfg.min_event_snr:g}σ on their own"
    if n <= cfg.sparse_events and len(weak) >= n - 1:
        return Check(4, "No single event dominates", "fail", text, v)
    if share > cfg.max_event_share and n >= 3:
        return Check(4, "No single event dominates", "warn", text, v)
    return Check(4, "No single event dominates", "pass", text, v)
