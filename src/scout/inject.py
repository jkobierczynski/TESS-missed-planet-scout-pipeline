"""Injection-recovery and false-alarm tests (M2, design §3.2, §5, §8).

For each host star, synthetic limb-darkened transits are multiplied into the
*stitched* light curve (so detrending is part of what gets tested), then the
normal detrend and periodic search run on the result. A detection recovers the
injection when its period and epoch match.

Null samples measure the false-alarm side: the flattened light curve with each
sector time-rolled by a random amount (real periodic signals no longer line up
across sectors, the noise stays the same) is searched as is and inverted, and
the unscrambled inverted curve is searched too. Systematics that look like dips
also look like bumps, so the nulls see the same junk the real search sees.

``calibrate`` pools every host in a run and picks the SDE/SNR pair that
recovers the most injections while keeping the false-alarm rate per null
search at or below ``inject.false_alarm_target``.

Layout::

    results/<run>/inject/TIC<id>/
        inject.json          key of the config it was run with, host summary
        injections.jsonl     one line per injection (resumable)
        nulls.jsonl          one line per null search (resumable)
    results/<run>/calibration/
        calibration.json     chosen thresholds, recovery tables
        calibration.html     recovery map and null distributions
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config, InjectConfig, PeriodicSearchConfig, _to_plain
from scout.detrend import detrend
from scout.lightcurve import LightCurve
from scout.search.periodic import Detection, period_range, search_periodic
from scout.synthetic import Transit, transit_model

log = logging.getLogger(__name__)

R_EARTH_RSUN = 0.009168
AU_RSUN = 215.032
ALIASES = (0.5, 2.0, 1 / 3, 3.0)

# Threshold grid searched by calibrate().
SDE_GRID = np.round(np.arange(6.0, 30.01, 0.25), 2)
SNR_GRID = np.round(np.arange(5.0, 20.01, 0.5), 2)


@dataclass(frozen=True)
class Injection:
    index: int
    period_days: float
    t0_btjd: float
    radius_rearth: float
    impact: float
    rp_rs: float
    depth: float  # (Rp/R*)², before limb darkening
    duration_hours: float

    def transit(self) -> Transit:
        return Transit(
            t0_btjd=self.t0_btjd,
            period_days=self.period_days,
            depth=self.depth,
            duration_hours=self.duration_hours,
            impact=self.impact,
        )


def transit_duration_hours(
    period_days: float, rp_rs: float, impact: float, radius_rsun: float, mass_msun: float
) -> float:
    """Total (T14) duration for a circular orbit."""
    a_au = (mass_msun * (period_days / 365.25) ** 2) ** (1 / 3)
    a_rs = a_au * AU_RSUN / radius_rsun
    if impact >= a_rs:
        return 0.0
    sin_i = np.sqrt(1.0 - (impact / a_rs) ** 2)
    chord = np.sqrt(max((1 + rp_rs) ** 2 - impact**2, 0.0))
    x = np.clip(chord / (a_rs * sin_i), 0.0, 1.0)
    return float(period_days / np.pi * np.arcsin(x) * 24.0)


def _stellar(lc: LightCurve) -> tuple[float, float]:
    star = lc.meta.get("star") or {}
    r, m = star.get("radius_rsun"), star.get("mass_msun")
    r = float(r) if r and np.isfinite(r) else 1.0
    m = float(m) if m and np.isfinite(m) else 1.0
    return r, m


def draw_injection(
    i: int, tic: int, lc: LightCurve, icfg: InjectConfig, pmax_search: float
) -> Injection:
    """Injection ``i`` for this star. Seeded per (seed, tic, i), so runs resume."""
    rng = np.random.default_rng([icfg.seed, int(tic), i])
    pmax = icfg.period_max_days if icfg.period_max_days > 0 else pmax_search
    pmin = icfg.period_min_days
    period = float(np.exp(rng.uniform(np.log(pmin), np.log(pmax))))
    radius = float(
        np.exp(rng.uniform(np.log(icfg.radius_min_rearth), np.log(icfg.radius_max_rearth)))
    )
    impact = float(rng.uniform(0.0, icfg.impact_max))
    t0 = float(np.nanmin(lc.time) + rng.uniform(0.0, period))
    r_star, m_star = _stellar(lc)
    rp_rs = radius * R_EARTH_RSUN / r_star
    dur = transit_duration_hours(period, rp_rs, impact, r_star, m_star)
    return Injection(
        index=i,
        period_days=period,
        t0_btjd=t0,
        radius_rearth=radius,
        impact=impact,
        rp_rs=rp_rs,
        depth=rp_rs**2,
        duration_hours=dur,
    )


def inject(lc: LightCurve, inj: Injection) -> LightCurve:
    out = lc.select(np.ones(len(lc), dtype=bool))
    out.flux = lc.flux * transit_model(lc.time, inj.transit())
    return out


def _phase_diff(t_a: float, t_b: float, period: float) -> float:
    return abs((t_a - t_b + 0.5 * period) % period - 0.5 * period)


def match(inj: Injection, dets: list[Detection], tol: float) -> tuple[str, Detection | None]:
    """'recovered', 'alias' or 'missed', with the matching detection."""
    window = max(0.5 * inj.duration_hours / 24.0, 1.0 / 24.0)
    for d in dets:
        if not np.isfinite(d.period_days):
            continue
        if abs(d.period_days / inj.period_days - 1) < tol:
            if _phase_diff(d.t0_btjd, inj.t0_btjd, inj.period_days) < window:
                return "recovered", d
    for d in dets:
        if not np.isfinite(d.period_days):
            continue
        for k in ALIASES:
            p = k * inj.period_days
            if abs(d.period_days / p - 1) < tol:
                # Epochs line up modulo the shorter of the two periods.
                if _phase_diff(d.t0_btjd, inj.t0_btjd, min(p, inj.period_days)) < window:
                    return "alias", d
    return "missed", None


def scramble(flat: LightCurve, rng: np.random.Generator) -> LightCurve:
    """Roll each sector's flux by a random offset, keeping its timestamps."""
    out = flat.select(np.ones(len(flat), dtype=bool))
    for s in out.sectors:
        idx = np.flatnonzero(out.sector == s)
        n = len(idx)
        if n < 10:
            continue
        shift = int(rng.integers(n // 10, n - n // 10))
        out.flux[idx] = np.roll(out.flux[idx], shift)
        out.flux_err[idx] = np.roll(out.flux_err[idx], shift)
    return out


def invert(lc: LightCurve) -> LightCurve:
    out = lc.select(np.ones(len(lc), dtype=bool))
    out.flux = 2.0 - lc.flux
    return out


def _det_summary(d: Detection | None) -> dict[str, Any] | None:
    if d is None:
        return None
    keep = ("index", "period_days", "t0_btjd", "duration_hours", "depth_ppm", "snr", "sde")
    out = {k: getattr(d, k) for k in keep}
    out["n_transits"] = d.n_transits
    out["flags"] = list(d.flags)
    return out


def _injection_search_cfg(cfg: Config) -> PeriodicSearchConfig:
    # The inverted check is a per-star yardstick, not part of recovery.
    return dataclasses.replace(cfg.search.periodic, run_inverted=False)


def _null_search_cfg(cfg: Config) -> PeriodicSearchConfig:
    # Only the strongest peak of a null matters.
    return dataclasses.replace(cfg.search.periodic, run_inverted=False, max_signals=1)


# --- per-star driver -------------------------------------------------------------


def inject_dir(cfg: Config, tic: int) -> Path:
    return cfg.run_dir / "inject" / f"TIC{int(tic)}"


def config_key(cfg: Config, upstream: str) -> str:
    """What invalidates earlier results: detrend, search, the injection law, the data."""
    from scout.pipeline import _key

    law = {
        k: v
        for k, v in asdict(cfg.inject).items()
        if k not in {"n_injections", "n_scrambles", "false_alarm_target"}
    }
    return _key("inject", cfg.detrend, cfg.search.periodic, law, upstream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            # A crash mid-write leaves at most one partial last line.
            log.warning("%s: skipping unreadable line", path)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(_to_plain(row), default=str) + "\n")
        fh.flush()


def _prepare(cfg: Config, tic: int, upstream: str) -> Path:
    d = inject_dir(cfg, tic)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "inject.json"
    key = config_key(cfg, upstream)
    old_key = None
    if meta_path.exists():
        try:
            old_key = json.loads(meta_path.read_text()).get("key")
        except ValueError:
            old_key = None
    if old_key != key:
        if old_key is not None:
            log.info("TIC %s: config or data changed, discarding earlier injections", tic)
        for name in ("injections.jsonl", "nulls.jsonl"):
            (d / name).unlink(missing_ok=True)
        meta_path.write_text(json.dumps({"tic": tic, "key": key}, indent=2))
    return d


def run_star_injections(
    cfg: Config,
    tic: int,
    stitched: LightCurve,
    upstream: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run (or resume) the injections and nulls for one star."""
    from scout.provenance import provenance

    d = _prepare(cfg, tic, upstream)
    icfg = cfg.inject
    say = progress or (lambda msg: log.info("%s", msg))

    flat = detrend(stitched, cfg.detrend)
    _, pmax = period_range(flat, cfg.search.periodic)
    meta = json.loads((d / "inject.json").read_text())
    meta.update(
        {
            "sectors": stitched.sectors,
            "baseline_days": stitched.baseline_days,
            "n_cadences": len(stitched),
            "star": stitched.meta.get("star", {}),
            "flat_rms_ppm": float(np.std(flat.flux) * 1e6),
            "rotation_prewhiten": flat.meta.get("detrend", {}).get("rotation_prewhiten"),
            "search_period_max_days": pmax,
            "provenance": provenance({"config_digest": cfg.digest()}),
        }
    )
    (d / "inject.json").write_text(json.dumps(_to_plain(meta), indent=2, default=str))

    # Nulls first: they are few and set the threshold.
    null_path = d / "nulls.jsonl"
    done = {(r["kind"], r["i"]) for r in _read_jsonl(null_path)}
    ncfg = _null_search_cfg(cfg)
    todo: list[tuple[str, int, LightCurve]] = []
    if ("inverted", 0) not in done:
        todo.append(("inverted", 0, invert(flat)))
    for i in range(1, icfg.n_scrambles + 1):
        need = [k for k in ("scrambled", "scrambled_inverted") if (k, i) not in done]
        if not need:
            continue
        sc = scramble(flat, np.random.default_rng([icfg.seed, int(tic), 10_000_000 + i]))
        for kind in need:
            todo.append((kind, i, sc if kind == "scrambled" else invert(sc)))
    for n, (kind, i, lc) in enumerate(todo, 1):
        res = search_periodic(lc, ncfg)
        best = res.detections[0] if res.detections else None
        _append_jsonl(null_path, {"tic": tic, "kind": kind, "i": i, "best": _det_summary(best)})
        say(f"TIC {tic}: null {n}/{len(todo)} ({kind} #{i})")

    inj_path = d / "injections.jsonl"
    done_i = {r["index"] for r in _read_jsonl(inj_path)}
    scfg = _injection_search_cfg(cfg)
    todo_i = [i for i in range(icfg.n_injections) if i not in done_i]
    for n, i in enumerate(todo_i, 1):
        inj = draw_injection(i, tic, stitched, icfg, pmax)
        if inj.duration_hours <= 0:
            row = {**asdict(inj), "tic": tic, "status": "no_transit", "detection": None}
        else:
            res = search_periodic(detrend(inject(stitched, inj), cfg.detrend), scfg)
            status, det = match(inj, res.detections, icfg.period_tolerance)
            row = {
                **asdict(inj),
                "tic": tic,
                "status": status,
                "detection": _det_summary(det),
                "best": _det_summary(res.detections[0] if res.detections else None),
            }
        _append_jsonl(inj_path, row)
        say(f"TIC {tic}: injection {n}/{len(todo_i)} -> {row['status']}")

    return {
        "tic": tic,
        "injections": len(_read_jsonl(inj_path)),
        "nulls": len(_read_jsonl(null_path)),
        "dir": str(d),
    }


# --- calibration -----------------------------------------------------------------


def _iter_star_dirs(cfg: Config) -> Iterator[Path]:
    root = cfg.run_dir / "inject"
    if root.exists():
        yield from sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("TIC"))


def load_run(cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict]]:
    """All injections, nulls and host summaries of a run."""
    injections: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    for d in _iter_star_dirs(cfg):
        injections += _read_jsonl(d / "injections.jsonl")
        nulls += _read_jsonl(d / "nulls.jsonl")
        if (d / "inject.json").exists():
            hosts.append(json.loads((d / "inject.json").read_text()))
    return injections, nulls, hosts


def _passes(det: dict[str, Any] | None, sde: float, snr: float, min_tr: int) -> bool:
    return (
        det is not None
        and det["sde"] >= sde
        and det["snr"] >= snr
        and det.get("n_transits", min_tr) >= min_tr
    )


def recovered_mask(
    injections: list[dict[str, Any]], sde: float, snr: float, min_tr: int
) -> np.ndarray:
    hits = [
        r["status"] == "recovered" and _passes(r["detection"], sde, snr, min_tr) for r in injections
    ]
    return np.array(hits, dtype=bool)


def false_alarm_rate(nulls: list[dict[str, Any]], sde: float, snr: float, min_tr: int) -> float:
    if not nulls:
        return float("nan")
    return float(np.mean([_passes(r["best"], sde, snr, min_tr) for r in nulls]))


def choose_thresholds(
    injections: list[dict[str, Any]],
    nulls: list[dict[str, Any]],
    target: float,
    min_tr: int,
) -> dict[str, Any]:
    """Max recovery subject to false-alarm rate ≤ target, over the SDE × SNR grid."""
    inj = [r for r in injections if r["status"] != "no_transit"]
    if not nulls:
        raise ValueError("no null samples: run `scout inject` first")

    # Vectorise: arrays of (sde, snr, enough transits) for nulls and recovered injections.
    def arr(rows: list[dict[str, Any]], key: str, only_rec: bool) -> np.ndarray:
        out = []
        for r in rows:
            det = r[key]
            ok = det is not None and (not only_rec or r["status"] == "recovered")
            ok = ok and det.get("n_transits", min_tr) >= min_tr
            out.append((det["sde"], det["snr"]) if ok else (-np.inf, -np.inf))
        return np.array(out, dtype=float).reshape(-1, 2)

    nv = arr(nulls, "best", False)
    iv = arr(inj, "detection", True)
    best: dict[str, Any] | None = None
    for s in SDE_GRID:
        for n in SNR_GRID:
            far = float(np.mean((nv[:, 0] >= s) & (nv[:, 1] >= n)))
            if far > target:
                continue
            rec = float(np.mean((iv[:, 0] >= s) & (iv[:, 1] >= n))) if len(iv) else 0.0
            # Ties go to the lower (more sensitive, simpler) pair found first.
            if best is None or rec > best["recovery"] + 1e-12:
                best = {"sde": float(s), "snr": float(n), "recovery": rec, "false_alarm": far}
    if best is None:
        s, n = float(SDE_GRID[-1]), float(SNR_GRID[-1])
        best = {
            "sde": s,
            "snr": n,
            "recovery": float(np.mean((iv[:, 0] >= s) & (iv[:, 1] >= n))) if len(iv) else 0.0,
            "false_alarm": float(np.mean((nv[:, 0] >= s) & (nv[:, 1] >= n))),
            "warning": "no grid point meets the false-alarm target; showing the strictest",
        }
    best["n_nulls"] = len(nulls)
    best["n_injections"] = len(inj)
    if len(nulls) < 3 / target:
        best["warning"] = (
            f"only {len(nulls)} null samples; a false-alarm rate of {target:g} needs "
            f"≥ {int(np.ceil(3 / target))} to be measured, so the threshold is rough"
        )
    return best


def recovery_table(
    injections: list[dict[str, Any]],
    sde: float,
    snr: float,
    min_tr: int,
    period_edges: np.ndarray,
    radius_edges: np.ndarray,
) -> dict[str, Any]:
    inj = [r for r in injections if r["status"] != "no_transit"]
    if not inj:
        shape = (len(radius_edges) - 1, len(period_edges) - 1)
        return {"fraction": np.full(shape, np.nan), "count": np.zeros(shape, int)}
    p = np.array([r["period_days"] for r in inj])
    rad = np.array([r["radius_rearth"] for r in inj])
    rec = recovered_mask(inj, sde, snr, min_tr)
    count, _, _ = np.histogram2d(rad, p, bins=[radius_edges, period_edges])
    hit, _, _ = np.histogram2d(rad[rec], p[rec], bins=[radius_edges, period_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, hit / count, np.nan)
    return {"fraction": frac, "count": count.astype(int)}


def calibrate(cfg: Config) -> dict[str, Any]:
    """Pool a run's injections and nulls; choose thresholds; write the report."""
    from scout.provenance import provenance
    from scout.report.calibration import write_calibration_report

    injections, nulls, hosts = load_run(cfg)
    if not injections and not nulls:
        raise ValueError(f"no injection results under {cfg.run_dir / 'inject'}")
    pcfg = cfg.search.periodic
    min_tr = pcfg.min_transits
    chosen = choose_thresholds(injections, nulls, cfg.inject.false_alarm_target, min_tr)
    current = {
        "sde": pcfg.sde_threshold,
        "snr": pcfg.snr_threshold,
        "recovery": float(
            np.mean(
                recovered_mask(
                    [r for r in injections if r["status"] != "no_transit"],
                    pcfg.sde_threshold,
                    pcfg.snr_threshold,
                    min_tr,
                )
            )
        )
        if injections
        else float("nan"),
        "false_alarm": false_alarm_rate(nulls, pcfg.sde_threshold, pcfg.snr_threshold, min_tr),
    }
    per_host = []
    for h in hosts:
        tic = h["tic"]
        hn = [r for r in nulls if r["tic"] == tic]
        hi = [r for r in injections if r["tic"] == tic and r["status"] != "no_transit"]
        per_host.append(
            {
                "tic": tic,
                "sectors": h.get("sectors"),
                "flat_rms_ppm": h.get("flat_rms_ppm"),
                "rotation_periods_days": (h.get("rotation_prewhiten") or {}).get("periods_days"),
                "n_injections": len(hi),
                "n_nulls": len(hn),
                "recovery": float(np.mean(recovered_mask(hi, chosen["sde"], chosen["snr"], min_tr)))
                if hi
                else float("nan"),
                "false_alarm": false_alarm_rate(hn, chosen["sde"], chosen["snr"], min_tr),
                "max_null_sde": max(
                    (r["best"]["sde"] for r in hn if r["best"] is not None), default=float("nan")
                ),
            }
        )
    status_counts: dict[str, int] = {}
    for r in injections:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    result = {
        "run": cfg.run.name,
        "chosen": chosen,
        "current": current,
        "false_alarm_target": cfg.inject.false_alarm_target,
        "status_counts": status_counts,
        "hosts": per_host,
        "provenance": provenance({"config_digest": cfg.digest()}),
    }
    out = cfg.run_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(_to_plain(result), indent=2, default=str))
    write_calibration_report(out / "calibration.html", cfg, injections, nulls, result)
    result["report"] = str(out / "calibration.html")
    return result
