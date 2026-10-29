"""Per-star HTML report: one self-contained page for human review (design §7).

Plots are embedded as base64 PNG so a report is a single file that can be
opened anywhere. The vetting section is a placeholder until M5; every row of
the §6 checklist is listed so a reviewer sees what has *not* been checked yet.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import ReportConfig
from scout.lightcurve import LightCurve, bin_per_sector
from scout.search.periodic import Detection, PeriodicSearchResult

VETTING_CHECKLIST = (
    "Transit shape plausible (not V-shaped unless grazing is justified)",
    "Odd/even depth consistent",
    "No secondary eclipse",
    "Depth consistent across sectors and light-curve sources",
    "No centroid shift in TPF difference images",
    "Gaia DR3: no bright neighbour in aperture; RUWE fine",
    "Implied companion radius < ~2 R_Jup",
    "Duration consistent with stellar density",
    "Not coincident with spacecraft events",
    "TRICERATOPS FPP below threshold",
    "Not already known (re-check before submission)",
)


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    # Plain flux values on axes, never "+1" offsets.
    matplotlib.rcParams["axes.formatter.useoffset"] = False


def _fig_to_data_uri(fig: Any, dpi: int) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _phase(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in days from mid-transit, in [-P/2, P/2)."""
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def mask_signals(lc: LightCurve, dets: list[Detection], factor: float = 2.0) -> LightCurve:
    """Drop cadences within ``factor`` × duration of the given signals' transits."""
    keep = np.ones(len(lc), dtype=bool)
    for d in dets:
        keep &= np.abs(_phase(lc.time, d.period_days, d.t0_btjd)) > factor * d.duration_hours / 48
    return lc.select(keep)


def _binned(x: np.ndarray, y: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x, y
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges)
    xs, ys = [], []
    for i in np.unique(idx):
        m = idx == i
        if m.sum() >= 3:
            xs.append(x[m].mean())
            ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def plot_overview(stitched: LightCurve, flat: LightCurve, dets: list[Detection], dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    ax = axes[0]
    ax.plot(stitched.time, stitched.flux, ",", color="0.55", rasterized=True)
    if flat.trend is not None:
        ax.plot(flat.time, flat.trend, "-", color="tab:orange", lw=0.8, label="trend")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("normalised flux")
    ax.set_title("Stitched (top) and flattened (bottom)", fontsize=10)
    ax = axes[1]
    ax.plot(flat.time, flat.flux, ",", color="0.35", rasterized=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for d, c in zip(dets, colors, strict=False):
        for tt in d.transit_times_btjd:
            ax.axvline(tt, color=c, alpha=0.35, lw=0.8)
    for s in flat.sectors:
        t = flat.time[flat.sector == s]
        ax.text(
            t.min(),
            1.0,
            f"S{s}",
            fontsize=7,
            va="bottom",
            color="0.3",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlabel("Time (BTJD = BJD_TDB − 2457000)")
    ax.set_ylabel("flattened flux")
    lo, hi = np.nanpercentile(flat.flux, [0.1, 99.9])
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    return _fig_to_data_uri(fig, dpi)


def plot_detection(flat: LightCurve, d: Detection, dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    dur = d.duration_hours / 24.0
    window = max(3 * dur, 0.25)
    ph = _phase(flat.time, d.period_days, d.t0_btjd)
    epoch = np.round((flat.time - d.t0_btjd) / d.period_days).astype(int)
    near = np.abs(ph) < window

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(ph[near] * 24, flat.flux[near], ".", ms=1.5, color="0.6", rasterized=True)
    bx, by = _binned(ph[near] * 24, flat.flux[near], max(dur * 24 / 8, 0.25))
    ax.plot(bx, by, "o", ms=3.5, color="tab:blue")
    ax.axhline(1 - d.depth_ppm * 1e-6, color="tab:red", lw=0.8, ls="--")
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux")
    ax.set_title(f"Phase fold  P = {d.period_days:.5f} d", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    for parity, color, label in ((0, "tab:blue", "even"), (1, "tab:orange", "odd")):
        m = near & (epoch % 2 == parity)
        bx, by = _binned(ph[m] * 24, flat.flux[m], max(dur * 24 / 6, 0.3))
        ax.plot(bx, by, "o", ms=3.5, color=color, label=label)
    ax.axhline(1 - d.depth_ppm * 1e-6, color="tab:red", lw=0.8, ls="--")
    ax.set_title(f"Odd / even  (mismatch {_fmt(d.odd_even_sigma, 1)}σ)", fontsize=10)
    ax.set_xlabel("hours from mid-transit")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ph2 = _phase(flat.time, d.period_days, d.t0_btjd + 0.5 * d.period_days)
    m = np.abs(ph2) < window
    ax.plot(ph2[m] * 24, flat.flux[m], ".", ms=1.5, color="0.6", rasterized=True)
    bx, by = _binned(ph2[m] * 24, flat.flux[m], max(dur * 24 / 6, 0.3))
    ax.plot(bx, by, "o", ms=3, color="tab:purple")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_title("Phase 0.5 (secondary eclipse?)", fontsize=10)
    ax.set_xlabel("hours from phase 0.5")

    ax = fig.add_subplot(gs[2, :])
    times = d.transit_times_btjd[:24]
    shown = []
    for i, tt in enumerate(times):
        m = np.abs(flat.time - tt) < window
        if not m.any():
            continue
        ax.plot(
            (flat.time[m] - tt) * 24 + i * (2 * window * 24 + 2),
            flat.flux[m],
            ".",
            ms=1.2,
            color="0.4",
            rasterized=True,
        )
        shown.append(flat.flux[m])
    if shown:
        lo, hi = np.nanpercentile(np.concatenate(shown), [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        ax.set_ylim(min(lo, 1 - 1.3 * d.depth_ppm * 1e-6) - pad, hi + pad)
    ax.set_xticks([])
    ax.set_ylabel("flux")
    ax.set_title(
        f"Individual transits (first {len(times)}, each ±{window * 24:.1f} h)", fontsize=10
    )
    fig.tight_layout()
    return _fig_to_data_uri(fig, dpi)


_CSS = """
:root { color-scheme: light dark; --fg:#1d1d1f; --bg:#fafafa; --muted:#666; --line:#ddd;
  --pass:#1a7f37; --fail:#b42318; --todo:#8a6d00; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6e6e6; --bg:#16171a; --muted:#9a9a9a;
  --line:#333; --pass:#4ac26b; --fail:#ff7b72; --todo:#d4a72c; } }
body { font: 14px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg);
  max-width: 1200px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin-top: 28px;
  border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.muted { color: var(--muted); } img { max-width: 100%; height: auto; background: #fff; }
table { border-collapse: collapse; margin: 8px 0; } td, th { padding: 3px 10px;
  border-bottom: 1px solid var(--line); text-align: left; font-variant-numeric: tabular-nums; }
.pass { color: var(--pass); } .fail { color: var(--fail); } .todo { color: var(--todo); }
pre { font-size: 12px; overflow-x: auto; border: 1px solid var(--line); padding: 8px; }
.banner { border: 1px solid var(--todo); padding: 6px 10px; margin: 10px 0; }
"""


def _row(k: str, v: Any) -> str:
    return f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"


def _fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def detection_table(d: Detection, star: dict[str, Any]) -> str:
    rows = [
        ("Period (d)", f"{_fmt(d.period_days, 5)} ± {_fmt(d.period_err_days, 5)}"),
        ("T0 (BTJD)", _fmt(d.t0_btjd, 4)),
        ("T0 (BJD_TDB)", _fmt(d.t0_btjd + 2457000.0, 4)),
        ("Duration (h)", _fmt(d.duration_hours, 2)),
        ("Depth (ppm)", _fmt(d.depth_ppm, 0)),
        ("SDE", _fmt(d.sde, 1)),
        ("SNR", _fmt(d.snr, 1)),
        ("Transits with data", f"{d.n_transits} of {d.n_transits_expected}"),
        ("Odd/even mismatch (σ)", _fmt(d.odd_even_sigma, 2)),
        ("Rp/R*", _fmt(d.rp_rs, 4)),
    ]
    r_star = star.get("radius_rsun")
    if r_star and np.isfinite(d.rp_rs):
        rows.append(("Rp (R⊕) from TIC R*", _fmt(d.rp_rs * r_star * 109.08, 2)))
    rows.append(("Flags", ", ".join(d.flags) or "none"))
    if d.warnings:
        rows.append(("TLS warnings", "; ".join(d.warnings)))
    return "<table>" + "".join(_row(k, v) for k, v in rows) + "</table>"


def write_star_report(
    path: str | Path,
    stitched: LightCurve,
    flat: LightCurve,
    result: PeriodicSearchResult,
    provenance: dict[str, Any],
    cfg: ReportConfig,
) -> Path:
    path = Path(path)
    star = flat.meta.get("star") or {}
    tic = flat.tic
    title = f"TIC {tic}" if tic is not None else "Unknown target"
    # Plot from a lightly binned copy to keep the page small; metrics come from
    # the search, not from these plots.
    st_plot = bin_per_sector(stitched, 10.0)
    fl_plot = bin_per_sector(flat, 10.0)
    overview = plot_overview(st_plot, fl_plot, result.detections, cfg.dpi)

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)} report</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='muted'>sectors {', '.join(map(str, flat.sectors))} · "
        f"baseline {flat.baseline_days:.1f} d · {len(flat)} cadences · "
        f"source {html.escape(str(flat.meta.get('source', '?')))}</div>",
        "<div class='banner'>Not vetted. Detections below have only passed the search "
        "thresholds; the §6 checklist has not been run.</div>",
        "<h2>Star</h2><table>",
        *(_row(k, v) for k, v in star.items() if v is not None),
        "</table>",
        "<h2>Light curve</h2>",
        f"<img alt='light curve' src='{overview}'>",
        "<h2>Search</h2>",
        f"<p>TLS over {result.period_grid_days[0]:.2f}–{result.period_grid_days[1]:.2f} d on "
        f"{result.n_points} points. Inverted-flux best SDE: "
        f"<b>{_fmt(result.inverted_sde, 1)}</b> (a real signal should clearly beat this).</p>",
    ]
    if not result.detections:
        parts.append("<p>No detections.</p>")
    for d in result.detections:
        # Each signal was found after masking the ones before it; plot it the same way.
        earlier = [e for e in result.detections if e.index < d.index and e.passes_threshold]
        flat_d = mask_signals(flat, earlier)
        status = (
            "<span class='pass'>above threshold</span>"
            if d.passes_threshold
            else "<span class='fail'>below threshold</span>"
        )
        parts += [
            f"<h2>Signal {d.index} · {status}</h2>",
            detection_table(d, star),
            f"<img alt='signal {d.index}' src='{plot_detection(flat_d, d, cfg.dpi)}'>",
        ]
    parts += [
        "<h2>Vetting checklist</h2><table>",
        *(
            f"<tr><td class='todo'>not run</td><td>{html.escape(c)}</td></tr>"
            for c in VETTING_CHECKLIST
        ),
        "</table>",
        "<h2>Provenance</h2>",
        f"<pre>{html.escape(json.dumps(provenance, indent=2, default=str))}</pre>",
        "</body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path
