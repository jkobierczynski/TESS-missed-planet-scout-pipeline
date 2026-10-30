"""Calibration report: recovery maps and false-alarm distributions (M2).

One self-contained page per run, written by ``scout calibrate``, so the
threshold choice can be checked by eye: where in period × radius the search
works, how the null searches are distributed, and how each host behaved.
"""

from __future__ import annotations

import html
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config
from scout.inject import recovered_mask, recovery_table
from scout.report.star import _CSS, _fig_to_data_uri, _fmt, _row, _setup_matplotlib


def _edges(lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))


def _injection_edges(cfg: Config, injections: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    p = [r["period_days"] for r in injections] or [cfg.inject.period_min_days, 10.0]
    i = cfg.inject
    return _edges(min(p), max(p) * 1.0001, 6), _edges(i.radius_min_rearth, i.radius_max_rearth, 6)


def plot_recovery_map(
    injections: list[dict[str, Any]], sde: float, snr: float, min_tr: int, cfg: Config, dpi: int
) -> str:
    import matplotlib.pyplot as plt

    pe, re_ = _injection_edges(cfg, injections)
    tab = recovery_table(injections, sde, snr, min_tr, pe, re_)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mesh = ax.pcolormesh(pe, re_, tab["fraction"], vmin=0, vmax=1, cmap="viridis")
    for iy in range(len(re_) - 1):
        for ix in range(len(pe) - 1):
            n = tab["count"][iy, ix]
            if n == 0:
                continue
            f = tab["fraction"][iy, ix]
            xc, yc = np.sqrt(pe[ix] * pe[ix + 1]), np.sqrt(re_[iy] * re_[iy + 1])
            ax.text(
                xc,
                yc,
                f"{f:.0%}\n{n}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if f < 0.6 else "black",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Injected period (d)")
    ax.set_ylabel("Injected radius (R⊕)")
    ax.set_title(f"Recovered fraction at SDE ≥ {sde:g}, SNR ≥ {snr:g} (count per cell)")
    fig.colorbar(mesh, ax=ax, label="recovered fraction")
    return _fig_to_data_uri(fig, dpi)


def plot_recovery_curves(
    injections: list[dict[str, Any]], pairs: list[tuple[str, float, float]], min_tr: int, dpi: int
) -> str:
    import matplotlib.pyplot as plt

    inj = [r for r in injections if r["status"] != "no_transit"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if not inj:
        return _fig_to_data_uri(fig, dpi)
    rad = np.array([r["radius_rearth"] for r in inj])
    per = np.array([r["period_days"] for r in inj])
    for ax, x, label in ((axes[0], rad, "Injected radius (R⊕)"), (axes[1], per, "Period (d)")):
        edges = _edges(x.min(), x.max() * 1.0001, 8)
        mid = np.sqrt(edges[:-1] * edges[1:])
        for name, s, n in pairs:
            rec = recovered_mask(inj, s, n, min_tr)
            frac = [
                rec[(x >= a) & (x < b)].mean() if ((x >= a) & (x < b)).any() else np.nan
                for a, b in itertools.pairwise(edges)
            ]
            ax.plot(mid, frac, "o-", label=f"{name}: SDE {s:g}, SNR {n:g}")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(label)
        ax.set_ylabel("recovered fraction")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_data_uri(fig, dpi)


def plot_nulls(
    injections: list[dict[str, Any]],
    nulls: list[dict[str, Any]],
    pairs: list[tuple[str, float, float]],
    dpi: int,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    rec = [r["detection"] for r in injections if r["status"] == "recovered" and r["detection"]]
    if rec:
        ax.scatter(
            [d["sde"] for d in rec],
            [d["snr"] for d in rec],
            s=8,
            c="#4c72b0",
            alpha=0.5,
            label="recovered injections",
        )
    for kind, color in (
        ("inverted", "#c44e52"),
        ("scrambled", "#dd8452"),
        ("scrambled_inverted", "#8172b3"),
    ):
        b = [r["best"] for r in nulls if r["kind"] == kind and r["best"]]
        if b:
            ax.scatter(
                [d["sde"] for d in b],
                [d["snr"] for d in b],
                s=14,
                c=color,
                marker="x",
                label=f"null: {kind}",
            )
    for (name, s, n), ls in zip(pairs, ("-", "--"), strict=False):
        ax.axvline(s, color="k", ls=ls, lw=0.8)
        ax.axhline(n, color="k", ls=ls, lw=0.8, label=f"{name} thresholds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SDE")
    ax.set_ylabel("SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _fig_to_data_uri(fig, dpi)


def write_calibration_report(
    path: Path,
    cfg: Config,
    injections: list[dict[str, Any]],
    nulls: list[dict[str, Any]],
    result: dict[str, Any],
) -> Path:
    _setup_matplotlib()
    dpi = cfg.report.dpi
    ch, cur = result["chosen"], result["current"]
    min_tr = cfg.search.periodic.min_transits
    pairs = [("chosen", ch["sde"], ch["snr"]), ("config", cur["sde"], cur["snr"])]

    rmap = plot_recovery_map(injections, ch["sde"], ch["snr"], min_tr, cfg, dpi)
    curves = plot_recovery_curves(injections, pairs, min_tr, dpi)
    null_plot = plot_nulls(injections, nulls, pairs, dpi)

    host_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(v))}</td>"
            for v in (
                h["tic"],
                len(h.get("sectors") or []),
                _fmt(h.get("flat_rms_ppm"), 0),
                ", ".join(f"{p:.2f}" for p in (h.get("rotation_periods_days") or [])) or "—",
                h["n_injections"],
                _fmt(h["recovery"], 2),
                h["n_nulls"],
                _fmt(h["false_alarm"], 2),
                _fmt(h["max_null_sde"], 1),
            )
        )
        + "</tr>"
        for h in result["hosts"]
    )
    toml = (
        "[search.periodic]\n"
        f"sde_threshold = {ch['sde']:g}   # M2 calibration, run '{result['run']}'\n"
        f"snr_threshold = {ch['snr']:g}"
    )
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Calibration {html.escape(result['run'])}</title><style>{_CSS}</style></head>",
        "<body>",
        f"<h1>Injection-recovery calibration · {html.escape(result['run'])}</h1>",
        f"<div class='muted'>{len(result['hosts'])} host stars · "
        f"{ch['n_injections']} injections · {ch['n_nulls']} null searches · "
        f"false-alarm target {result['false_alarm_target']:g} per search</div>",
    ]
    if ch.get("warning"):
        parts.append(f"<div class='banner'>{html.escape(ch['warning'])}</div>")
    parts += [
        "<h2>Thresholds</h2><table>",
        "<tr><th></th><th>SDE</th><th>SNR</th><th>recovered</th><th>false alarms</th></tr>",
        *(
            f"<tr><th>{name}</th><td>{d['sde']:g}</td><td>{d['snr']:g}</td>"
            f"<td>{_fmt(d['recovery'], 3)}</td><td>{_fmt(d['false_alarm'], 3)}</td></tr>"
            for name, d in (("chosen", ch), ("config now", cur))
        ),
        "</table>",
        f"<p>To adopt the chosen pair, set in the config:</p><pre>{html.escape(toml)}</pre>",
        "<h2>Recovery map</h2>",
        f"<img alt='recovery map' src='{rmap}'>",
        f"<img alt='recovery curves' src='{curves}'>",
        "<h2>Null searches versus recovered injections</h2>",
        "<p>Each null is the strongest peak of one search on a scrambled or inverted "
        "light curve. Thresholds must sit above the nulls.</p>",
        f"<img alt='nulls' src='{null_plot}'>",
        "<h2>Injection outcomes</h2><table>",
        *(_row(k, v) for k, v in sorted(result["status_counts"].items())),
        "</table>",
        "<h2>Hosts</h2><table>",
        "<tr><th>TIC</th><th>sectors</th><th>flat rms (ppm)</th><th>P_rot removed (d)</th>"
        "<th>injections</th><th>recovered</th><th>nulls</th><th>false alarms</th>"
        "<th>max null SDE</th></tr>",
        host_rows,
        "</table>",
        "<h2>Provenance</h2>",
        f"<pre>{html.escape(json.dumps(result['provenance'], indent=2, default=str))}</pre>",
        "</body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path
