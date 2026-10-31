"""Vetting report per signal and the run's vetting summary (design §6, §7)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config
from scout.report.star import _CSS, _fig_to_data_uri, _fmt, _row, _setup_matplotlib
from scout.vet import checks as ck

_EXTRA_CSS = """
.warn { color: var(--todo); } .n\\/a, .na { color: var(--muted); }
.verdict { font-size: 18px; font-weight: 600; margin: 8px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 8px; }
th[data-k] { cursor: pointer; } td { white-space: nowrap; } .wide { overflow-x: auto; }
td.detail { white-space: normal; }
"""

_VERDICT_CLASS = {"fail": "fail", "review": "todo", "candidate": "pass"}
_STATUS_CLASS = {"pass": "pass", "fail": "fail", "warn": "warn", "n/a": "na"}


def _binned(
    x: np.ndarray, y: np.ndarray, width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0 or width <= 0:
        return x, y, np.zeros(0)
    idx = np.floor((x - x.min()) / width).astype(int)
    n = np.bincount(idx)
    keep = n >= 3
    xs = np.bincount(idx, x)[keep] / n[keep]
    ys = np.bincount(idx, y)[keep] / n[keep]
    ss = np.sqrt(np.maximum(np.bincount(idx, y * y)[keep] / n[keep] - ys**2, 0)) / np.sqrt(n[keep])
    return xs, ys, ss


def plot_fold(f: ck.Folded, fit: dict[str, Any], dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    eph = f.eph
    w = 3 * eph.dur_days
    m = np.abs(f.phase) < w
    x, y = f.phase[m] * 24, f.flux[m]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(x, y, ",", color="0.6", rasterized=True)
    xb, yb, _ = _binned(x, y, eph.duration_hours / 12)
    ax.plot(xb, yb, "o", ms=3, color="k")
    if fit.get("ok"):
        xx = np.linspace(-w, w, 600)
        ax.plot(
            xx * 24,
            ck.transit_model_curve({**fit, "dt0_days": 0.0}, eph, xx),
            color="tab:red",
            lw=1.2,
        )
    d = eph.depth_ppm * 1e-6
    ax.set_ylim(1 - 2.2 * d - 3 * f.sigma / np.sqrt(30), 1 + 1.0 * d + 3 * f.sigma / np.sqrt(30))
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flattened flux")
    ax.set_title("Phase fold with transit fit", fontsize=10)
    # Odd and even overlaid.
    ax = axes[1]
    for par, color, label in ((1, "tab:blue", "odd"), (0, "tab:orange", "even")):
        mm = m & (f.epoch % 2 == par)
        xb, yb, sb = _binned(f.phase[mm] * 24, f.flux[mm], eph.duration_hours / 8)
        ax.errorbar(xb, yb, sb, fmt="o", ms=3, color=color, label=label, lw=0.8)
    ax.axhline(1 - d, color="0.4", ls=":", lw=0.8)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("hours from mid-transit")
    ax.legend(fontsize=8)
    ax.set_title("Odd and even transits", fontsize=10)
    fig.tight_layout()
    return _fig_to_data_uri(fig, dpi)


def plot_full_phase(f: ck.Folded, sec: dict[str, Any] | None, dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    eph = f.eph
    ph = (f.phase / eph.period_days + 0.25) % 1.0 - 0.25
    fig, ax = plt.subplots(figsize=(12, 3.2))
    width = max(eph.dur_days / eph.period_days / 3, 1e-3)
    xb, yb, sb = _binned(ph, f.flux, width)
    ax.errorbar(xb, yb, sb, fmt="o", ms=2.5, color="k", lw=0.6)
    d = eph.depth_ppm * 1e-6
    ax.axvline(0.5, color="tab:green", ls=":", lw=0.8)
    if sec and sec.get("phase") is not None:
        ax.axvline(
            sec["phase"] if sec["phase"] < 0.75 else sec["phase"] - 1,
            color="tab:red",
            lw=0.8,
            alpha=0.6,
        )
    lo = np.nanpercentile(yb, 1) if len(yb) else 1 - d
    ax.set_ylim(min(lo, 1 - 1.1 * d), 1 + 0.4 * d + 3 * np.nanmedian(sb) if len(sb) else 1 + d)
    ax.set_xlim(-0.25, 0.75)
    ax.set_xlabel("orbital phase (transit at 0; dotted line at 0.5; red: deepest off-transit dip)")
    ax.set_ylabel("binned flux")
    ax.set_title("Whole orbit: secondary eclipse and variability", fontsize=10)
    return _fig_to_data_uri(fig, dpi)


def plot_events(f: ck.Folded, events: dict[int, ck.Depth], per_sector: dict, dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    eph = f.eph
    if 0 < len(events) <= 12:
        n = len(events)
        cols = min(n, 6)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(2.1 * cols, 2.0 * rows), squeeze=False, sharey=True
        )
        d = eph.depth_ppm * 1e-6
        for ax, (e, dep) in zip(axes.flat, sorted(events.items()), strict=False):
            m = (f.epoch == e) & (np.abs(f.phase) < 2.5 * eph.dur_days)
            ax.plot(f.phase[m] * 24, f.flux[m], ".", ms=1, color="0.6")
            xb, yb, _ = _binned(f.phase[m] * 24, f.flux[m], eph.duration_hours / 6)
            ax.plot(xb, yb, "o", ms=2.5, color="k")
            ax.axhline(1 - d, color="tab:red", lw=0.6, ls=":")
            tt = eph.t0_btjd + e * eph.period_days
            ax.set_title(f"{tt:.2f} ({dep.snr:.1f}σ)", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes[0, 0].set_ylim(1 - 3 * d - 3 * f.sigma / 4, 1 + 1.5 * d + 3 * f.sigma / 4)
        fig.suptitle("Individual transits (BTJD, single-transit SNR)", fontsize=10)
        fig.tight_layout()
        return _fig_to_data_uri(fig, dpi)
    fig, ax = plt.subplots(figsize=(8, 3))
    if per_sector:
        s = sorted(per_sector, key=int)
        v = np.array([per_sector[k][0] for k in s])
        e = np.array([per_sector[k][1] for k in s])
        ax.errorbar([int(k) for k in s], v, e, fmt="o", color="k")
        ax.axhline(eph.depth_ppm, color="tab:red", ls=":")
    ax.set_xlabel("sector")
    ax.set_ylabel("depth (ppm)")
    ax.set_title("Depth per sector", fontsize=10)
    return _fig_to_data_uri(fig, dpi)


def plot_neighbours(rows: list[dict[str, Any]], cfg: Config, dpi: int) -> str:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    r = cfg.vet.neighbour_radius_arcsec
    for k in (1, 2, 3):
        ax.add_patch(
            plt.Circle((0, 0), k * cfg.vet.pixel_arcsec, fill=False, color="0.7", lw=0.6, ls=":")
        )
    for s in rows:
        if s.get("g") is None or "dx" not in s:
            continue
        size = max(4, 2.2 * (19 - s["g"])) ** 1.4
        color = "tab:blue" if s.get("is_target") else "tab:red" if s.get("capable") else "0.55"
        ax.scatter(-s["dx"], s["dy"], s=size, color=color, alpha=0.7, edgecolor="none")
    ax.set_xlim(r, -r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xlabel("ΔRA (″, east left)")
    ax.set_ylabel("ΔDec (″)")
    ax.set_title("Gaia DR3 (red: could make the dip;\ncircles: 1, 2, 3 TESS pixels)", fontsize=9)
    return _fig_to_data_uri(fig, dpi)


def _checks_table(checks: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{c['item']}</td><td>{html.escape(c['name'])}</td>"
        f"<td class='{_STATUS_CLASS.get(c['status'], '')}'>{html.escape(c['status'])}</td>"
        f"<td class='detail'>{html.escape(c['detail'])}</td></tr>"
        for c in checks
    )
    return (
        "<table><tr><th>#</th><th>check</th><th>status</th><th>detail</th></tr>" + rows + "</table>"
    )


def verdict_html(res: dict[str, Any]) -> str:
    return (
        f"<div class='verdict {_VERDICT_CLASS[res['verdict']]}'>Verdict: {res['verdict']} "
        f"({res['n_fail']} fail, {res['n_warn']} warn)</div>"
    )


def update_vet_report(path: Path, res: dict[str, Any]) -> None:
    """Replace the verdict and checklist of an existing report (after `scout fpp`)."""
    import re

    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"<!--verdict-->.*?<!--/verdict-->",
        lambda _: "<!--verdict-->" + verdict_html(res) + "<!--/verdict-->",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"<!--checks-->.*?<!--/checks-->",
        lambda _: "<!--checks-->" + _checks_table(res["checks"]) + "<!--/checks-->",
        text,
        flags=re.S,
    )
    path.write_text(text, encoding="utf-8")


def write_vet_report(
    path: Path,
    cfg: Config,
    row: dict[str, Any],
    star: dict[str, Any],
    res: dict[str, Any],
    gaia_rows: list[dict[str, Any]] | None,
) -> Path:
    dpi = cfg.report.dpi
    f, fit, events = res["_folded"], res["_fit"], res["_events"]
    by_item = {c["item"]: c for c in res["checks"]}
    sec = by_item.get(3, {}).get("values")
    per_sector = by_item.get(4, {}).get("values", {}).get("per_sector", {})
    imgs = [
        ("fold", plot_fold(f, fit, dpi)),
        ("whole orbit", plot_full_phase(f, sec, dpi)),
        ("transits", plot_events(f, events, per_sector, dpi)),
    ]
    eph = res["ephemeris"]
    tic = res["tic"]
    params = [
        ("Period (d)", _fmt(eph["period_days"], 5)),
        ("T0 (BTJD, refitted)", _fmt(eph["t0_btjd"], 4)),
        ("T0 (BJD_TDB)", _fmt(eph["t0_btjd"] + 2457000.0, 4)),
        ("Duration (h)", _fmt(eph["duration_hours"], 2)),
        ("Depth (ppm)", _fmt(eph["depth_ppm"], 0)),
        ("SDE / SNR", f"{_fmt(row.get('sde'), 1)} / {_fmt(row.get('snr'), 1)}"),
        ("Transits with data", str(row.get("n_transits"))),
        (
            "Fit Rp/R*, b, a/R*",
            f"{_fmt(fit.get('rp_rs'), 4)}, {_fmt(fit.get('b'), 2)}, {_fmt(fit.get('a_rs'), 1)}",
        ),
        ("Fit Rp (R⊕)", _fmt(fit.get("rp_rearth"), 2)),
        (
            "Star: Tmag, Teff, R*, M*",
            f"{_fmt(star.get('tmag'), 2)}, {_fmt(star.get('teff_k'), 0)} K, "
            f"{_fmt(star.get('radius_rsun'), 2)} R☉, {_fmt(star.get('mass_msun'), 2)} M☉",
        ),
        ("RA, Dec (deg)", f"{_fmt(star.get('ra_deg'), 5)}, {_fmt(star.get('dec_deg'), 5)}"),
        ("Triage", f"{row.get('triage')} {row.get('match') or ''}"),
    ]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Vetting TIC {tic}.{res['index']}</title>",
        f"<style>{_CSS}{_EXTRA_CSS}</style></head><body>",
        f"<h1>TIC {tic} · signal {res['index']} · P = {eph['period_days']:.5f} d</h1>",
        "<!--verdict-->" + verdict_html(res) + "<!--/verdict-->",
        "<div class='banner'>Automatic checks only. No signal becomes a CTOI without a "
        "human read of this page.</div>",
        "<h2>Checklist (design §6)</h2>",
        "<!--checks-->" + _checks_table(res["checks"]) + "<!--/checks-->",
        "<h2>Signal</h2><table>",
        *(_row(k, v) for k, v in params),
        "</table>",
    ]
    for alt, uri in imgs:
        parts.append(f"<img alt='{alt}' src='{uri}'>")
    if gaia_rows is not None:
        near = sorted((r for r in gaia_rows if "sep_arcsec" in r), key=lambda r: r["sep_arcsec"])[
            :12
        ]
        trs = "".join(
            f"<tr><td>{html.escape(r['source_id'])}{' (target)' if r.get('is_target') else ''}</td>"
            f"<td>{_fmt(r['sep_arcsec'], 1)}</td><td>{_fmt(r.get('g'), 2)}</td>"
            f"<td>{_fmt(r.get('ruwe'), 2)}</td><td>{_fmt(r.get('max_depth_ppm'), 0)}</td>"
            + ("<td class='fail'>yes</td></tr>" if r.get("capable") else "<td></td></tr>")
            for r in near
        )
        parts += [
            "<h2>Gaia DR3 neighbours</h2><div class='grid'><div>",
            f"<img alt='neighbours' src='{plot_neighbours(gaia_rows, cfg, dpi)}'></div><div>",
            "<table><tr><th>source</th><th>sep (″)</th><th>G</th><th>RUWE</th>"
            "<th>max depth if fully eclipsed (ppm)</th><th>could make dip</th></tr>",
            trs,
            "</table></div></div>",
        ]
    numbers = {c["name"]: c["values"] for c in res["checks"] if c["values"]}
    parts += [
        "<h2>Numbers behind the checks</h2>",
        f"<pre>{html.escape(json.dumps(numbers, indent=1, default=str))}</pre>",
        f"<p class='muted'>Search report: <a href='../reports/TIC{tic}.html'>TIC{tic}.html</a></p>",
        "</body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path


_JS = """
document.querySelectorAll('th[data-k]').forEach((th, i) => th.addEventListener('click', () => {
  const tb = th.closest('table').tBodies[0], rows = [...tb.rows], asc = th.dataset.asc !== '1';
  th.dataset.asc = asc ? '1' : '0';
  const v = r => { const c = r.cells[i]; const x = c.dataset.v ?? c.textContent;
                   const n = parseFloat(x); return isNaN(n) ? x : n; };
  rows.sort((a, b) => (v(a) > v(b) ? 1 : v(a) < v(b) ? -1 : 0) * (asc ? 1 : -1));
  rows.forEach(r => tb.appendChild(r));
}));
"""


def write_vet_summary(
    path: Path,
    cfg: Config,
    results: list[dict[str, Any]],
    by_key: dict[tuple[int, int], dict[str, Any]],
    errors: list[dict[str, Any]],
) -> Path:
    order = {"candidate": 0, "review": 1, "fail": 2}
    results = sorted(
        results,
        key=lambda r: (
            order[r["verdict"]],
            r["n_fail"],
            r["n_warn"],
            -float(by_key[(r["tic"], r["index"])]["sde"]),
        ),
    )
    items = sorted({c["item"] for r in results for c in r["checks"]})
    head = ["TIC", "verdict", "P (d)", "depth (ppm)", "Rp (R⊕)", "transits", "SDE", "Tmag"]
    head += [f"{i}" for i in items]
    rows = []
    for r in results:
        b = by_key[(r["tic"], r["index"])]
        status_by_item: dict[int, str] = {}
        for c in r["checks"]:
            prev = status_by_item.get(c["item"])
            if prev is None or STATUS_RANK[c["status"]] < STATUS_RANK[prev]:
                status_by_item[c["item"]] = c["status"]
        fails = "; ".join(
            f"{c['item']}: {c['detail']}" for c in r["checks"] if c["status"] == "fail"
        )
        cells = [
            f"<td data-v='{r['tic']}'><a href='{html.escape(r['report'])}'>"
            f"{r['tic']}.{r['index']}</a></td>",
            f"<td class='{_VERDICT_CLASS[r['verdict']]}' data-v='{order[r['verdict']]}'>"
            f"{r['verdict']}</td>",
            f"<td>{_fmt(b['period_days'], 4)}</td>",
            f"<td>{_fmt(b['depth_ppm'], 0)}</td>",
            f"<td>{_fmt(r['fit'].get('rp_rearth'), 2)}</td>",
            f"<td>{b['n_transits']}</td>",
            f"<td>{_fmt(b['sde'], 1)}</td>",
            f"<td>{_fmt(b.get('tmag'), 2)}</td>",
        ]
        for i in items:
            s = status_by_item.get(i, "")
            cells.append(f"<td class='{_STATUS_CLASS.get(s, '')}'>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + f"<td class='detail'>{html.escape(fails)}</td></tr>")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    legend = "".join(
        f"<li>{i}. {html.escape(name)}</li>"
        for i, name in enumerate(
            (
                "shape and variability",
                "odd/even",
                "secondary",
                "depth consistency",
                "centroid",
                "Gaia neighbours / RUWE",
                "radius",
                "density",
                "spacecraft events",
                "TRICERATOPS (not run)",
                "not already known",
            ),
            start=1,
        )
    )
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Vetting {html.escape(cfg.run.name)}</title>",
        f"<style>{_CSS}{_EXTRA_CSS}</style></head><body>",
        f"<h1>Run {html.escape(cfg.run.name)}: vetting</h1>",
        "<div class='muted'>"
        + " · ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: order[kv[0]]))
        + f" · {len(errors)} errors</div>",
        "<div class='banner'>“candidate” means every automatic check that ran passed. "
        "TRICERATOPS and a human read are still needed before a CTOI.</div>",
        "<h2>Signals</h2><p class='muted'>Columns 1–11 are the design §6 checklist "
        "(worst status where an item has two checks). Click a column to sort.</p>",
        "<div class='wide'><table><thead><tr>",
        *(f"<th data-k='{i}'>{html.escape(h)}</th>" for i, h in enumerate(head)),
        "<th>failures</th></tr></thead><tbody>",
        *rows,
        "</tbody></table></div>",
        f"<h2>Checklist</h2><ol class='muted'>{legend}</ol>",
    ]
    if errors:
        parts.append("<h2>Errors</h2><pre>" + html.escape(json.dumps(errors, indent=1)) + "</pre>")
    parts.append(f"<script>{_JS}</script></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path


STATUS_RANK = {"fail": 0, "warn": 1, "pass": 2, "n/a": 3}
