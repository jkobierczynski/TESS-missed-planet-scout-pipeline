"""Run summary: one sortable table of every detection above threshold (design §7).

New signals come first. Each row links to its star report. This is the page a
reviewer starts from; nothing on it is vetted yet.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config
from scout.report.star import _CSS

ORDER = {
    "new": 0,
    "new_on_known_host": 1,
    "neighbour": 2,
    "known_false_positive": 3,
    "known": 4,
}

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

_EXTRA_CSS = """
th[data-k] { cursor: pointer; } th[data-k]:hover { text-decoration: underline; }
.t-new, .t-new_on_known_host { font-weight: 600; }
td { white-space: nowrap; }
.wide { overflow-x: auto; }
"""


def _fmt(x: Any, nd: int) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def _batch_failures(cfg: Config) -> dict[int, dict[str, Any]]:
    path = cfg.run_dir / "batch.jsonl"
    last: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            last[row["tic"]] = row
    return {t: r for t, r in last.items() if r.get("status") != "ok"}


def write_summary(
    path: Path,
    cfg: Config,
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    meta: dict[str, Any],
    n_stars: int,
) -> Path:
    passing = [r for r in rows if r["triage"] != "below_threshold"]
    passing.sort(key=lambda r: (ORDER.get(r["triage"], 9), -float(r["sde"])))
    cols = [
        ("TIC", "tic"),
        ("triage", "triage"),
        ("match", "match"),
        ("P (d)", "period"),
        ("depth (ppm)", "depth"),
        ("dur (h)", "dur"),
        ("Rp (R⊕)", "rp"),
        ("SDE", "sde"),
        ("SNR", "snr"),
        ("transits", "n"),
        ("inv. SDE", "inv"),
        ("odd/even σ", "oe"),
        ("Tmag", "tmag"),
        ("flags", "flags"),
    ]
    body = []
    for r in passing:
        tic = html.escape(str(r["tic"]))
        link = f"<a href='{html.escape(r['report'])}'>{tic}</a>" if r.get("report") else tic
        match = r.get("match") or ""
        if r.get("match_sep_arcsec") and np.isfinite(r["match_sep_arcsec"]):
            match += f" ({r['match_sep_arcsec']:.0f}″)"
        cells = [
            f"<td data-v='{tic}'>{link}</td>",
            f"<td class='t-{r['triage']}' data-v='{ORDER.get(r['triage'], 9)}'>"
            f"{html.escape(r['triage'])}</td>",
            f"<td>{html.escape(match)}</td>",
            f"<td>{_fmt(r['period_days'], 5)}</td>",
            f"<td>{_fmt(r['depth_ppm'], 0)}</td>",
            f"<td>{_fmt(r['duration_hours'], 2)}</td>",
            f"<td>{_fmt(r.get('rp_rearth'), 2)}</td>",
            f"<td>{_fmt(r['sde'], 1)}</td>",
            f"<td>{_fmt(r['snr'], 1)}</td>",
            f"<td>{r['n_transits']}</td>",
            f"<td>{_fmt(r.get('inverted_sde'), 1)}</td>",
            f"<td>{_fmt(r['odd_even_sigma'], 1)}</td>",
            f"<td>{_fmt(r.get('tmag'), 2)}</td>",
            f"<td>{html.escape(r.get('flags') or '')}</td>",
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    failures = _batch_failures(cfg)
    fail_rows = "".join(
        f"<tr><td>{t}</td><td>{html.escape(str(r.get('stage')))}</td>"
        f"<td>{html.escape(str(r.get('error', ''))[:200])}</td></tr>"
        for t, r in sorted(failures.items())
    )
    count_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>"
        for k, v in sorted(counts.items(), key=lambda kv: ORDER.get(kv[0], 9))
    )
    thr = cfg.search.periodic
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Run {html.escape(cfg.run.name)}</title>",
        f"<style>{_CSS}{_EXTRA_CSS}</style></head><body>",
        f"<h1>Run {html.escape(cfg.run.name)}: detections</h1>",
        f"<div class='muted'>{n_stars} stars searched · {len(passing)} detections at "
        f"SDE ≥ {thr.sde_threshold:g}, SNR ≥ {thr.snr_threshold:g} · "
        f"{len(failures)} stars failed</div>",
        "<div class='banner'>Not vetted. Triage only says whether a signal is already "
        "known; every “new” row still needs the §6 checklist.</div>",
        "<h2>Triage</h2><table>",
        count_rows,
        "</table>",
        "<h2>Detections above threshold</h2>",
        "<p class='muted'>Click a column to sort.</p><div class='wide'><table><thead><tr>",
        *(f"<th data-k='{k}'>{html.escape(n)}</th>" for n, k in cols),
        "</tr></thead><tbody>",
        *body,
        "</tbody></table></div>",
    ]
    if failures:
        parts += [
            "<h2>Failed stars</h2><table><tr><th>TIC</th><th>stage</th><th>error</th></tr>",
            fail_rows,
            "</table>",
        ]
    parts += [
        "<h2>Catalogues</h2>",
        f"<pre>{html.escape(json.dumps(meta, indent=2, default=str))}</pre>",
        f"<script>{_JS}</script></body></html>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path
