"""The ``vet`` stage: run the §6 checklist on every triaged signal (design §6).

Input is ``results/<run>/candidates.parquet`` from triage; every row whose
triage class is in ``vet.triage_classes`` is vetted. Output:

- ``results/<run>/vet/TIC<id>-<n>.json``: every check with its numbers;
- ``results/<run>/vet/TIC<id>-<n>.html``: the vetting report for review;
- ``results/<run>/vetting.parquet`` and ``vet_summary.html``.

Verdicts: ``fail`` if any check fails; ``review`` if any warns; ``candidate``
if everything that ran passed. TRICERATOPS (item 10) does not run here yet, so
no signal is called fully vetted by this stage alone.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scout.config import Config
from scout.lightcurve import LightCurve
from scout.vet import checks as ck

log = logging.getLogger(__name__)

STATUS_ORDER = {"fail": 0, "warn": 1, "n/a": 2, "pass": 3}


def vet_dir(cfg: Config) -> Path:
    return cfg.run_dir / "vet"


def _verdict(results: list[ck.Check]) -> str:
    return _verdict_from_dicts([c.to_dict() for c in results])


def _verdict_from_dicts(checks: list[dict[str, Any]]) -> str:
    s = {c["status"] for c in checks}
    return "fail" if "fail" in s else "review" if "warn" in s else "candidate"


def _other_events(cfg: Config, rows: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    """Observed transit times of every sparse detection in the run, passing or not.

    Systematics put dips on many stars at the same dates, and the weakest
    detections trace them best, so the below-threshold ones are kept too.
    """
    from scout.pipeline import load_search, star_dir

    out: list[tuple[int, float, float]] = []
    limit = 2 * cfg.vet.sparse_events
    for tic in sorted({int(r["tic"]) for r in rows}):
        sd = star_dir(cfg, tic)
        try:
            res = load_search(sd / "search.json")
        except (OSError, ValueError, KeyError):
            continue
        sparse = [d for d in res.detections if 0 < d.n_transits <= limit]
        if not sparse:
            continue
        try:
            t = LightCurve.read(sd / "flat.parquet").time
        except OSError:
            continue
        t = np.sort(t)
        for d in sparse:
            half = d.duration_hours / 48
            for tt in d.transit_times_btjd:
                i = np.searchsorted(t, tt - half)
                if i < len(t) and t[i] < tt + half:
                    out.append((tic, float(tt), d.duration_hours / 24))
    return out


def _fits_files(cfg: Config, tic: int) -> tuple[list[tuple[str, str]], dict[int, str], bool]:
    """Light-curve FITS (path, source) for the star, downloading them again if pruned."""
    from scout import fetch as fetch_mod
    from scout.pipeline import stage_fetch, star_dir

    path = star_dir(cfg, tic) / "fetch.json"
    manifest = fetch_mod.read_manifest(path)
    if "raw" in manifest:
        return [], {}, False
    refetched = False
    if cfg.vet.centroids and any(not Path(p["local_path"]).exists() for p in manifest["products"]):
        stage_fetch(cfg, tic, force=True)
        manifest = fetch_mod.read_manifest(path)
        refetched = True
    files = [(p["local_path"], p["source"]) for p in manifest["products"]]
    sources = {int(p["sector"]): p["source"] for p in manifest["products"]}
    return files, sources, refetched


def vet_signal(
    cfg: Config,
    row: dict[str, Any],
    flat: LightCurve,
    star: dict[str, Any],
    files: list[tuple[str, str]],
    sources: dict[int, str],
    other_events: list[tuple[int, float, float]],
    gaia_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    vc = cfg.vet
    eph = ck.Ephemeris(
        float(row["period_days"]),
        float(row["t0_btjd"]),
        float(row["duration_hours"]),
        float(row["depth_ppm"]),
    )
    f = ck.Folded.build(flat.time, flat.flux, flat.sector, eph, vc)
    events = f.event_depths()
    fit = ck.fit_transit(f, star.get("radius_rsun"), star.get("mass_msun"))
    if fit.get("ok") and abs(fit["dt0_days"]) > 0:
        # Re-centre on the fitted mid-time before measuring anything else.
        eph = dataclasses.replace(eph, t0_btjd=eph.t0_btjd + fit["dt0_days"])
        f = ck.Folded.build(flat.time, flat.flux, flat.sector, eph, vc)
        events = f.event_depths()
    checks: list[ck.Check] = [
        ck.check_shape(fit),
        ck.check_variability(f, vc),
        ck.check_odd_even(f, events, vc),
        ck.check_secondary(f, vc),
        ck.check_depth_consistency(f, events, sources, vc),
        ck.check_single_event_dominance(events, vc),
    ]
    if vc.centroids and files:
        from scout.vet.centroid import check_centroid

        cen = check_centroid(files, eph, star.get("tmag"), vc, cfg.stitch.quality_bitmask)
    else:
        cen = ck.Check(5, "No centroid shift", "n/a", "centroids not checked")
    checks.append(cen)
    if gaia_rows is not None:
        from scout.vet.gaia import check_gaia

        checks.append(check_gaia(gaia_rows, star, eph.depth_ppm, cen.status, vc))
    else:
        checks.append(ck.Check(6, "Gaia neighbours / RUWE", "n/a", "Gaia not queried"))
    checks += [
        ck.check_radius(fit, vc),
        ck.check_density(fit, vc),
        ck.check_spacecraft(
            f, other_events, int(row["tic"]), ck.data_gaps(flat.time, vc.gap_min_days), vc
        ),
        ck.Check(10, "TRICERATOPS FPP", "n/a", "not run yet (needs lightkurve; separate step)"),
    ]
    known = row.get("triage")
    if known == "new":
        checks.append(
            ck.Check(11, "Not already known", "pass", "no TOI, CTOI or planet on or near the star")
        )
    else:
        checks.append(
            ck.Check(11, "Not already known", "warn", f"{known}: {row.get('match') or ''}".strip())
        )
    checks.sort(key=lambda c: c.item)
    verdict = _verdict(checks)
    return {
        "tic": int(row["tic"]),
        "index": int(row["index"]),
        "triage": known,
        "ephemeris": dataclasses.asdict(eph),
        "fit": {k: v for k, v in fit.items() if k not in ("x_binned", "y_binned")},
        "checks": [c.to_dict() for c in checks],
        "verdict": verdict,
        "n_fail": sum(c.status == "fail" for c in checks),
        "n_warn": sum(c.status == "warn" for c in checks),
        "_folded": f,
        "_fit": fit,
        "_events": events,
    }


def _mask_others(
    flat: LightCurve, row: dict[str, Any], all_rows: list[dict[str, Any]]
) -> LightCurve:
    """Remove the star's other passing signals so they do not bias this one."""
    keep = np.ones(len(flat), bool)
    for o in all_rows:
        if o["tic"] != row["tic"] or o["index"] == row["index"] or not o["passes_threshold"]:
            continue
        ph = ck.phase_days(flat.time, o["period_days"], o["t0_btjd"])
        keep &= np.abs(ph) > o["duration_hours"] / 24
    return flat.select(keep)


def vet_star(
    cfg: Config,
    tic: int,
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    other_events: list[tuple[int, float, float]],
) -> list[dict[str, Any]]:
    from scout.pipeline import star_dir
    from scout.report.vet import write_vet_report

    sd = star_dir(cfg, tic)
    star = json.loads((sd / "fetch.json").read_text()).get("star") or {}
    flat_all = LightCurve.read(sd / "flat.parquet")
    files, sources, _ = _fits_files(cfg, tic)
    gaia_rows = None
    if cfg.vet.gaia and star.get("ra_deg") is not None:
        from scout.vet.gaia import query_neighbours

        gaia_rows = query_neighbours(
            star["ra_deg"],
            star["dec_deg"],
            cfg.vet.neighbour_radius_arcsec,
            cfg.vet.gaia_tap_url,
            cfg.run.cache_dir / "gaia" / f"TIC{tic}-{cfg.vet.neighbour_radius_arcsec:g}.json",
        )
    out = []
    for row in rows:
        flat = _mask_others(flat_all, row, all_rows)
        res = vet_signal(cfg, row, flat, star, files, sources, other_events, gaia_rows)
        name = f"TIC{tic}-{row['index']}"
        res["report"] = f"vet/{name}.html"
        write_vet_report(vet_dir(cfg) / f"{name}.html", cfg, row, star, res, gaia_rows)
        public = {k: v for k, v in res.items() if not k.startswith("_")}
        (vet_dir(cfg) / f"{name}.json").write_text(json.dumps(public, indent=2, default=str))
        out.append(public)
    # FITS downloaded again for vetting are kept (unlike after stitch): there are
    # few vetted stars, and `scout fpp` and reruns need them.
    return out


def _vet_worker(cfg, tic, rows, all_rows, other_events) -> list[dict[str, Any]] | dict[str, Any]:
    try:
        return vet_star(cfg, tic, rows, all_rows, other_events)
    except Exception as exc:
        log.exception("TIC %s: vetting failed", tic)
        return {"tic": tic, "error": f"{type(exc).__name__}: {exc}"}


def vet_run(
    cfg: Config, workers: int = 1, tics: list[int] | None = None, progress=None
) -> dict[str, Any]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pyarrow.parquet as pq

    cand = cfg.run_dir / "candidates.parquet"
    if not cand.exists():
        raise FileNotFoundError(f"{cand} not found: run `scout triage` first")
    all_rows = pq.read_table(cand).to_pylist()
    todo = [r for r in all_rows if r["triage"] in cfg.vet.triage_classes]
    if tics:
        todo = [r for r in todo if int(r["tic"]) in set(tics)]
    by_star: dict[int, list[dict[str, Any]]] = {}
    for r in todo:
        by_star.setdefault(int(r["tic"]), []).append(r)
    vet_dir(cfg).mkdir(parents=True, exist_ok=True)
    other = _other_events(cfg, all_rows)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def collect(res: Any, tic: int) -> None:
        if isinstance(res, dict):
            errors.append(res)
        else:
            results.extend(res)
        if progress:
            progress(tic, res, len(results) + len(errors), len(todo))

    if workers <= 1:
        for tic, rows in by_star.items():
            collect(_vet_worker(cfg, tic, rows, all_rows, other), tic)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = {
                pool.submit(_vet_worker, cfg, tic, rows, all_rows, other): tic
                for tic, rows in by_star.items()
            }
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"tic": futs[fut], "error": str(exc)}
                collect(res, futs[fut])

    # The summary covers every vetted signal of the run, not just this call's stars.
    summary = refresh_summary(cfg, errors)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {
        "vetting": str(cfg.run_dir / "vetting.parquet"),
        "summary": str(summary),
        "counts": counts,
        "errors": errors,
    }


def _write_outputs(
    cfg: Config,
    results: list[dict[str, Any]],
    by_key: dict[tuple[int, int], dict[str, Any]],
    errors: list[dict[str, Any]],
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scout.provenance import provenance
    from scout.report.vet import write_vet_summary

    flat_rows = []
    keys = (
        "tic", "index", "triage", "match", "period_days", "depth_ppm", "duration_hours",
        "sde", "snr", "n_transits", "tmag", "teff_k", "radius_rsun",
    )  # fmt: skip
    for res in results:
        base = by_key[(res["tic"], res["index"])]
        row = {k: base.get(k) for k in keys}
        row.update(
            verdict=res["verdict"],
            n_fail=res["n_fail"],
            n_warn=res["n_warn"],
            report=res["report"],
            rp_rearth_fit=res["fit"].get("rp_rearth"),
            b_fit=res["fit"].get("b"),
        )
        for c in res["checks"]:
            row[f"c{c['item']}_{_slug(c['name'])}"] = c["status"]
        row["fails"] = "; ".join(
            f"{c['item']}. {c['detail']}" for c in res["checks"] if c["status"] == "fail"
        )
        flat_rows.append(row)
    out = cfg.run_dir / "vetting.parquet"
    meta = {"provenance": provenance({"config_digest": cfg.digest()}), "errors": errors}
    table = (
        pa.Table.from_pylist(flat_rows)
        if flat_rows
        else pa.table({"tic": pa.array([], pa.int64())})
    )
    table = table.replace_schema_metadata({b"scout": json.dumps(meta, default=str)})
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return write_vet_summary(cfg.run_dir / "vet_summary.html", cfg, results, by_key, errors)


def refresh_summary(cfg: Config, errors: list[dict[str, Any]] | None = None) -> Path:
    """Rebuild vetting.parquet and vet_summary.html from the per-signal JSON files.

    Only signals still in ``vet.triage_classes`` after the latest triage count.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(cfg.run_dir / "candidates.parquet").to_pylist()
    by_key = {
        (int(r["tic"]), int(r["index"])): r for r in rows if r["triage"] in cfg.vet.triage_classes
    }
    results = [json.loads(p.read_text()) for p in sorted(vet_dir(cfg).glob("TIC*.json"))]
    results = [r for r in results if (r["tic"], r["index"]) in by_key]
    return _write_outputs(cfg, results, by_key, errors or [])


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
