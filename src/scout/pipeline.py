"""Per-star stage orchestration with checkpointing.

Layout of a run directory::

    results/<run>/
        run.json                      config, code version, invocations
        stars/TIC<id>/
            fetch.json                chosen MAST files + TIC parameters
            raw.parquet               (synthetic stars only) raw light curve
            stitched.parquet
            flat.parquet
            search.json
        reports/TIC<id>.html

Each stage output has a ``.stamp`` file holding a key built from the stage's
own config section and its input's key. A stage is skipped when its stamp
matches, so reruns are cheap, a crash resumes where it stopped, and changing
e.g. the detrending window reruns detrend and everything after it, but not
fetch or stitch (design §3.6).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scout import fetch as fetch_mod
from scout.config import Config, _to_plain
from scout.detrend import detrend
from scout.lightcurve import LightCurve
from scout.provenance import now_utc, provenance
from scout.report.star import write_star_report
from scout.search.periodic import Detection, PeriodicSearchResult, search_periodic
from scout.stitch import stitch

log = logging.getLogger(__name__)

STAGES = ("fetch", "stitch", "detrend", "search", "report")


class StageError(RuntimeError):
    """A stage failed; ``stage`` names which one."""

    def __init__(self, message: str, stage: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage


def star_dir(cfg: Config, tic: int) -> Path:
    return cfg.run_dir / "stars" / f"TIC{int(tic)}"


def report_path(cfg: Config, tic: int) -> Path:
    return cfg.run_dir / "reports" / f"TIC{int(tic)}.html"


def _key(*parts: Any) -> str:
    blob = json.dumps([_to_plain(p) for p in parts], sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _stamp_path(out: Path) -> Path:
    return out.with_name(out.name + ".stamp")


def _read_stamp(out: Path) -> str | None:
    sp = _stamp_path(out)
    if not out.exists() or not sp.exists():
        return None
    try:
        return json.loads(sp.read_text())["key"]
    except (OSError, ValueError, KeyError):
        return None


def _write_stamp(out: Path, key: str) -> None:
    _stamp_path(out).write_text(json.dumps({"key": key, "written_utc": now_utc()}))


def _run_stage(
    name: str, out: Path, key: str, force: bool, produce: Callable[[], None]
) -> tuple[str, bool]:
    if not force and _read_stamp(out) == key:
        log.info("%s: up to date (%s)", name, out)
        return key, False
    log.info("%s: running -> %s", name, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        produce()
    except StageError as exc:
        exc.stage = exc.stage or name
        raise
    except Exception as exc:
        raise StageError(str(exc), stage=name) from exc
    _write_stamp(out, key)
    return key, True


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


# --- stages -------------------------------------------------------------------


def stage_fetch(cfg: Config, tic: int, force: bool = False) -> str:
    sd = star_dir(cfg, tic)
    raw = sd / "raw.parquet"
    out = sd / "fetch.json"
    if raw.exists():
        # Synthetic or hand-supplied raw light curve: nothing to download.
        key = _key("raw", raw.stat().st_mtime_ns, raw.stat().st_size)
        if _read_stamp(out) != key or force:
            lc = LightCurve.read(raw)
            _write_json(
                out,
                {
                    "tic": tic,
                    "star": lc.meta.get("star", {}),
                    "raw": str(raw),
                    "source": lc.meta.get("source", "raw"),
                },
            )
            _write_stamp(out, key)
        return key
    key = _key("fetch", cfg.fetch, tic)

    def produce() -> None:
        manifest = fetch_mod.fetch(tic, cfg.run.cache_dir, cfg.fetch)
        if not manifest["products"]:
            raise StageError(f"TIC {tic}: MAST has no light curves from {list(cfg.fetch.sources)}")
        fetch_mod.write_manifest(manifest, out)

    return _run_stage("fetch", out, key, force, produce)[0]


def _load_raw(cfg: Config, tic: int) -> tuple[list[LightCurve] | LightCurve, dict[str, Any]]:
    manifest = fetch_mod.read_manifest(star_dir(cfg, tic) / "fetch.json")
    if "raw" in manifest:
        return LightCurve.read(manifest["raw"]), manifest
    return fetch_mod.load_from_manifest(manifest, cfg.fetch), manifest


def stage_stitch(cfg: Config, tic: int, upstream: str, force: bool = False) -> str:
    out = star_dir(cfg, tic) / "stitched.parquet"
    key = _key("stitch", cfg.stitch, upstream)

    def produce() -> None:
        parts, manifest = _load_raw(cfg, tic)
        per_sector = [p.meta for p in parts] if isinstance(parts, list) else [parts.meta]
        lc = stitch(parts, cfg.stitch)
        lc.meta["tic"] = tic
        lc.meta["star"] = manifest.get("star") or lc.meta.get("star") or {}
        sources = sorted({str(m.get("source")) for m in per_sector})
        lc.meta["source"] = "+".join(sources)
        lc.meta["inputs"] = per_sector
        lc.write(out)

    return _run_stage("stitch", out, key, force, produce)[0]


def stage_detrend(cfg: Config, tic: int, upstream: str, force: bool = False) -> str:
    sd = star_dir(cfg, tic)
    out = sd / "flat.parquet"
    key = _key("detrend", cfg.detrend, upstream)

    def produce() -> None:
        detrend(LightCurve.read(sd / "stitched.parquet"), cfg.detrend).write(out)

    return _run_stage("detrend", out, key, force, produce)[0]


def stage_search(cfg: Config, tic: int, upstream: str, force: bool = False) -> str:
    sd = star_dir(cfg, tic)
    out = sd / "search.json"
    key = _key("search", cfg.search.periodic, upstream)

    def produce() -> None:
        res = search_periodic(LightCurve.read(sd / "flat.parquet"), cfg.search.periodic)
        _write_json(
            out,
            {
                "tic": tic,
                "periodic": res.to_dict(),
                "provenance": provenance({"config_digest": cfg.digest()}),
            },
        )

    return _run_stage("search", out, key, force, produce)[0]


def load_search(path: Path) -> PeriodicSearchResult:
    d = json.loads(path.read_text())["periodic"]
    return PeriodicSearchResult(
        detections=[Detection(**x) for x in d["detections"]],
        inverted_sde=d["inverted_sde"],
        period_grid_days=tuple(d["period_grid_days"]),
        n_points=d["n_points"],
        meta=d["meta"],
    )


def stage_report(cfg: Config, tic: int, upstream: str, force: bool = False) -> str:
    sd = star_dir(cfg, tic)
    out = report_path(cfg, tic)
    key = _key("report", cfg.report, upstream)

    def produce() -> None:
        stitched = LightCurve.read(sd / "stitched.parquet")
        flat = LightCurve.read(sd / "flat.parquet")
        res = load_search(sd / "search.json")
        prov = provenance(
            {
                "run": cfg.run.name,
                "config_file": str(cfg.source_path) if cfg.source_path else None,
                "config_digest": cfg.digest(),
                "light_curves": stitched.meta.get("inputs"),
                "stitch": stitched.meta.get("stitch"),
                "detrend": flat.meta.get("detrend"),
                "search": res.meta,
            }
        )
        write_star_report(out, stitched, flat, res, prov, cfg.report)

    return _run_stage("report", out, key, force, produce)[0]


_STAGE_FUNCS = {
    "stitch": stage_stitch,
    "detrend": stage_detrend,
    "search": stage_search,
    "report": stage_report,
}


def run_star(cfg: Config, tic: int, until: str = "report", force: bool = False) -> dict[str, Any]:
    """Run stages for one star up to and including ``until``."""
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}")
    key = stage_fetch(cfg, tic, force=force)
    for name in STAGES[1 : STAGES.index(until) + 1]:
        key = _STAGE_FUNCS[name](cfg, tic, key, force=force)
    summary: dict[str, Any] = {"tic": tic, "until": until}
    sp = star_dir(cfg, tic) / "search.json"
    if STAGES.index(until) >= STAGES.index("search") and sp.exists():
        res = load_search(sp)
        summary["detections"] = [
            {
                k: v
                for k, v in asdict(d).items()
                if k not in {"transit_times_btjd", "per_transit_depth_ppm"}
            }
            for d in res.detections
        ]
        summary["inverted_sde"] = res.inverted_sde
    if until == "report":
        summary["report"] = str(report_path(cfg, tic))
    return summary


def record_invocation(cfg: Config, command: str, tics: list[int], status: dict[int, str]) -> Path:
    """Write/append ``run.json`` (design §7)."""
    path = cfg.run_dir / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = {}
    data["run"] = cfg.run.name
    data["config"] = cfg.to_dict()
    data["config_digest"] = cfg.digest()
    data.setdefault("invocations", []).append(
        {
            "command": command,
            "tics": tics,
            "status": {str(k): v for k, v in status.items()},
            **provenance(),
        }
    )
    _write_json(path, data)
    return path
