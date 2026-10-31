"""Command-line interface. Every stage is a subcommand (design §3.9)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scout import __version__
from scout.config import Config, ConfigError, load_config

# Stages from the design that later milestones fill in.
_PLANNED = {
    "ctoi": "M6 (ExoFOP CTOI package)",
    "ephem": "M6 (transit windows from the observatory)",
}

log = logging.getLogger("scout")


def _parse_sectors(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("no sectors given")
    return tuple(out)


def _parse_tic(text: str) -> int:
    t = text.strip().upper().removeprefix("TIC").strip(" -_")
    try:
        return int(t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a TIC ID: {text!r}") from exc


def _load(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if getattr(args, "run", None):
        overrides["run"] = {"name": args.run}
    inject = {}
    if getattr(args, "n", None) is not None:
        inject["n_injections"] = args.n
    if getattr(args, "scrambles", None) is not None:
        inject["n_scrambles"] = args.scrambles
    if inject:
        overrides["inject"] = inject
    if getattr(args, "max", None) is not None:
        overrides["targets"] = {"max_targets": args.max}
    return load_config(args.config, overrides or None)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, help="TOML config file (default: built-ins)")
    p.add_argument("--run", help="override run.name from the config")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="scout",
        description="Find, vet and package TESS transit candidates missed by SPOC and QLP.",
    )
    ap.add_argument("--version", action="version", version=f"scout {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("config", help="show or check the effective configuration")
    _add_common(p)
    p.add_argument("action", choices=("show", "check"), nargs="?", default="show")

    p = sub.add_parser("synth", help="write a synthetic star into the run as raw input")
    _add_common(p)
    from scout.synthetic import PRESETS

    p.add_argument("--preset", choices=PRESETS, default="sub-neptune")
    p.add_argument(
        "--tic",
        type=_parse_tic,
        default=999_000_001,
        help="fake TIC ID to file it under (default 999000001)",
    )
    p.add_argument(
        "--sectors", type=_parse_sectors, default=(14, 15, 16), help="e.g. 14-16 or 14,15,21"
    )
    p.add_argument("--seed", type=int, default=0)

    for name, help_ in (
        ("fetch", "download light curves from MAST into the cache"),
        ("stitch", "quality-mask, normalise and join sectors"),
        ("detrend", "flatten with wotan"),
        ("search", "periodic TLS search (+ inverted-flux check)"),
        ("report", "write the per-star HTML report"),
        ("run", "run every stage that is out of date, up to the report"),
    ):
        p = sub.add_parser(name, help=help_)
        _add_common(p)
        p.add_argument("tic", nargs="+", type=_parse_tic, help="TIC ID(s)")
        p.add_argument("--force", action="store_true", help="rerun even if up to date")

    p = sub.add_parser("targets", help="select target stars from the TIC (writes targets.csv)")
    _add_common(p)
    p.add_argument("--max", type=int, help="override targets.max_targets")

    p = sub.add_parser(
        "batch",
        help="run every star in targets.csv (resumable, parallel)",
        description="Runs fetch → report for each target, skipping finished stars and "
        "retrying failed ones. One line per star goes to results/<run>/batch.jsonl.",
    )
    _add_common(p)
    p.add_argument("--targets", type=Path, help="CSV with a tic column (default: the run's)")
    p.add_argument("--workers", type=int, default=1, help="stars processed at once")
    p.add_argument("--limit", type=int, help="only the first N targets")
    p.add_argument(
        "--until", choices=("fetch", "stitch", "detrend", "search", "report"), default="report"
    )

    p = sub.add_parser(
        "survey",
        help="the whole survey in one go: targets, search, triage, vetting (rerun to resume)",
        description="Runs `targets` (once), `batch` with automatic retries of failed stars, "
        "`triage` and `vet`. Interrupt it any time; the same command continues where it "
        "stopped.",
    )
    _add_common(p)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="stars processed at once (default: half the CPU cores)",
    )
    p.add_argument("--limit", type=int, help="only the first (brightest) N targets")
    p.add_argument("--retries", type=int, default=2, help="extra passes over failed stars")
    p.add_argument("--new-targets", action="store_true", help="query the TIC again")
    p.add_argument("--skip-vet", action="store_true", help="stop after triage")

    p = sub.add_parser(
        "triage", help="cross-match detections with known objects; write candidates + summary"
    )
    _add_common(p)

    p = sub.add_parser(
        "vet",
        help="run the vetting checklist on triaged signals; write reports + vet_summary.html",
        description="Vets every signal whose triage class is in vet.triage_classes. "
        "Needs `scout triage` first. Downloads pruned light-curve FITS again for centroids "
        "and queries Gaia DR3 at MAST.",
    )
    _add_common(p)
    p.add_argument("tic", nargs="*", type=_parse_tic, help="only these stars")
    p.add_argument("--workers", type=int, default=1, help="stars vetted at once")

    p = sub.add_parser(
        "fpp",
        help="TRICERATOPS false-positive probabilities for signals that survived `scout vet`",
        description="Needs the [fpp] extra (TRICERATOPS), which installs on Python ≤ 3.12 only.",
    )
    _add_common(p)
    p.add_argument("tic", nargs="*", type=_parse_tic, help="only these stars (any verdict)")

    p = sub.add_parser(
        "inject",
        help="injection-recovery and null searches on host stars (resumable)",
        description="Fetch and stitch each host if needed, then inject transits and run "
        "null searches. Hosts should have no known planet or TOI.",
    )
    _add_common(p)
    p.add_argument("tic", nargs="+", type=_parse_tic, help="host TIC ID(s)")
    p.add_argument("--n", type=int, help="override inject.n_injections")
    p.add_argument("--scrambles", type=int, help="override inject.n_scrambles")

    p = sub.add_parser(
        "calibrate", help="pool a run's injections and nulls, choose thresholds, write report"
    )
    _add_common(p)

    for name, when in _PLANNED.items():
        p = sub.add_parser(name, help=f"not yet implemented: {when}")
        p.add_argument("rest", nargs=argparse.REMAINDER)
    return ap


def _cmd_config(cfg: Config, action: str) -> int:
    if action == "check":
        print(f"ok: run '{cfg.run.name}', digest {cfg.digest()}")
    else:
        print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def _cmd_synth(cfg: Config, args: argparse.Namespace) -> int:
    from scout.pipeline import star_dir
    from scout.synthetic import generate, preset

    star = preset(args.preset, tic=args.tic, sectors=args.sectors, seed=args.seed)
    lc = generate(star)
    out = lc.write(star_dir(cfg, args.tic) / "raw.parquet")
    print(f"wrote {out} ({len(lc)} cadences, sectors {lc.sectors}, preset {args.preset})")
    return 0


def _cmd_stage(cfg: Config, args: argparse.Namespace) -> int:
    from scout.pipeline import record_invocation, run_star

    until = "report" if args.command == "run" else args.command
    status: dict[int, str] = {}
    rc = 0
    for tic in args.tic:
        try:
            summary = run_star(cfg, tic, until=until, force=args.force)
        except Exception as exc:
            log.debug("TIC %s failed", tic, exc_info=True)
            stage = getattr(exc, "stage", None) or until
            print(f"TIC {tic}: FAILED in {stage}: {exc}", file=sys.stderr)
            status[tic] = f"failed: {exc}"
            rc = 1
            continue
        status[tic] = "ok"
        _print_summary(summary)
    record_invocation(cfg, args.command, list(args.tic), status)
    return rc


def _cmd_inject(cfg: Config, args: argparse.Namespace) -> int:
    from scout.inject import run_star_injections
    from scout.lightcurve import LightCurve
    from scout.pipeline import record_invocation, stage_fetch, stage_stitch, star_dir

    def progress(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    status: dict[int, str] = {}
    rc = 0
    for tic in args.tic:
        try:
            upstream = stage_stitch(cfg, tic, stage_fetch(cfg, tic))
            stitched = LightCurve.read(star_dir(cfg, tic) / "stitched.parquet")
            out = run_star_injections(cfg, tic, stitched, upstream, progress=progress)
        except Exception as exc:
            log.debug("TIC %s failed", tic, exc_info=True)
            stage = getattr(exc, "stage", None) or "inject"
            print(f"TIC {tic}: FAILED in {stage}: {exc}", file=sys.stderr)
            status[tic] = f"failed: {exc}"
            rc = 1
            continue
        status[tic] = "ok"
        print(f"TIC {tic}: {out['injections']} injections, {out['nulls']} nulls in {out['dir']}")
    record_invocation(cfg, "inject", list(args.tic), status)
    return rc


def _cmd_targets(cfg: Config) -> int:
    from scout.targets import query_targets, targets_path, write_targets

    rows = query_targets(cfg.targets, cfg.fetch.timeout_s)
    out = write_targets(rows, targets_path(cfg))
    if rows:
        print(f"{len(rows)} targets, Tmag {rows[0]['tmag']:.2f}–{rows[-1]['tmag']:.2f} -> {out}")
    else:
        print(f"no targets match the cuts -> {out}")
    return 0


def _batch_progress():
    import time

    start = time.monotonic()

    def progress(row: dict, done: int, total: int) -> None:
        rate = (time.monotonic() - start) / done
        eta_h = rate * (total - done) / 3600
        what = (
            f"{row.get('n_passing', 0)} above threshold"
            if row["status"] == "ok"
            else f"FAILED in {row.get('stage')}: {row.get('error')}"
        )
        print(
            f"[{done}/{total}] TIC {row['tic']}: {what} ({row.get('seconds', '?')} s; "
            f"ETA {eta_h:.1f} h)",
            file=sys.stderr,
            flush=True,
        )

    return progress


def _cmd_batch(cfg: Config, args: argparse.Namespace) -> int:
    from scout.pipeline import record_invocation, run_batch
    from scout.targets import read_targets, targets_path

    path = args.targets or targets_path(cfg)
    if not Path(path).exists():
        print(f"no target list at {path}: run `scout targets` first", file=sys.stderr)
        return 2
    tics = [r["tic"] for r in read_targets(path)]
    if args.limit:
        tics = tics[: args.limit]
    res = run_batch(cfg, tics, until=args.until, workers=args.workers, progress=_batch_progress())
    status = {t: r["status"] for t, r in res.items()}
    record_invocation(cfg, "batch", tics, status)
    failed = sum(s != "ok" for s in status.values())
    print(f"{len(tics) - failed} ok, {failed} failed; log: {cfg.run_dir / 'batch.jsonl'}")
    return 1 if failed else 0


def _cmd_survey(cfg: Config, args: argparse.Namespace) -> int:
    """targets → batch → triage → vet, each step resumable; rerun to continue."""
    from scout.pipeline import record_invocation, run_batch
    from scout.targets import read_targets, targets_path

    def step(text: str) -> None:
        print(f"\n== {text}", file=sys.stderr, flush=True)

    try:
        path = targets_path(cfg)
        if not path.exists() or args.new_targets:
            step("1/4 targets: querying the TIC at MAST")
            if _cmd_targets(cfg) != 0:
                return 1
        else:
            step(f"1/4 targets: reusing {path}")
        tics = [r["tic"] for r in read_targets(path)]
        if args.limit:
            tics = tics[: args.limit]
        step(
            f"2/4 search: {len(tics)} stars, {args.workers} at a time (finished stars are skipped)"
        )
        status: dict[int, str] = {}
        for attempt in range(1 + args.retries):
            todo = [t for t in tics if status.get(t, "failed") != "ok"]
            if not todo:
                break
            if attempt:
                step(f"2/4 search: retrying {len(todo)} failed stars (MAST hiccups are common)")
            res = run_batch(cfg, todo, workers=args.workers, progress=_batch_progress())
            status.update({t: r["status"] for t, r in res.items()})
        record_invocation(cfg, "survey", tics, status)
        failed = [t for t in tics if status.get(t) != "ok"]
        if failed:
            print(f"{len(failed)} stars still failed; see {cfg.run_dir / 'batch.jsonl'}")
        step("3/4 triage: cross-matching with TOI, CTOI and confirmed planets")
        _cmd_triage(cfg)
        if args.skip_vet:
            return 0
        step("4/4 vetting: the design §6 checklist on every new signal")
        code = _cmd_vet(cfg, argparse.Namespace(tic=[], workers=args.workers))
    except KeyboardInterrupt:
        print(
            "\nStopped. Nothing finished is lost: run the same command again to continue.",
            file=sys.stderr,
        )
        return 130
    print(
        f"\nDone. Open {cfg.run_dir / 'vet_summary.html'} (vetting) and "
        f"{cfg.run_dir / 'summary.html'} (all detections)."
    )
    return code


def _cmd_triage(cfg: Config) -> int:
    from scout.triage import triage_run

    res = triage_run(cfg)
    counts = ", ".join(f"{k} {v}" for k, v in sorted(res["counts"].items()))
    print(f"{res['n_stars']} stars, {counts or 'no detections'}")
    print(f"summary: {res['summary']}")
    return 0


def _cmd_vet(cfg: Config, args: argparse.Namespace) -> int:
    from scout.vet.run import vet_run

    def progress(tic: int, res: object, done: int, total: int) -> None:
        if isinstance(res, dict):
            what = f"ERROR {res.get('error')}"
        else:
            what = ", ".join(f"{r['index']}: {r['verdict']}" for r in res)  # type: ignore[attr-defined]
        print(f"[{done}/{total}] TIC {tic}: {what}", file=sys.stderr, flush=True)

    try:
        res = vet_run(cfg, workers=args.workers, tics=args.tic or None, progress=progress)
    except FileNotFoundError as exc:
        print(f"vet: {exc}", file=sys.stderr)
        return 2
    counts = ", ".join(f"{k} {v}" for k, v in sorted(res["counts"].items()))
    print(f"{counts or 'nothing vetted'}; {len(res['errors'])} errors")
    print(f"summary: {res['summary']}")
    return 1 if res["errors"] else 0


def _cmd_fpp(cfg: Config, args: argparse.Namespace) -> int:
    try:
        import triceratops  # noqa: F401

        from scout.vet.fpp import fpp_run
    except ImportError:
        print(
            "fpp: TRICERATOPS is not installed. It needs Python ≤ 3.12: "
            'pip install -e ".[mast,models,fpp]" in such an environment.',
            file=sys.stderr,
        )
        return 2

    def progress(res: dict, check: object, done: int, total: int) -> None:
        what = getattr(check, "detail", "ERROR")
        print(f"[{done}/{total}] TIC {res['tic']}.{res['index']}: {what}", file=sys.stderr)

    res = fpp_run(cfg, tics=args.tic or None, progress=progress)
    print(f"{len(res['done'])} done, {len(res['errors'])} errors; summary: {res['summary']}")
    return 1 if res["errors"] else 0


def _cmd_calibrate(cfg: Config) -> int:
    from scout.inject import calibrate

    try:
        res = calibrate(cfg)
    except ValueError as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 1
    ch, cur = res["chosen"], res["current"]
    print(
        f"chosen:     SDE ≥ {ch['sde']:g}, SNR ≥ {ch['snr']:g}  recovers {ch['recovery']:.1%}, "
        f"false alarms {ch['false_alarm']:.1%} of {ch['n_nulls']} null searches"
    )
    print(
        f"config now: SDE ≥ {cur['sde']:g}, SNR ≥ {cur['snr']:g}  recovers {cur['recovery']:.1%}, "
        f"false alarms {cur['false_alarm']:.1%}"
    )
    if ch.get("warning"):
        print(f"warning: {ch['warning']}")
    print(f"report: {res['report']}")
    return 0


def _print_summary(s: dict) -> None:
    line = f"TIC {s['tic']}: {s['until']} done"
    print(line)
    for d in s.get("detections", []):
        mark = "*" if d["passes_threshold"] else " "
        print(
            f"  {mark} #{d['index']} P={d['period_days']:.5f} d  T0={d['t0_btjd']:.4f}  "
            f"depth={d['depth_ppm']:.0f} ppm  dur={d['duration_hours']:.2f} h  "
            f"SDE={d['sde']:.1f}  SNR={d['snr']:.1f}  n={d['n_transits']}"
            + (f"  [{', '.join(d['flags'])}]" if d["flags"] else "")
        )
    if "inverted_sde" in s and s["inverted_sde"] is not None:
        print(f"    inverted-flux SDE {s['inverted_sde']:.1f}")
    if "report" in s:
        print(f"  report: {s['report']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.command in _PLANNED:
        print(
            f"scout {args.command}: not implemented yet, planned for {_PLANNED[args.command]}",
            file=sys.stderr,
        )
        return 2
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.command == "config":
        return _cmd_config(cfg, args.action)
    if args.command == "synth":
        return _cmd_synth(cfg, args)
    if args.command == "inject":
        return _cmd_inject(cfg, args)
    if args.command == "calibrate":
        return _cmd_calibrate(cfg)
    if args.command == "targets":
        return _cmd_targets(cfg)
    if args.command == "batch":
        return _cmd_batch(cfg, args)
    if args.command == "survey":
        return _cmd_survey(cfg, args)
    if args.command == "triage":
        return _cmd_triage(cfg)
    if args.command == "vet":
        return _cmd_vet(cfg, args)
    if args.command == "fpp":
        return _cmd_fpp(cfg, args)
    return _cmd_stage(cfg, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
