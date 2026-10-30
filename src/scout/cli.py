"""Command-line interface. Every stage is a subcommand (design §3.9)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from scout import __version__
from scout.config import Config, ConfigError, load_config

# Stages from the design that later milestones fill in.
_PLANNED = {
    "targets": "M3 (batch CVZ target selection from TIC/Gaia)",
    "triage": "M3 (cross-match against TOI/CTOI, NASA Exoplanet Archive, EB catalogue, VSX)",
    "vet": "M5 (centroids, Gaia neighbours, TRICERATOPS)",
    "ctoi": "M6 (ExoFOP CTOI package)",
    "ephem": "M6 (transit windows from the observatory)",
    "inject": "M2 (injection-recovery maps)",
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
    overrides = {"run": {"name": args.run}} if getattr(args, "run", None) else None
    return load_config(args.config, overrides)


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
            print(f"TIC {tic}: FAILED in {until}: {exc}", file=sys.stderr)
            status[tic] = f"failed: {exc}"
            rc = 1
            continue
        status[tic] = "ok"
        _print_summary(summary)
    record_invocation(cfg, args.command, list(args.tic), status)
    return rc


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
    return _cmd_stage(cfg, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
