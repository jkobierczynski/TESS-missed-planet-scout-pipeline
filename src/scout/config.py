"""Run configuration, loaded from a versioned TOML file.

Every threshold, grid and path the pipeline uses lives here (design §3.8, §10).
Unknown keys are rejected so a typo in a config file fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Raised for an invalid configuration file."""


@dataclass(frozen=True)
class RunConfig:
    name: str = "default"
    results_dir: Path = Path("results")
    cache_dir: Path = Path("cache")


@dataclass(frozen=True)
class TargetsConfig:
    # Northern CVZ: stars within ~12° of the north ecliptic pole.
    ecliptic_lat_min_deg: float = 78.0
    tmag_max: float = 13.0
    teff_min_k: float = 2500.0
    teff_max_k: float = 7500.0


@dataclass(frozen=True)
class FetchConfig:
    # MAST provenance names, in order of preference when a sector has several.
    sources: tuple[str, ...] = ("SPOC", "TESS-SPOC", "QLP")
    # Cadences to accept, in seconds.
    cadences_s: tuple[int, ...] = (120, 200, 600, 1800)
    # Flux column to use per source. QLP's KSPSAP_FLUX is already detrended,
    # so the raw SAP_FLUX is taken and detrended here like everything else.
    flux_columns: dict[str, str] = field(
        default_factory=lambda: {
            "SPOC": "PDCSAP_FLUX",
            "TESS-SPOC": "PDCSAP_FLUX",
            "QLP": "SAP_FLUX",
        }
    )
    timeout_s: float = 300.0


@dataclass(frozen=True)
class StitchConfig:
    # Bits of the TESS QUALITY column that reject a cadence. 175 is the SPOC
    # "default" mask (attitude tweak, safe mode, coarse/earth point, desat,
    # manual exclude). Momentum dumps (bit 6, value 32) are handled separately
    # below so they can be padded.
    quality_bitmask: int = 175
    mask_momentum_dumps: bool = True
    momentum_dump_bit: int = 32
    momentum_dump_pad_hours: float = 0.5


@dataclass(frozen=True)
class DetrendConfig:
    method: str = "biweight"
    # Window length = window_factor × max_transit_duration_hours (design §4).
    window_factor: float = 3.0
    max_transit_duration_hours: float = 12.0
    edge_cutoff_days: float = 0.0
    break_tolerance_days: float = 0.5
    # Upper-only sigma clip on the flattened flux: removes flares and cosmic
    # rays, never dips. 0 disables it.
    upper_sigma_clip: float = 4.0


@dataclass(frozen=True)
class PeriodicSearchConfig:
    period_min_days: float = 0.5
    # Upper limit as a fraction of the total baseline: 0.5 ⇒ at least 2 transits.
    period_max_baseline_fraction: float = 0.5
    # Optional absolute cap; 0 disables it.
    period_max_days: float = 0.0
    min_transits: int = 2
    oversampling_factor: int = 3
    duration_grid_step: float = 1.1
    # Provisional. M2 replaces these with values measured from injection-recovery
    # and inverted-light-curve false-alarm rates (design §5).
    sde_threshold: float = 9.0
    snr_threshold: float = 7.0
    max_signals: int = 3
    threads: int = 0  # 0 = all cores
    # Explicit, recorded binning before TLS to keep 2-min data tractable.
    # 0 searches at native cadence. Never applied silently (design §4).
    bin_minutes: float = 10.0
    run_inverted: bool = True


@dataclass(frozen=True)
class SingleSearchConfig:
    # M4. Present now so configs written today stay valid.
    duration_min_hours: float = 1.0
    duration_max_hours: float = 15.0
    snr_threshold: float = 7.0


@dataclass(frozen=True)
class SearchConfig:
    periodic: PeriodicSearchConfig = field(default_factory=PeriodicSearchConfig)
    single: SingleSearchConfig = field(default_factory=SingleSearchConfig)


@dataclass(frozen=True)
class ReportConfig:
    dpi: int = 110


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    source_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return self.run.results_dir / self.run.name

    def to_dict(self) -> dict[str, Any]:
        d = _to_plain(self)
        d.pop("source_path", None)
        return d

    def digest(self) -> str:
        """Short stable hash of the effective configuration (for provenance)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


T = TypeVar("T")


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_plain(x) for x in obj]
    return obj


def _coerce(value: Any, default: Any, where: str) -> Any:
    """Coerce a TOML value to the type of the dataclass default."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if isinstance(default, Path):
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        return Path(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if isinstance(default, dict):
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise ConfigError(f"{where}: expected a table of strings, got {value!r}")
        return dict(value)
    if isinstance(default, tuple):
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        if default:
            return tuple(_coerce(v, default[0], f"{where}[{i}]") for i, v in enumerate(value))
        return tuple(value)
    raise ConfigError(f"{where}: unsupported setting type")  # pragma: no cover


def _build(cls: type[T], data: dict[str, Any], where: str) -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"[{where}] must be a table")
    known = {f.name for f in dataclasses.fields(cls) if f.name != "source_path"}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"[{where}] unknown key(s): {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, value in data.items():
        default = getattr(defaults, name)
        sub = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, sub)
        else:
            kwargs[name] = _coerce(value, default, sub)
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    errors = []
    if not cfg.run.name or any(c in cfg.run.name for c in "/\\"):
        errors.append("run.name must be a non-empty name without path separators")
    d = cfg.detrend
    if d.method not in {"biweight", "median", "mean"}:
        errors.append(f"detrend.method {d.method!r} not supported (biweight, median, mean)")
    if d.window_factor <= 1:
        errors.append("detrend.window_factor must be > 1 or transits get flattened")
    if d.max_transit_duration_hours <= 0:
        errors.append("detrend.max_transit_duration_hours must be > 0")
    p = cfg.search.periodic
    if p.period_min_days <= 0:
        errors.append("search.periodic.period_min_days must be > 0")
    if not 0 < p.period_max_baseline_fraction <= 1:
        errors.append("search.periodic.period_max_baseline_fraction must be in (0, 1]")
    if p.min_transits < 2:
        errors.append("search.periodic.min_transits must be ≥ 2 (single events go to M4)")
    if p.max_signals < 1:
        errors.append("search.periodic.max_signals must be ≥ 1")
    if not cfg.fetch.sources:
        errors.append("fetch.sources must list at least one light-curve source")
    missing = [s for s in cfg.fetch.sources if s not in cfg.fetch.flux_columns]
    if missing:
        errors.append(f"fetch.flux_columns has no entry for: {', '.join(missing)}")
    if errors:
        raise ConfigError("; ".join(errors))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a TOML config file. ``None`` gives the built-in defaults.

    ``overrides`` is a nested dict merged over the file, used by the CLI.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    if overrides:
        data = _merge(data, overrides)
    cfg = _build(Config, data, "")
    if path is not None:
        cfg = dataclasses.replace(cfg, source_path=path)
    _validate(cfg)
    return cfg


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
