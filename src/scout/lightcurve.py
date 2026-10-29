"""The light-curve container every stage reads and writes.

Stages exchange light curves as Parquet files (design §3.5) with one row per
cadence and a JSON metadata blob carrying provenance. Times are BTJD
(BJD_TDB − 2457000) throughout; cadence is kept per row and never resampled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BTJD_OFFSET = 2457000.0

_COLUMNS = ("time", "flux", "flux_err", "quality", "sector", "cadence_s")
_OPTIONAL = ("trend",)


@dataclass
class LightCurve:
    time: np.ndarray  # BTJD, days
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray  # TESS QUALITY bit flags (0 = good)
    sector: np.ndarray  # TESS sector per cadence
    cadence_s: np.ndarray  # exposure cadence per cadence, seconds
    meta: dict[str, Any] = field(default_factory=dict)
    trend: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_err = np.asarray(self.flux_err, dtype=float)
        self.quality = np.asarray(self.quality, dtype=np.int32)
        self.sector = np.asarray(self.sector, dtype=np.int16)
        self.cadence_s = np.asarray(self.cadence_s, dtype=np.int32)
        if self.trend is not None:
            self.trend = np.asarray(self.trend, dtype=float)
        n = len(self.time)
        for name in (*_COLUMNS, *_OPTIONAL):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(f"column {name!r} has length {len(arr)}, expected {n}")

    def __len__(self) -> int:
        return len(self.time)

    @property
    def tic(self) -> int | None:
        tic = self.meta.get("tic")
        return int(tic) if tic is not None else None

    @property
    def sectors(self) -> list[int]:
        return sorted(int(s) for s in np.unique(self.sector))

    @property
    def baseline_days(self) -> float:
        return float(np.nanmax(self.time) - np.nanmin(self.time)) if len(self) else 0.0

    def select(self, mask: np.ndarray) -> LightCurve:
        return LightCurve(
            time=self.time[mask],
            flux=self.flux[mask],
            flux_err=self.flux_err[mask],
            quality=self.quality[mask],
            sector=self.sector[mask],
            cadence_s=self.cadence_s[mask],
            meta=dict(self.meta),
            trend=None if self.trend is None else self.trend[mask],
        )

    def sort(self) -> LightCurve:
        return self.select(np.argsort(self.time, kind="stable"))

    @staticmethod
    def concatenate(parts: list[LightCurve], meta: dict[str, Any] | None = None) -> LightCurve:
        if not parts:
            raise ValueError("nothing to concatenate")
        with_trend = all(p.trend is not None for p in parts)
        lc = LightCurve(
            time=np.concatenate([p.time for p in parts]),
            flux=np.concatenate([p.flux for p in parts]),
            flux_err=np.concatenate([p.flux_err for p in parts]),
            quality=np.concatenate([p.quality for p in parts]),
            sector=np.concatenate([p.sector for p in parts]),
            cadence_s=np.concatenate([p.cadence_s for p in parts]),
            meta=dict(meta if meta is not None else parts[0].meta),
            trend=np.concatenate([p.trend for p in parts]) if with_trend else None,  # type: ignore[misc]
        )
        return lc.sort()

    # --- I/O -----------------------------------------------------------------

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = {name: getattr(self, name) for name in _COLUMNS}
        if self.trend is not None:
            cols["trend"] = self.trend
        table = pa.table(cols)
        table = table.replace_schema_metadata({b"scout": json.dumps(self.meta, default=str)})
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)  # atomic: a crash never leaves a half-written stage output
        return path

    @classmethod
    def read(cls, path: str | Path) -> LightCurve:
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(b"scout", b"{}")
        cols = {name: table.column(name).to_numpy() for name in table.column_names}
        return cls(
            time=cols["time"],
            flux=cols["flux"],
            flux_err=cols["flux_err"],
            quality=cols["quality"],
            sector=cols["sector"],
            cadence_s=cols["cadence_s"],
            trend=cols.get("trend"),
            meta=json.loads(raw),
        )


def bin_per_sector(lc: LightCurve, bin_minutes: float) -> LightCurve:
    """Bin each sector onto its own ``bin_minutes`` grid.

    Only used where a stage asks for it explicitly and records it. Sectors whose
    native cadence is already as long as the bin are passed through unchanged.
    """
    if bin_minutes <= 0:
        return lc
    width = bin_minutes / 1440.0
    parts = []
    for s in lc.sectors:
        part = lc.select(lc.sector == s)
        if np.median(part.cadence_s) >= bin_minutes * 60:
            parts.append(part)
            continue
        # Small epsilon so cadences exactly on a bin edge don't flip bins on rounding.
        idx = np.floor((part.time - part.time[0]) / width + 1e-6).astype(np.int64)
        uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
        nb = len(uniq)
        w = 1.0 / np.square(part.flux_err)
        wtot = np.bincount(inv, weights=w, minlength=nb)
        parts.append(
            LightCurve(
                time=np.bincount(inv, weights=part.time, minlength=nb) / counts,
                flux=_wmean(part.flux, w, inv, wtot),
                flux_err=1.0 / np.sqrt(wtot),
                quality=np.zeros(nb, dtype=np.int32),
                sector=np.full(nb, s),
                cadence_s=np.full(nb, round(bin_minutes * 60)),
                trend=None if part.trend is None else _wmean(part.trend, w, inv, wtot),
            )
        )
    out = LightCurve.concatenate(parts, meta=dict(lc.meta))
    out.meta["binned_minutes"] = bin_minutes
    return out


def _wmean(x: np.ndarray, w: np.ndarray, inv: np.ndarray, wtot: np.ndarray) -> np.ndarray:
    return np.bincount(inv, weights=x * w, minlength=len(wtot)) / wtot
