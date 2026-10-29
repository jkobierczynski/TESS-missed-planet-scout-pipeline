"""Provenance: what produced a result (design §3.3)."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGES = ("scout", "numpy", "scipy", "astropy", "wotan", "transitleastsquares", "astroquery")


def git_version(cwd: Path | None = None) -> str:
    """Git commit of the code, with ``-dirty`` if there are local changes."""
    here = cwd or Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d = {
        "code_version": git_version(),
        "packages": package_versions(),
        "python": platform.python_version(),
        "timestamp_utc": now_utc(),
    }
    if extra:
        d.update(extra)
    return d
