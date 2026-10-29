"""TESS missed-planet scout."""

from importlib import metadata

try:
    __version__ = metadata.version("scout")
except metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+unknown"
