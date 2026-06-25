"""EpiQ-Map reciprocal-space mapping tools."""

try:
    from ._version import __version__
except ImportError:  # Source tree before setuptools-scm has generated the file.
    __version__ = "0.0.0"

__all__ = ["__version__"]
