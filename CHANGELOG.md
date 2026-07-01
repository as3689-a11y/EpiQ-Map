# Changelog

All notable changes to EpiQ-Map are documented here.

## [Unreleased]

- Add CPython 3.14 CI and release wheels.
- Update GitHub Actions and cibuildwheel versions.

## [0.1.0] - 2026-06-25

Initial packaged release of EpiQ-Map.

### Added

- PyPI packaging with a `src/epiq_map` layout and setuptools-scm versioning.
- `rsm_viewer` and `rsm_monitor` console commands.
- Platform wheels for Linux x86-64, Windows AMD64, Intel macOS, and Apple
  Silicon macOS across CPython 3.10-3.13.
- Modern setuptools build for the native `hklBen` kernel, using OpenMP on
  Linux and Windows and a portable serial build on macOS.
- GitHub Actions build, test, wheel, GitHub Release, and PyPI Trusted
  Publishing workflows.
- Separate unit and integration test directories.
- Packaged substrate lattice data and an example beamtime configuration.
- QtPy compatibility with PyQt5, PyQt6, PySide2, and PySide6.
- `qt5`, `qt6`, `pyside2`, and `pyside6` installation extras.

### Changed

- Napari and pyqtgraph are installed by default with the viewer.
- Source modules and acquisition tools now live under the importable
  `epiq_map` package.
- Bundled helper programs are invoked as package modules instead of through
  repository-relative script paths.
- Beamtime-specific paths were removed from runtime defaults and moved to the
  example configuration.

### Removed

- Legacy Makefile-based native build.
- Obsolete top-level source layout and generated build artifacts.
