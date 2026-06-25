# Changelog

All notable changes to EpiQ-Map are documented here.

## [Unreleased]

- Restructure the project as an installable `src/epiq_map` package.
- Add PyPI metadata, console scripts, setuptools-scm versioning, and
  cross-platform native-kernel wheels.
- Support QtPy-selected PyQt5, PyQt6, PySide2, and PySide6 bindings. The
  default install includes the napari viewer; Qt 5 is available through the
  `qt5` extra.
