# EpiQ-Map

**Reciprocal-space inspection for epitaxial thin films.**

Tools for working with bounded reciprocal-space maps (RSMs): an interactive
viewer, a processing monitor, and the shared scientific library.

Created by **Ben Gregory** and **Andrej Singer**.

## Installation

Install a published wheel:

```bash
python -m pip install EpiQ-Map
```

The default install includes the napari viewer and pyqtgraph, alongside the
scientific library, reconstruction stack, monitor, and native kernel. Select a
concrete Qt binding through an extra:

```bash
# Recommended Qt 6 installation
python -m pip install "EpiQ-Map[qt6]"

# Optional Qt 5 target
python -m pip install "EpiQ-Map[qt5]"

# Qt 6, Qt for Python
python -m pip install "EpiQ-Map[pyside6]"

# Qt 5, Qt for Python (PySide2, not PySide5)
python -m pip install "EpiQ-Map[pyside2]"
```

QtPy selects the installed binding. When several bindings are installed, set
`QT_API` to `pyqt5`, `pyqt6`, `pyside2`, or `pyside6` before launching.
Current napari supports PyQt5, PyQt6, and PySide6. PySide2 remains usable by
the monitor, but current napari no longer supports it.

Install a development checkout:

```bash
python -m pip install -e ".[test]"
```

The PyPI distribution is named `EpiQ-Map`; its import package is
`epiq_map` because Python package identifiers cannot contain hyphens.

## Running

    rsm_viewer --file scan.nxs
    rsm_monitor --config epiq_monitor.toml

Optional viewer flags (all have sensible defaults): `--u-matrix FILE`,
`--q1-range LO HI`, `--q2-range LO HI`, `--q3-range LO HI`, `--shape N1 N2 N3`,
`--memory-limit-mb MB`.

The default target installs the comparatively heavy `napari` and `pyqtgraph`
dependencies. A concrete Qt binding is selected through a separate extra, as
shown under Installation.

Copy `examples/epiq_monitor.toml` into the working directory and edit the
beamtime paths. The bundled reconstruction modules are used by default; an
external script can still be selected with `--autorsm`.

## Contents

| Path | Purpose |
|---|---|
| `src/epiq_map/rsm_viewer.py` | EpiQ-Map napari viewer. |
| `src/epiq_map/rsm_monitor.py` | Qt processing monitor. |
| `src/epiq_map/rsm_viewer_ctr.py` | High-resolution HKL-rod front end. |
| `src/epiq_map/visualize_rsm_lib.py` | Shared scientific library. |
| `src/epiq_map/hkl_convert/` | autoRSM acquisition stack and native C kernel. |
| `tests/unit/` | Dependency-light unit tests. |
| `tests/integration/` | Native-kernel and acquisition-stack tests. |

## Viewer: the U matrix

`U` is the orientation matrix that aligns the measured data to the crystal
frame. The viewer offers four ways to set it:

- **Calculate U** -- pick a substrate from the dropdown (read from
  `substrate_lattice_constants.txt`) and click **Calculate U**. Peaks are found
  in the loaded volume and indexed against the substrate cell; the status line
  reports the inlier count and RMS.
- **Substrate normal** -- the `Normal` field (e.g. `[0 0 1]`) pins the indexed
  frame so the result is *reproducible*: the normal becomes the out-of-plane
  axis and two in-plane axes are chosen orthogonal to it. Leave blank for an
  unconstrained (run-dependent) orientation.
- **Load U matrix** / **Save U matrix** -- read or write a U as a plain text
  file (`numpy.loadtxt`/`savetxt` format).
- **Use identity** -- skip indexing and view the raw data frame.

## Viewer: oriented Q axes and the region

The **Oriented Q axes** table defines the bounded region to interpolate. Each
row is one output axis:

- **Q1** (horizontal), **Q2** (vertical), **Q3** (napari slider).
- **Direction** -- the crystal direction for that axis in the U-aligned frame
  (default `[1 0 0]`, `[0 1 0]`, `[0 0 1]`). The three must be mutually
  orthogonal. The range below each is cut *along* that (rotated) direction.
- **Q min / Q max / Samples** -- the bounds and grid size for that axis.

Click **Interpolate region** to resample the source onto this grid (in a
background thread). Other controls:

- **Order** -- spline interpolation order for the resampling (0 = nearest,
  1 = trilinear (default), up to 5).
- **Intensity** -- display transform: `Linear`, `log1p`, or `log10(I + 1)`.
- **Use UB matrix (plot in RLU)** -- sample at `U.B*.x` so the axes are in
  **reciprocal lattice units (hkl)** instead of A^-1. Requires a substrate cell
  from **Calculate U**; the axis labels and units update accordingly.
- **Equal axes (cube)** -- render the region as a cube regardless of how
  unequal the Q ranges are. Uncheck for the true physical proportions.

Colormap and 3D rendering mode are set from napari's built-in layer-controls
panel on the left (select the **RSM intensity** layer).

## Viewer: navigation and line cuts

In the 2D slice view:

- **Box zoom** -- left-click-drag a rectangle to zoom to it. Normal
  scroll/drag pan and zoom still work; **Reset view** / **Center camera**
  restore the full view.
- **Hover readout** -- the status bar shows the exact Q (or hkl) position under
  the cursor to three decimals, plus the intensity there.

The **Line cuts** dock (right) takes axis-aligned or arbitrary-line cuts. Plots
**accumulate**: each cut becomes a named curve (`#<file number> <axis>`) in the
plot, with a checklist to toggle curves on/off for comparison, and **Clear
all** to reset. **Save CSV** / **Copy CSV** export the most recent cut.

## Viewer: HKL rods (CTR)

The **RSM rods** dock opens a multi-rod CTR file (the `autoRSM_rods` output --
one `NXdata` per rod). **Open rod file…** lists the rods by `(h, k)`;
**Show rod in viewer** makes the chosen rod the active volume, so the RSM
region / image / line-cut docks all operate on that one rod (it is loaded as
`file.nxs::rod_<h>_<k>`). **Plot rod profile** / **Plot all** overlay the
integrated rod intensity *I(L)* (sum or mean over the H/K window), accumulating
as named curves with **Log Y** and **Save CSV** -- the CTR line shapes side by
side.

## Monitor: CTR rods (high-resolution HKL rods)

For crystal-truncation-rod work the monitor can reconstruct a small,
high-resolution sampling of just the HKL rods instead of one big coarse cube.
Select a group of phi-scan rows (click the **Dataset** column) and press
**CTR rods…**:

1. **Find UB** -- pick a substrate (and surface normal) and re-index every
   selected scan's volume against it; the per-scan orientations are averaged
   into one UB. Two non-blocking helper windows open on demand and can be closed
   (window **X**) without interrupting setup:
   - **Found peaks…** -- a table of the found-and-indexed peaks (`hkl`, `|q|`,
     residual, inlier) so you can see what is there; **Add selected (h, k) as
     rods** turns reflections into rods.
   - **Projection…** -- the data summed along the substrate normal, with the two
     in-plane crystal directions on the axes.
2. **HKL rods** -- set integer **H** and **K** ranges and **Populate pairs** to
   list every `(h, k)`, or **From indexed peaks**; delete rows to make the set
   sparse. **L** spans the full range at high resolution (default 2000 points)
   and each rod is a narrow **(H, K) window** about its center (default ±0.1,
   100 points). The **Coordinates** directions rotate the rod axes to chosen
   crystal directions (like the viewer's oriented Q axes).

`autoRSM_rods.py` then reconstructs every rod in a **single pass** over the CBF
frames (read once; each rod is binned with its own tight, self-clipping grid).
The output is one NeXus file under `{Output Directory}/rod_objects/` with **one
`NXdata` per rod** (`rod_<h>_<k>`, each carrying its own `H`, `K`, `L`,
`counts`, `norm`) plus the averaged `UB`, substrate lattice, and scan list.

## Tests

    python -m pytest tests/unit

`tests/unit/test_rsm_viewer.py` covers the GUI-independent viewer helpers (cuts,
interpolation, U-from-normal, equal-axes scaling, file naming).
`tests/unit/test_rsm_viewer_ctr.py` covers the CTR-rod helpers (pair population, U
averaging, rod-config round trip; the rod-grid/kernel tests need the
the acquisition stack and native kernel). Integration tests live under
`tests/integration/`.

## Build and release

The native `hklBen` library is built by setuptools as part of the wheel:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

No Makefile or hand-copied shared library is needed. Linux wheels use OpenMP,
Windows wheels use MSVC OpenMP, and macOS wheels use a portable serial fallback.
The regular CI workflow in `.github/workflows/build.yml` installs the package
with the `test` and `qt6` extras, checks the console entry points, runs
`tests/unit`, builds the wheel and source distribution, and runs
`twine check`.

Release checklist:

1. Move the intended changes from `## [Unreleased]` into a dated
   `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`. The release workflow
   extracts this section for the GitHub release notes and fails if it is
   missing.
2. Verify the distributions locally:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   ```

3. Commit the changelog and release-ready packaging changes.
4. Create and push a version tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

Pushing the tag runs `.github/workflows/release.yml`. That workflow builds
CPython 3.10-3.14 wheels with `cibuildwheel` for Linux x86-64, Windows AMD64,
Intel macOS, and Apple Silicon macOS, builds and checks the source
distribution, creates a GitHub release with the changelog section as the body,
and publishes all artifacts to PyPI using trusted publishing from the `pypi`
environment.
