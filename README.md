# EpiQ-Map

**Reciprocal-space inspection for epitaxial thin films.**

Tools for working with bounded reciprocal-space maps (RSMs): an interactive
viewer, a processing monitor, and the shared scientific library.

## Contents

| File                            | Purpose                                                        |
|---------------------------------|----------------------------------------------------------------|
| `rsm_viewer.py`                 | EpiQ-Map napari viewer: load a volume, set/compute/save U, orient the Q1/Q2/Q3 axes, interpolate a bounded region, take overlaid line cuts. |
| `rsm_monitor.py`                | Qt monitor: watch an output directory, index against substrates, run reconstructions. |
| `rsm_workflow.py`               | Scientific + persistence helpers used by the monitor.          |
| `Visualize_RSM_Lib.py`          | Shared library: loading, U-matrix indexing, slab transforms.   |
| `substrate_lattice_constants.txt` | Substrate cells (LaAlO3, SrTiO3, LSAT, ...) for U calculation. |
| `make_log_files.py`, `rsm_output_status.py` | Monitor support scripts.                       |
| `HKL_Convert/`                  | Acquisition stack (autoRSM + C kernel); populated separately.  |

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

## Running

    conda activate viz
    python rsm_viewer.py --file scan.nxs
    python rsm_monitor.py

Optional viewer flags (all have sensible defaults): `--u-matrix FILE`,
`--q1-range LO HI`, `--q2-range LO HI`, `--q3-range LO HI`, `--shape N1 N2 N3`,
`--memory-limit-mb MB`.

GUI dependencies: `napari`, `pyqt`, `pyqtgraph`. Install into the `viz` env
with:

    conda install -n viz -c conda-forge napari pyqt pyqtgraph

The monitor's reconstruction calls the production `autoRSM.py` via a configured
path; see `HKL_Convert/README.md`.

## Tests

    conda run -n viz python -m unittest test_rsm_viewer

`test_rsm_viewer.py` covers the GUI-independent helpers (cuts, interpolation,
U-from-normal, equal-axes scaling, file naming). The acquisition tests in
`test_wrapper3.py` (pytest) are skipped until the `HKL_Convert/` stack is in
place.
