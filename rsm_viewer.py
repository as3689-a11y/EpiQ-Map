#!/usr/bin/env python3
"""EpiQ-Map: interactive napari viewer for bounded reciprocal-space maps.

Reciprocal-space inspection for epitaxial thin films.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from Visualize_RSM_Lib import (compute_U_from_substrate, load_U_matrix,
                               load_rsm, save_U_matrix, transform_slab)

# Substrate lattice file lives alongside this script in the EpiQ-Map suite.
LATTICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "substrate_lattice_constants.txt")


AXIS_NAMES = ("H", "K", "L")

# Q axes shown to the user, in range/UI order: Q1 horizontal, Q2 vertical,
# Q3 slider. Each is a crystal direction in the U-aligned frame.
QAXIS_NAMES = ("Q1", "Q2", "Q3")

# Same names in volume-axis order (axis0=Q3 slider, axis1=Q2, axis2=Q1).
VOLUME_AXIS_NAMES = ("Q3", "Q2", "Q1")

# napari needs the slider as volume axis 0 and the two displayed axes as
# 1 (vertical) and 2 (horizontal). So Q1/Q2/Q3 map to volume axes as below.
QAXIS_TO_VOLUME = (2, 1, 0)   # Q1->axis2 (horiz), Q2->axis1 (vert), Q3->axis0

@dataclass
class RegionModel:
    volume: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    U: np.ndarray = field(default_factory=lambda: np.eye(3))
    source: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.volume = np.asarray(self.volume, dtype=np.float32)
        self.axes = tuple(np.asarray(a, dtype=float) for a in self.axes)
        self.U = validate_u_matrix(self.U)
        if self.volume.ndim != 3 or self.volume.shape != tuple(map(len, self.axes)):
            raise ValueError("volume shape must match the H, K, and L axes")

    @property
    def scale(self) -> tuple[float, float, float]:
        return tuple(float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in self.axes)

    @property
    def translate(self) -> tuple[float, float, float]:
        return tuple(float(a[0]) for a in self.axes)


def load_substrate_names(path: str = LATTICE_FILE) -> list[str]:
    """Return the substrate formulas listed in the lattice file, in order."""
    import re
    names: list[str] = []
    try:
        with open(path) as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                match = re.match(r"([A-Za-z0-9_.\-()]+)", line)
                if match and match.group(1) not in names:
                    names.append(match.group(1))
    except OSError:
        pass
    return names


def validate_u_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("U matrix must be a finite 3x3 matrix")
    return matrix


def orientation_matrix(axis0: Any, axis1: Any, axis2: Any) -> np.ndarray:
    """Return normalized crystal directions as output volume-axis columns.

    The three arguments are the directions for output volume axes 0, 1, 2
    respectively; they must be mutually orthogonal. The returned matrix has
    them as its columns.
    """
    directions = np.asarray((axis0, axis1, axis2), dtype=float)
    if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
        raise ValueError("orientation directions must be finite 3-vectors")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("orientation directions cannot be zero")
    directions = directions / lengths[:, None]
    products = directions @ directions.T
    if not np.allclose(products, np.eye(3), atol=1e-6):
        raise ValueError("orientation directions must be mutually orthogonal")
    return directions.T


def oriented_u_matrix(U: Any, axis0: Any, axis1: Any, axis2: Any) -> np.ndarray:
    """Compose U with an output orientation frame.

    ``U`` maps the new grid into the data frame (data is sampled at ``U @ x``).
    ``orientation_matrix`` builds a rotation whose columns are the crystal
    directions for output volume axes 0, 1, 2, so a unit step along each
    resampled axis runs along the requested direction.
    """
    R = orientation_matrix(axis0, axis1, axis2)
    return validate_u_matrix(np.asarray(U, dtype=float) @ R)


def parse_direction(text: str) -> np.ndarray:
    cleaned = text.strip().strip("[]()").replace(",", " ")
    values = np.fromstring(cleaned, sep=" ")
    if len(values) != 3:
        raise ValueError(f"expected three direction indices, got {text!r}")
    return values


def format_direction(direction: Any) -> str:
    values = np.asarray(direction, dtype=float)
    parts = [str(int(value)) if float(value).is_integer() else f"{value:g}"
             for value in values]
    return "[" + " ".join(parts) + "]"


def source_tag(path: str) -> str:
    """Short label for a source file, preferring its trailing scan number.

    e.g. '/data/scan_0042.nxs' -> '0042', 'rsm.nxs' -> 'rsm'. Used to name
    overlaid line cuts so curves from different files are distinguishable.
    """
    import re
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if not stem:
        return "?"
    match = re.search(r"(\d+)\s*$", stem)
    return match.group(1) if match else stem


def validate_region(bounds: list[tuple[float, float]], shape: tuple[int, int, int]) -> None:
    if len(bounds) != 3 or any(not np.isfinite(pair).all() or pair[0] >= pair[1]
                               for pair in map(np.asarray, bounds)):
        raise ValueError("each range must contain finite, increasing limits")
    if len(shape) != 3 or any(int(n) <= 0 for n in shape):
        raise ValueError("H, K, and L sample counts must be positive")


def make_region_axes(bounds: list[tuple[float, float]],
                     shape: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    validate_region(bounds, shape)
    return tuple(np.linspace(lo, hi, int(n), dtype=float)
                 for (lo, hi), n in zip(bounds, shape))


def estimated_megabytes(shape: tuple[int, int, int], working_factor: float = 8.0) -> float:
    """Conservative peak allocation estimate for transform_slab's grids."""
    return float(np.prod(shape, dtype=np.int64) * 4 * working_factor / 2**20)


def intensity_view(volume: np.ndarray, mode: str) -> np.ndarray:
    data = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)
    if mode == "Linear":
        return data
    data = np.maximum(data, 0)
    if mode == "log1p":
        return np.log1p(data, dtype=np.float32)
    if mode == "log10(I + 1)":
        return np.log10(data + np.float32(1.0), dtype=np.float32)
    raise ValueError(f"unknown intensity mode: {mode}")


def _window_indices(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    if width < 0:
        raise ValueError("integration widths cannot be negative")
    if width == 0:
        return np.array([int(np.argmin(np.abs(axis - center)))])
    indices = np.flatnonzero(np.abs(axis - center) <= width / 2)
    return indices if len(indices) else np.array([int(np.argmin(np.abs(axis - center)))])


def axis_aligned_cut(model: RegionModel, scan_axis: int,
                     fixed: tuple[float, float], widths: tuple[float, float],
                     reduction: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    if scan_axis not in (0, 1, 2):
        raise ValueError("scan_axis must be 0, 1, or 2")
    other = [i for i in range(3) if i != scan_axis]
    selected = [_window_indices(model.axes[i], c, w)
                for i, c, w in zip(other, fixed, widths)]
    index = [np.arange(model.volume.shape[i]) if i == scan_axis else
             selected[other.index(i)] for i in range(3)]
    block = model.volume[np.ix_(*index)]
    reduce_axes = tuple(i for i in range(3) if i != scan_axis)
    func = np.nansum if reduction == "sum" else np.nanmean
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")
    return model.axes[scan_axis].copy(), np.asarray(func(block, axis=reduce_axes))


def arbitrary_line_cut(model: RegionModel, start: np.ndarray, end: np.ndarray,
                       samples: int = 300, transverse_width: float = 0.0,
                       reduction: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Sample a physical-coordinate line, optionally averaging transverse offsets."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if start.shape != (3,) or end.shape != (3,) or samples < 2:
        raise ValueError("line endpoints must be 3D and samples must be at least 2")
    direction = end - start
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length == 0:
        raise ValueError("line endpoints must be finite and distinct")
    t = np.linspace(0, 1, int(samples))
    points = start[:, None] + direction[:, None] * t
    offsets = [np.zeros(3)]
    if transverse_width > 0:
        unit = direction / length
        seed = np.array([1., 0., 0.]) if abs(unit[0]) < 0.9 else np.array([0., 1., 0.])
        p1 = np.cross(unit, seed); p1 /= np.linalg.norm(p1)
        p2 = np.cross(unit, p1)
        delta = transverse_width / 2
        offsets = [np.zeros(3), delta*p1, -delta*p1, delta*p2, -delta*p2]
    values = []
    for offset in offsets:
        shifted = points + offset[:, None]
        coords = [(shifted[i] - model.axes[i][0]) / model.scale[i] for i in range(3)]
        values.append(map_coordinates(model.volume, coords, order=1,
                                      mode="constant", cval=np.nan))
    stack = np.asarray(values)
    y = np.nansum(stack, axis=0) if reduction == "sum" else np.nanmean(stack, axis=0)
    return t * length, np.asarray(y)


def save_region(path: str, model: RegionModel) -> None:
    np.savez_compressed(path, volume=model.volume, H=model.axes[0], K=model.axes[1],
                        L=model.axes[2], U=model.U, source=np.array(model.source),
                        settings=np.array(model.settings, dtype=object))


def load_region(path: str) -> RegionModel:
    with np.load(path, allow_pickle=True) as saved:
        settings = saved["settings"].item() if "settings" in saved else {}
        source = str(saved["source"].item()) if "source" in saved else ""
        return RegionModel(saved["volume"], (saved["H"], saved["K"], saved["L"]),
                           saved["U"], source, settings)


def save_csv(path: str, x: np.ndarray, y: np.ndarray, x_label: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_label, "intensity"])
        writer.writerows(zip(x, y))


class RSMViewerController:
    """Owns source data, displayed region, and napari layer state."""

    def __init__(self, viewer: Any, memory_limit_mb: float = 2048) -> None:
        self.viewer = viewer
        self.memory_limit_mb = memory_limit_mb
        self.source_data: Optional[np.ndarray] = None
        self.source_axes: Optional[tuple[np.ndarray, ...]] = None
        self.source_path = ""
        self.U = np.eye(3)
        self.colormap = "inferno"          # image colormap (change via napari panel)
        self.equal_axes = False            # render as a cube; see display_scale
        # B* (reciprocal-cell matrix) from the last Calculate U; needed to plot
        # in RLU (sample at U@B*@x so x is in hkl). None until a U is indexed
        # from a substrate -- a loaded/identity U has no associated cell.
        self.Bstar: Optional[np.ndarray] = None
        self.model: Optional[RegionModel] = None
        self.image_layer = None
        self.worker = None
        self.last_cut: Optional[tuple[np.ndarray, np.ndarray, str]] = None
        # Callbacks fired after a region is displayed (e.g. so the line-cut
        # dock can refresh its unit labels when RLU vs A^-1 changes).
        self.model_listeners: list[Any] = []
        # Optional sink for live readouts (slider position, hover). The region
        # dock sets this to its status label so the Q value shows there, where
        # it is easy to see; falls back to the napari status bar.
        self.readout_sink: Optional[Any] = None
        # Dock-widget handles by key ('region', 'line') so a closed dock can be
        # re-shown; populated in main() after the docks are added.
        self.dock_handles: dict[str, Any] = {}

    def load_source(self, path: str) -> None:
        data, H, K, L = load_rsm(path)
        self.source_data = data
        self.source_axes = (H, K, L)
        self.source_path = os.path.abspath(path)

    def set_u(self, matrix: Any, bstar: Any = None) -> None:
        self.U = validate_u_matrix(matrix)
        # B* only travels with a U indexed from a substrate; clear it for a
        # loaded or identity U so RLU plotting can't use a stale cell.
        self.Bstar = None if bstar is None else validate_u_matrix(bstar)

    def calculate_u(self, substrate: str, path: str,
                    normal: Optional[np.ndarray] = None) -> dict:
        """Index the raw source against a substrate cell and set U.

        Loads the source if needed, runs the find-peaks + lattice-indexing
        pipeline from Visualize_RSM_Lib, and stores the resulting U. Returns
        the result dict (keys include 'U', 'rms', 'n_inliers').

        ``normal`` is the substrate surface normal (h,k,l), e.g. [0,0,1]; it
        pins U reproducibly so the indexed frame is consistent across runs.
        """
        if not substrate:
            raise ValueError("choose a substrate")
        if self.source_path != os.path.abspath(path):
            self.load_source(path)
        H, K, L = self.source_axes
        result = compute_U_from_substrate(self.source_data, H, K, L,
                                          LATTICE_FILE, substrate,
                                          normal=normal, verbose=False)
        if result is None:
            raise ValueError(f"could not index against {substrate} "
                             "(too few peaks or no match)")
        self.set_u(result["U"], result.get("Bstar"))
        return result

    def display_scale(self, model: RegionModel) -> tuple[float, float, float]:
        """Per-voxel scale for rendering.

        Normally the model's true scale, so the box keeps its physical
        proportions. With ``equal_axes`` the scale is chosen so every axis
        spans the same world extent -- the region renders as a cube no matter
        how unequal the Q ranges are.
        """
        if not self.equal_axes:
            return model.scale
        spans = [s * max(n - 1, 1) for s, n in zip(model.scale, model.volume.shape)]
        target = max(spans) if spans else 1.0
        return tuple(target / max(n - 1, 1) for n in model.volume.shape)

    def show_model(self, model: RegionModel, mode: str = "Linear") -> None:
        self.model = model
        displayed = intensity_view(model.volume, mode)
        scale = self.display_scale(model)
        kwargs = dict(name="RSM intensity", colormap=self.colormap, scale=scale,
                      translate=model.translate)
        if self.image_layer is None or self.image_layer not in self.viewer.layers:
            self.image_layer = self.viewer.add_image(displayed, **kwargs)
        else:
            self.image_layer.data = displayed
            self.image_layer.scale = scale
            self.image_layer.translate = model.translate
        labels = model.settings.get("orientation")
        self.viewer.dims.axis_labels = tuple(labels) if labels else VOLUME_AXIS_NAMES
        for listener in self.model_listeners:
            listener(model)

    def set_equal_axes(self, equal: bool) -> None:
        """Toggle cube (equal-extent) rendering and re-apply to the layer."""
        self.equal_axes = bool(equal)
        if self.model is not None and self.image_layer is not None:
            self.image_layer.scale = self.display_scale(self.model)
            self.viewer.reset_view()

    def set_intensity_mode(self, mode: str) -> None:
        if self.model is not None and self.image_layer is not None:
            self.image_layer.data = intensity_view(self.model.volume, mode)

    def _units(self) -> str:
        return self.model.settings.get("units", "A^-1") if self.model is not None else "A^-1"

    def _emit_readout(self, text: str) -> None:
        """Show a live readout in the dock status label (if set) and the
        napari status bar."""
        if self.readout_sink is not None:
            try:
                self.readout_sink(text)
            except Exception:
                pass
        self.viewer.status = text

    def install_canvas_callbacks(self) -> None:
        """Wire the 2D hover readout and the slider-position readout."""

        def hover(viewer, event):
            if self.model is None:
                return
            pos = viewer.cursor.position          # world coords, floats, full ndim
            names = viewer.dims.axis_labels
            unit = self._units()
            parts = [f"{names[i]}={pos[i]:.3f}" for i in range(len(pos))]
            value = ""
            if self.image_layer is not None:
                try:
                    val = self.image_layer.get_value(pos, world=True)
                    if val is not None:
                        value = f"  I={float(val):.3f}"
                except Exception:
                    pass
            self._emit_readout(f"{', '.join(parts)} {unit}{value}")

        self.viewer.mouse_move_callbacks.append(hover)
        self.viewer.dims.events.current_step.connect(self._show_slider_position)

    def _show_slider_position(self, event: Any = None) -> None:
        """Show the slider axis position in physical Q (2 decimals), not the
        integer stack index -- e.g. 'Q3 = 2.35 A^-1'."""
        if self.model is None:
            return
        dims = self.viewer.dims
        point = dims.point                        # world coords per axis
        names = dims.axis_labels
        sliders = [a for a in range(dims.ndim) if a not in dims.displayed]
        if not sliders:
            return
        unit = self._units()
        parts = [f"{names[a]} = {point[a]:.2f} {unit}" for a in sliders]
        self._emit_readout(", ".join(parts))


def _install_command() -> str:
    return "conda install -n viz -c conda-forge napari pyqt pyqtgraph"


def build_gui(controller: RSMViewerController, initial: argparse.Namespace) -> tuple[Any, Any]:
    """Build Qt docks lazily so numerical helpers work without napari/Qt."""
    from napari.qt.threading import thread_worker
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                                QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                                QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                QListWidgetItem, QMessageBox, QPushButton,
                                QSpinBox, QTabWidget, QVBoxLayout, QWidget)
    import pyqtgraph as pg

    class RegionDock(QWidget):
        def __init__(self) -> None:
            super().__init__()
            layout = QVBoxLayout(self)
            source_row = QHBoxLayout(); self.source = QLineEdit(initial.file or "")
            browse = QPushButton("Browse .nxs"); browse.clicked.connect(self.choose_source)
            source_row.addWidget(self.source); source_row.addWidget(browse); layout.addLayout(source_row)
            u_row = QHBoxLayout(); self.u_path = QLineEdit(initial.u_matrix or "")
            choose_u = QPushButton("Load U matrix"); choose_u.clicked.connect(self.choose_u)
            save_u = QPushButton("Save U matrix"); save_u.clicked.connect(self.save_u)
            identity = QPushButton("Use identity"); identity.clicked.connect(self.use_identity)
            u_row.addWidget(self.u_path); u_row.addWidget(choose_u); u_row.addWidget(save_u); u_row.addWidget(identity); layout.addLayout(u_row)
            calc_box = QGroupBox("Calculate U matrix")
            calc_row = QHBoxLayout(calc_box)
            self.substrate = QComboBox(); self.substrate.addItems(load_substrate_names())
            # Surface normal pins the indexed frame so U is reproducible (e.g.
            # [001] for a (001)-oriented cubic substrate). Two in-plane axes
            # are chosen orthogonal to it; see axes_from_normal in the library.
            self.normal = QLineEdit("[0 0 1]")
            self.normal.setToolTip("Substrate surface normal (h k l); pins U for "
                                   "reproducible orientation. Blank = unconstrained.")
            calc_u = QPushButton("Calculate U"); calc_u.clicked.connect(self.calculate_u)
            calc_row.addWidget(QLabel("Substrate")); calc_row.addWidget(self.substrate, 1)
            calc_row.addWidget(QLabel("Normal")); calc_row.addWidget(self.normal)
            calc_row.addWidget(calc_u)
            layout.addWidget(calc_box)
            # Each oriented Q axis carries its own direction and range cut on a
            # single row: [name | direction | lo | hi | count]. Q directions are
            # in the U-aligned frame (default Q1=[100], Q2=[010], Q3=[001]); the
            # lo/hi range is cut ALONG that (rotated) direction. Q1=horizontal,
            # Q2=vertical, Q3=slider. Ranges stay in Q1, Q2, Q3 order and are
            # mapped to volume axis order (Q3, Q2, Q1) in interpolate().
            qbox = QGroupBox("Oriented Q axes (direction + range cut)")  # units appended below
            grid = QGridLayout(qbox)
            for col, header in enumerate(("Axis", "Direction", "Q min", "Q max", "Samples")):
                grid.addWidget(QLabel(header), 0, col)
            self.limits = []; self.counts = []
            self.orient = []
            roles = ("horizontal", "vertical", "slider")
            directions = ("[1 0 0]", "[0 1 0]", "[0 0 1]")
            defaults = (initial.q1_range, initial.q2_range, initial.q3_range)
            shape = initial.shape
            for i, (name, role, dir_text, bounds, count) in enumerate(
                    zip(QAXIS_NAMES, roles, directions, defaults, shape)):
                r = i + 1
                direction = QLineEdit(dir_text)
                lo = QDoubleSpinBox(); hi = QDoubleSpinBox(); n = QSpinBox()
                for box in (lo, hi): box.setRange(-1e6, 1e6); box.setDecimals(2)
                lo.setValue(bounds[0]); hi.setValue(bounds[1]); n.setRange(1, 10000); n.setValue(count)
                grid.addWidget(QLabel(f"{name} ({role})"), r, 0)
                grid.addWidget(direction, r, 1); grid.addWidget(lo, r, 2)
                grid.addWidget(hi, r, 3); grid.addWidget(n, r, 4)
                self.orient.append(direction)
                self.limits.append((lo, hi)); self.counts.append(n)
            self.orient_q1, self.orient_q2, self.orient_q3 = self.orient
            layout.addWidget(qbox)
            # Colormap and 3D rendering are controllable from napari's built-in
            # layer-controls panel (left), so they are not duplicated here.
            row = QHBoxLayout(); self.order = QSpinBox(); self.order.setRange(0, 5); self.order.setValue(1)
            self.mode = QComboBox(); self.mode.addItems(["Linear", "log1p", "log10(I + 1)"])
            self.mode.currentTextChanged.connect(controller.set_intensity_mode)
            row.addWidget(QLabel("Order")); row.addWidget(self.order); row.addWidget(QLabel("Intensity")); row.addWidget(self.mode)
            # Equal axes: render the region as a cube regardless of the Q-range
            # spans. Uncheck for the true physical proportions of the data.
            self.equal_axes = QCheckBox("Equal axes (cube)")
            self.equal_axes.setToolTip("Show the region as a cube no matter how "
                                       "unequal the Q ranges are. Uncheck for true "
                                       "physical proportions.")
            self.equal_axes.toggled.connect(controller.set_equal_axes)
            row.addWidget(self.equal_axes); layout.addLayout(row)
            # UB / RLU: plot in reciprocal lattice units (hkl) instead of A^-1
            # by sampling at U@B*@x. Needs B* from Calculate U (substrate cell).
            self.rlu = QCheckBox("Use UB matrix (plot in RLU)")
            self.rlu.setToolTip("Sample at U.B*.x so axes are in reciprocal lattice "
                                "units (hkl). Requires Calculate U from a substrate.")
            self.rlu.toggled.connect(self._update_units_label)
            layout.addWidget(self.rlu)
            self.qbox = qbox
            self.memory = QLabel(); layout.addWidget(self.memory)
            for n in self.counts: n.valueChanged.connect(self.update_memory)
            self.run = QPushButton("Interpolate region"); self.run.clicked.connect(self.interpolate); layout.addWidget(self.run)
            buttons = QHBoxLayout()
            for text, callback in (("Save region", self.save), ("Load region", self.load),
                                   ("Reset view", controller.viewer.reset_view),
                                   ("Center camera", self.center_camera),
                                   ("Show plot window", self.show_plot_window)):
                button = QPushButton(text); button.clicked.connect(callback); buttons.addWidget(button)
            layout.addLayout(buttons); self.status = QLabel("Ready"); self.status.setWordWrap(True); layout.addWidget(self.status)
            self._dock_handles: dict[str, Any] = {}
            # Live Q readouts (slider position, hover) land in this status label.
            controller.readout_sink = self.status.setText
            self.update_memory(); self._update_units_label()

        def set_dock_handles(self, handles: dict[str, Any]) -> None:
            self._dock_handles = handles

        def show_plot_window(self) -> None:
            """Re-show the line-cut dock if it was closed (also under Window menu)."""
            dock = self._dock_handles.get("line")
            if dock is None:
                self.status.setText("Plot window handle unavailable; use the Window menu.")
                return
            dock.show(); dock.raise_()
            if hasattr(dock, "setFloating"):
                dock.setFloating(False)

        def error(self, exc: Any) -> None:
            self.run.setEnabled(True); self.status.setText(f"Error: {exc}")
            QMessageBox.critical(self, "EpiQ-Map", str(exc))

        def choose_source(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open autoRSM file", "", "NeXus (*.nxs)")
            if path: self.source.setText(path)

        def choose_u(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open U matrix", "", "Text (*.txt);;All files (*)")
            if not path: return
            try: controller.set_u(load_U_matrix(path)); self.u_path.setText(path); self.status.setText("Loaded U matrix")
            except Exception as exc: self.error(exc)

        def save_u(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Save U matrix", "U_matrix.txt", "Text (*.txt);;All files (*)")
            if not path: return
            if not path.endswith(".txt"): path += ".txt"
            try: save_U_matrix(controller.U, path); self.u_path.setText(path); self.status.setText(f"Saved U matrix to {path}")
            except Exception as exc: self.error(exc)

        def use_identity(self) -> None:
            controller.set_u(np.eye(3)); self.u_path.clear(); self.status.setText("Using identity U matrix")

        def calculate_u(self) -> None:
            try:
                path = self.source.text().strip()
                if not path: raise ValueError("select a .nxs source file first")
                substrate = self.substrate.currentText()
                normal_text = self.normal.text().strip()
                normal = parse_direction(normal_text) if normal_text else None
                tag = f" (normal {format_direction(normal)})" if normal is not None else ""
                self.status.setText(f"Indexing against {substrate}{tag}..."); QApplication.processEvents()
                result = controller.calculate_u(substrate, path, normal=normal)
                self.u_path.setText(f"computed ({substrate})")
                self.status.setText(
                    f"U from {substrate}{tag}: {result['n_inliers']} inliers, "
                    f"RMS {result['rms']:.4f} A^-1")
            except Exception as exc: self.error(exc)

        def read_orientation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return the Q1, Q2, Q3 crystal directions (U-aligned frame)."""
            return tuple(parse_direction(box.text()) for box in self.orient)

        def update_memory(self) -> None:
            shape = tuple(n.value() for n in self.counts)
            self.memory.setText(f"Array: {' x '.join(map(str, shape))}, {np.prod(shape)*4/2**20:.1f} MiB; estimated peak {estimated_megabytes(shape):.1f} MiB")

        def _update_units_label(self) -> None:
            unit = "rlu" if self.rlu.isChecked() else "A^-1"
            self.qbox.setTitle(f"Oriented Q axes (direction + range cut, {unit})")

        def interpolate(self) -> None:
            try:
                # UI order is Q1, Q2, Q3; reorder to napari volume axes (Q3, Q2, Q1).
                q_bounds = [(lo.value(), hi.value()) for lo, hi in self.limits]
                q_shape = tuple(n.value() for n in self.counts)
                bounds = [q_bounds[i] for i in QAXIS_TO_VOLUME]
                shape = tuple(q_shape[i] for i in QAXIS_TO_VOLUME)
                axes = make_region_axes(bounds, shape)
                estimate = estimated_megabytes(shape)
                if estimate > controller.memory_limit_mb and QMessageBox.question(
                        self, "Large allocation", f"Estimated peak allocation is {estimate:.0f} MiB. Continue?") != QMessageBox.Yes:
                    return
                path = self.source.text().strip()
                if not path: raise ValueError("select a .nxs source file")
                if controller.source_path != os.path.abspath(path):
                    self.status.setText("Loading source data..."); QApplication.processEvents(); controller.load_source(path)
                q1, q2, q3 = self.read_orientation()
                # Orientation matrix columns are in volume-axis order (Q3, Q2, Q1).
                vol_dirs = [(q1, q2, q3)[i] for i in QAXIS_TO_VOLUME]
                # RLU: sample at U@B*@x so grid coordinate x is in hkl (reciprocal
                # lattice units). Without it, x stays in A^-1 cartesian q-space.
                rlu = self.rlu.isChecked()
                if rlu and controller.Bstar is None:
                    raise ValueError("RLU plotting needs a substrate cell; run "
                                     "Calculate U first (a loaded/identity U has none).")
                base_u = controller.U @ controller.Bstar if rlu else controller.U
                U = oriented_u_matrix(base_u, *vol_dirs)
                data, source_axes, order = controller.source_data, controller.source_axes, self.order.value()
                labels = tuple(format_direction(d) for d in vol_dirs)
                settings = dict(bounds=bounds, shape=shape, order=order,
                                orientation=labels, units="rlu" if rlu else "A^-1")
                @thread_worker
                def work():
                    volume = transform_slab(data, *source_axes, U, *axes, order=order)
                    return RegionModel(np.asarray(volume, dtype=np.float32), axes, U,
                                       controller.source_path, settings)
                self.run.setEnabled(False); self.status.setText("Interpolating bounded region...")
                controller.worker = work()
                controller.worker.returned.connect(self.finished)
                controller.worker.errored.connect(self.error)
                controller.worker.start()
            except Exception as exc: self.error(exc)

        def finished(self, model: RegionModel) -> None:
            controller.show_model(model, self.mode.currentText()); self.run.setEnabled(True)
            self.status.setText(f"Displayed {model.volume.shape}, {model.volume.nbytes/2**20:.1f} MiB float32")

        def save(self) -> None:
            if controller.model is None: self.error(ValueError("nothing to save")); return
            path, _ = QFileDialog.getSaveFileName(self, "Save interpolated region", "region.npz", "NPZ (*.npz)")
            if path:
                if not path.endswith(".npz"): path += ".npz"
                try: save_region(path, controller.model); self.status.setText(f"Saved {path}")
                except Exception as exc: self.error(exc)

        def load(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Load interpolated region", "", "NPZ (*.npz)")
            if path:
                try: controller.show_model(load_region(path), self.mode.currentText()); self.status.setText(f"Loaded {path}")
                except Exception as exc: self.error(exc)

        def center_camera(self) -> None:
            controller.viewer.reset_view()
            if controller.viewer.dims.ndisplay == 3 and controller.model is not None:
                controller.viewer.camera.center = tuple(
                    float((axis[0] + axis[-1]) / 2) for axis in controller.model.axes)

    class LineCutDock(QWidget):
        def __init__(self) -> None:
            super().__init__(); layout = QVBoxLayout(self); self.tabs = QTabWidget(); layout.addWidget(self.tabs)
            axis_page = QWidget(); form = QFormLayout(axis_page); self.axis = QComboBox(); self.axis.addItems(VOLUME_AXIS_NAMES)
            self.fixed1 = QDoubleSpinBox(); self.fixed2 = QDoubleSpinBox(); self.width1 = QDoubleSpinBox(); self.width2 = QDoubleSpinBox()
            for box in (self.fixed1,self.fixed2): box.setRange(-1e6,1e6); box.setDecimals(6)
            for box in (self.width1,self.width2): box.setRange(0,1e6); box.setDecimals(6)
            self.reduce = QComboBox(); self.reduce.addItems(["mean", "sum"])
            form.addRow("Scan axis", self.axis)
            self.fixed1_label = QLabel(); self.fixed2_label = QLabel()
            self.width1_label = QLabel(); self.width2_label = QLabel()
            form.addRow(self.fixed1_label, self.fixed1); form.addRow(self.fixed2_label, self.fixed2)
            form.addRow(self.width1_label, self.width1); form.addRow(self.width2_label, self.width2); form.addRow("Integration", self.reduce)
            self.axis.currentIndexChanged.connect(self.update_axis_labels); self.update_axis_labels()
            run_axis = QPushButton("Plot axis-aligned cut"); run_axis.clicked.connect(self.plot_axis); form.addRow(run_axis); self.tabs.addTab(axis_page, "Axis aligned")
            line_page = QWidget(); line_form = QFormLayout(line_page); self.samples = QSpinBox(); self.samples.setRange(2,100000); self.samples.setValue(300)
            self.transverse = QDoubleSpinBox(); self.transverse.setRange(0,1e6); self.transverse.setDecimals(6)
            add_line = QPushButton("Create line layer"); add_line.clicked.connect(self.create_line)
            run_line = QPushButton("Plot selected line"); run_line.clicked.connect(self.plot_line)
            line_form.addRow("Samples", self.samples); line_form.addRow("Transverse width", self.transverse); line_form.addRow(add_line); line_form.addRow(run_line); self.tabs.addTab(line_page,"Arbitrary line")
            opts = QHBoxLayout(); self.log_y = QCheckBox("Log Y"); self.log_y.toggled.connect(lambda value: self.plot.setLogMode(y=value)); opts.addWidget(self.log_y)
            save = QPushButton("Save CSV"); save.clicked.connect(self.save); copy = QPushButton("Copy CSV"); copy.clicked.connect(self.copy); opts.addWidget(save); opts.addWidget(copy); layout.addLayout(opts)
            # Plot on the left, a checklist of overlaid curves on the right so
            # cuts accumulate and can be toggled for comparison. Each curve has
            # its own color; last_cut tracks the most recent one for Save/Copy.
            split = QHBoxLayout()
            self.plot = pg.PlotWidget(); self.plot.setLabel("left", "Intensity"); self.plot.showGrid(x=True,y=True,alpha=.25)
            self.plot.addLegend(); split.addWidget(self.plot, 1)
            side = QVBoxLayout(); side.addWidget(QLabel("Overlaid cuts"))
            self.curve_list = QListWidget(); self.curve_list.itemChanged.connect(self._toggle_curve)
            side.addWidget(self.curve_list)
            clear = QPushButton("Clear all"); clear.clicked.connect(self.clear_curves); side.addWidget(clear)
            split.addLayout(side); layout.addLayout(split)
            self.curves: list[Any] = []           # (item, PlotDataItem) per cut
            self._color_index = 0
            self.cursor = QLabel("Move over plot for coordinates"); layout.addWidget(self.cursor)
            self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self.mouse_moved)
            # Refresh unit labels whenever a new region is displayed (A^-1 vs RLU).
            controller.model_listeners.append(lambda _model: self.update_axis_labels())

        def units(self) -> str:
            """Axis units of the current region ('rlu' if it was UB-transformed)."""
            model = controller.model
            return model.settings.get("units", "A^-1") if model is not None else "A^-1"

        def update_axis_labels(self) -> None:
            unit = self.units()
            names = [name for i, name in enumerate(VOLUME_AXIS_NAMES) if i != self.axis.currentIndex()]
            self.fixed1_label.setText(f"Fixed {names[0]} ({unit})")
            self.fixed2_label.setText(f"Fixed {names[1]} ({unit})")
            self.width1_label.setText(f"{names[0]} integration width")
            self.width2_label.setText(f"{names[1]} integration width")

        # Distinct colors cycled across overlaid curves.
        _palette = ("#ff9d00", "#1f9ee0", "#43c25a", "#e0473f", "#b56cff",
                    "#f0c419", "#ff7eb6", "#22b3a4", "#9b8d5a", "#7a8cff")

        def display(self, x: np.ndarray, y: np.ndarray, label: str) -> None:
            """Add a cut as a new overlaid curve named '<file> <axis>'."""
            controller.last_cut = (x, y, label)
            self.plot.setLabel("bottom", label, units=self.units())
            tag = source_tag(controller.model.source) if controller.model is not None else "?"
            name = f"#{tag} {label}"
            color = self._palette[self._color_index % len(self._palette)]
            self._color_index += 1
            curve = self.plot.plot(x, y, pen=pg.mkPen(color, width=2), name=name)
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setForeground(pg.mkColor(color))
            self.curve_list.addItem(item)
            self.curves.append((item, curve))

        def _toggle_curve(self, item: Any) -> None:
            for it, curve in self.curves:
                if it is item:
                    curve.setVisible(item.checkState() == Qt.Checked)
                    break

        def clear_curves(self) -> None:
            for _item, curve in self.curves:
                self.plot.removeItem(curve)
            self.curves.clear(); self.curve_list.clear(); self._color_index = 0
            controller.last_cut = None

        def plot_axis(self) -> None:
            try:
                if controller.model is None: raise ValueError("interpolate or load a region first")
                i=self.axis.currentIndex(); x,y=axis_aligned_cut(controller.model,i,(self.fixed1.value(),self.fixed2.value()),(self.width1.value(),self.width2.value()),self.reduce.currentText())
                self.display(x,y,VOLUME_AXIS_NAMES[i])
            except Exception as exc: QMessageBox.critical(self,"Line cut",str(exc))

        def create_line(self) -> None:
            if controller.model is None: return
            axes=controller.model.axes; start=np.array([a[len(a)//4] for a in axes]); end=np.array([a[3*len(a)//4] for a in axes])
            controller.viewer.add_shapes([np.vstack((start,end))],shape_type="line",name="RSM line")

        def plot_line(self) -> None:
            try:
                if controller.model is None: raise ValueError("interpolate or load a region first")
                layer=controller.viewer.layers.selection.active
                if getattr(layer,"_type_string","")!="shapes" or len(layer.data)==0: raise ValueError("select a Shapes layer containing a line")
                points=np.asarray(layer.data[-1]); start=np.asarray(layer.data_to_world(points[0]))[-3:]; end=np.asarray(layer.data_to_world(points[-1]))[-3:]
                x,y=arbitrary_line_cut(controller.model,start,end,self.samples.value(),self.transverse.value(),self.reduce.currentText()); self.display(x,y,"distance")
            except Exception as exc: QMessageBox.critical(self,"Line cut",str(exc))

        def save(self) -> None:
            if controller.last_cut is None: return
            path,_=QFileDialog.getSaveFileName(self,"Save line data","line_cut.csv","CSV (*.csv)")
            if path:
                if not path.endswith(".csv"): path += ".csv"
                save_csv(path,*controller.last_cut)

        def copy(self) -> None:
            if controller.last_cut is None: return
            x,y,label=controller.last_cut; lines=[f"{label},intensity"]+[f"{a:.12g},{b:.12g}" for a,b in zip(x,y)]; QApplication.clipboard().setText("\n".join(lines))

        def mouse_moved(self, event: Any) -> None:
            pos=event[0]
            if self.plot.sceneBoundingRect().contains(pos):
                point=self.plot.plotItem.vb.mapSceneToView(pos); self.cursor.setText(f"x={point.x():.6g} {self.units()}, I={point.y():.6g}")

    return RegionDock(), LineCutDock()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file"); parser.add_argument("--u-matrix")
    parser.add_argument("--q1-range", nargs=2, type=float, default=(-4,4))
    parser.add_argument("--q2-range", nargs=2, type=float, default=(-.3,.3))
    parser.add_argument("--q3-range", nargs=2, type=float, default=(0,6))
    parser.add_argument("--shape", nargs=3, type=int, default=(400,80,500))
    parser.add_argument("--memory-limit-mb", type=float, default=2048)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import napari
        import qtpy  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        print(f"Missing GUI dependency: {exc}\nInstall with:\n  {_install_command()}", file=sys.stderr)
        return 2
    viewer = napari.Viewer(title="EpiQ-Map")
    viewer.theme = "light"
    controller = RSMViewerController(viewer, args.memory_limit_mb)
    if args.u_matrix:
        try: controller.set_u(load_U_matrix(args.u_matrix))
        except Exception as exc: print(f"Could not load U matrix: {exc}", file=sys.stderr)
    region_dock, line_dock = build_gui(controller, args)
    region_handle = viewer.window.add_dock_widget(region_dock, name="RSM region", area="right")
    line_handle = viewer.window.add_dock_widget(line_dock, name="Line cuts", area="right")
    # Keep handles so a closed dock can be re-shown. napari also lists both
    # under the Window menu; region_dock gets a button as a guaranteed path.
    controller.dock_handles = {"region": region_handle, "line": line_handle}
    if hasattr(region_dock, "set_dock_handles"):
        region_dock.set_dock_handles(controller.dock_handles)
    controller.install_canvas_callbacks()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
