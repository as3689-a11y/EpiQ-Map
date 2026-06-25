#!/usr/bin/env python3
"""CTR-rod reconstruction front-end for the EpiQ-Map monitor.

Defines :class:`CTRDialog` -- launched from the monitor's "CTR rods..." button
on a selected group of phi scans. The workflow is:

  1. **Find UB** -- re-index every selected scan's volume against a substrate
     (this is what carries the substrate orientation), then average the per-scan
     orientations into one UB. Two non-blocking helper windows open on demand:
       * **Found peaks** -- a table of the found-and-indexed peaks (hkl, |q|,
         residual, inlier) so you can see what is there and turn reflections
         into rods.
       * **Projection** -- the data summed along the substrate normal, with the
         two in-plane crystal directions on the axes.
  2. **Rods** -- choose the (h, k) rods (a full H/K grid, from the indexed
     peaks, or hand-edited), the high-resolution L grid, the narrow per-rod
     (H, K) windows, and an optional rotation to new coordinates (the rod axes
     run along chosen crystal directions, like rsm_viewer's oriented Q axes).

The averaged UB and the (cheap) peaks/projection live on the dialog after Find
UB; the long reconstruction itself is launched by the monitor. The heavy
indexing runs on a worker thread (``QThreadPool``) so the dialog stays live.

Created by Ben Gregory, Timo Fuchs, and Andrej Singer, Cornell University.
"""

import traceback

import numpy as np
# Qt comes through QtPy so these dialogs use the same PyQt/PySide and Qt 5/6
# binding already selected by the monitor or napari process.
from qtpy import QtCore, QtWidgets

from . import rsm_workflow as workflow


# ----------------------------------------------------------------------
# Indexing + projection (GUI-independent; run on a worker thread)
# ----------------------------------------------------------------------

def _format_direction(vec):
    """Short ``[h k l]`` label for a (possibly non-integer) crystal direction."""
    def part(value):
        if abs(value - round(value)) < 1e-3:
            return str(int(round(value)))
        return f'{value:.2g}'
    return '[' + ' '.join(part(v) for v in np.asarray(vec, float)) + ']'


def peak_rows(scan_number, result):
    """Flatten one ``index_against_lattice`` result into per-peak table rows.

    Each row: ``{scan, hkl, q, resid, inlier}`` -- ``hkl`` is the assigned
    integer index (None for an outlier), ``q`` the measured |q| (A^-1), and
    ``resid`` the |U.B*.hkl - q| residual (A^-1, NaN for outliers).
    """
    peaks = np.asarray(result['peaks'], float)
    hkl = np.asarray(result['hkl'], float)
    inliers = np.asarray(result['inliers'], bool)
    UB = np.asarray(result['U'], float) @ np.asarray(result['Bstar'], float)
    rows = []
    for peak, h, ok in zip(peaks, hkl, inliers):
        if ok and np.all(np.isfinite(h)):
            hkl_int = tuple(int(round(v)) for v in h)
            resid = float(np.linalg.norm(UB @ h - peak))
        else:
            hkl_int, resid = None, float('nan')
        rows.append({'scan': scan_number, 'hkl': hkl_int,
                     'q': float(np.linalg.norm(peak)),
                     'resid': resid, 'inlier': bool(ok)})
    return rows


def compute_projection(data, H, K, L, result, normal, samples=(220, 220, 80)):
    """Project the volume along the substrate normal into the in-plane frame.

    Returns ``{image, h_axis, v_axis, labels}`` -- the data summed (nansum) over
    the normal direction onto the two in-plane crystal axes from
    ``axes_from_normal``, in r.l.u., for a quick "what reflections are here" map.
    Bounds come from the indexed inlier peaks (padded), so the peaks sit inside.
    """
    Bstar = np.asarray(result['Bstar'], float)
    normal = normal if normal is not None else [0, 0, 1]
    ox, oy, oz = workflow.rl.axes_from_normal(Bstar, normal)
    R = np.column_stack((ox, oy, oz))
    UB = np.asarray(result['U'], float) @ Bstar
    U_view = UB @ R                      # rotated-rlu coords -> measured q

    hkl = np.asarray(result['hkl'], float)
    inliers = np.asarray(result['inliers'], bool)
    finite = inliers & np.all(np.isfinite(hkl), axis=1)
    if finite.any():
        abc = np.linalg.solve(R, hkl[finite].T).T   # peaks in the rotated frame
        lo = abc.min(axis=0) - 1.0
        hi = abc.max(axis=0) + 1.0
    else:
        lo, hi = np.array([-3.0, -3.0, -1.0]), np.array([3.0, 3.0, 1.0])
    n_h, n_v, n_t = (int(n) for n in samples)
    h_axis = np.linspace(lo[0], hi[0], n_h)
    v_axis = np.linspace(lo[1], hi[1], n_v)
    thin = np.linspace(lo[2], hi[2], n_t)
    volume = workflow.rl.transform_slab(data, H, K, L, U_view, h_axis, v_axis,
                                        thin, order=1)
    image = np.nansum(volume, axis=2).T          # (n_v, n_h) for display
    return {'image': np.asarray(image, np.float32),
            'h_axis': h_axis, 'v_axis': v_axis,
            'labels': (f'{_format_direction(ox)} (r.l.u.)',
                       f'{_format_direction(oy)} (r.l.u.)')}


def index_group(datasets, lattice_file, substrate, normal,
                status=lambda *_: None):
    """Re-index each scan against ``substrate`` and average the orientations.

    Returns ``{metadata, peaks, projection}`` -- the averaged U_S metadata
    (``workflow.average_U``), the pooled per-peak rows, and a projection of the
    first scan's volume along the normal (None if it could not be built). Heavy
    (loads each volume); call on a worker thread.
    """
    records, peaks, projection = [], [], None
    for index, ds in enumerate(datasets):
        out = ds.resolved_output()
        if out is None:
            raise ValueError(f'scan {ds.scan_number} has no reconstructed '
                             'volume to index; convert it first')
        status(f'Indexing scan {ds.scan_number} against {substrate} ...')
        data, H, K, L = workflow.rl.load_rsm(out)
        try:
            result = workflow.rl.compute_U_from_substrate(
                data, H, K, L, lattice_file, substrate, normal=normal,
                verbose=False)
            if result is None:
                raise RuntimeError(f'no consistent {substrate} indexing for '
                                   f'scan {ds.scan_number}')
            records.append(workflow.build_index_metadata(
                result, substrate, workflow.rl.load_lattice(lattice_file,
                                                            substrate),
                out, method='fresh', normal=normal))
            peaks.extend(peak_rows(ds.scan_number, result))
            if projection is None:
                try:
                    projection = compute_projection(data, H, K, L, result,
                                                    normal)
                except Exception:
                    projection = None
        finally:
            del data
    return {'metadata': workflow.average_U(records), 'peaks': peaks,
            'projection': projection}


# ----------------------------------------------------------------------
# Rod integration boxes (GUI-free; drive the box viewer and I(L) export)
# ----------------------------------------------------------------------

def rod_projections(data):
    """Three orthogonal projections of a rod volume ``data`` of shape (H, K, L).

    Returns ``{'Z', 'HL', 'KL'}`` -- ``Z`` is summed (nansum) along L (the H-K
    face, where an integration box is a rectangle), ``HL`` is summed along K and
    ``KL`` along H (the side faces, where the same box is a full-height vertical
    band). Each image is ``(rows, cols)`` with the first listed axis as columns:
    ``Z`` is (K, H), ``HL`` is (L, H), ``KL`` is (L, K) -- i.e. L on the rows so
    it draws vertically, matching the viewer.
    """
    data = np.asarray(data, float)
    return {'Z': np.nansum(data, axis=2).T,          # (K, H)
            'HL': np.nansum(data, axis=1).T,         # (L, H)
            'KL': np.nansum(data, axis=0).T}         # (L, K)


def _box_indices(H, K, box):
    """Indices of the H and K cells inside ``box = (h_lo, h_hi, k_lo, k_hi)``."""
    h_lo, h_hi, k_lo, k_hi = box
    ih = np.where((H >= min(h_lo, h_hi)) & (H <= max(h_lo, h_hi)))[0]
    ik = np.where((K >= min(k_lo, k_hi)) & (K <= max(k_lo, k_hi)))[0]
    return ih, ik


def integrate_rod(data, H, K, L, int_box, bkg_box=None):
    """Integrate a rod over an (H, K) box, optionally subtracting a background.

    ``int_box``/``bkg_box`` are ``(h_lo, h_hi, k_lo, k_hi)`` r.l.u. windows. At
    each L the integration box is summed (nansum) over its (H, K) cells; when a
    background box is given its per-cell mean is scaled to the integration box's
    cell count and subtracted, so ``I(L) = sum_int(L) - (n_int/n_bkg)*sum_bkg(L)``.
    Returns ``(L_copy, I)``. Raises ``ValueError`` if a box selects no cells.
    """
    data = np.asarray(data, float)
    ih, ik = _box_indices(H, K, int_box)
    if ih.size == 0 or ik.size == 0:
        raise ValueError('integration box selects no cells; widen it')
    sub = data[np.ix_(ih, ik, np.arange(data.shape[2]))]
    intensity = np.nansum(sub, axis=(0, 1))
    n_int = ih.size * ik.size
    if bkg_box is not None:
        bh, bk = _box_indices(H, K, bkg_box)
        if bh.size == 0 or bk.size == 0:
            raise ValueError('background box selects no cells; widen it')
        bkg = data[np.ix_(bh, bk, np.arange(data.shape[2]))]
        n_bkg = bh.size * bk.size
        intensity = intensity - (n_int / n_bkg) * np.nansum(bkg, axis=(0, 1))
    return np.asarray(L, float).copy(), np.asarray(intensity, float)


# ----------------------------------------------------------------------
# Worker plumbing (local, to avoid importing the monitor)
# ----------------------------------------------------------------------

class _Signals(QtCore.QObject):
    status = QtCore.Signal(str)
    done = QtCore.Signal(object)
    error = QtCore.Signal(str)


class _Task(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.done.emit(self.function(self.signals.status.emit))
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ----------------------------------------------------------------------
# Non-blocking helper windows
# ----------------------------------------------------------------------

class PeaksWindow(QtWidgets.QDialog):
    """Modeless table of found-and-indexed peaks. Closing it (the window X)
    does not interrupt the CTR setup."""

    def __init__(self, peaks, on_add_pairs, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Found peaks')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(440, 360)
        self._on_add_pairs = on_add_pairs
        layout = QtWidgets.QVBoxLayout(self)
        headers = ('Scan', 'h', 'k', 'l', '|q| A^-1', 'resid', 'in')
        table = QtWidgets.QTableWidget(len(peaks), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        for row, peak in enumerate(peaks):
            hkl = peak['hkl']
            cells = [str(peak['scan']),
                     *(['', '', ''] if hkl is None else [str(v) for v in hkl]),
                     f"{peak['q']:.3f}",
                     '' if np.isnan(peak['resid']) else f"{peak['resid']:.4f}",
                     '✓' if peak['inlier'] else '✗']
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, hkl)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()
        self.table = table
        layout.addWidget(table)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton('Add selected (h, k) as rods')
        add.clicked.connect(self._add_selected)
        row.addWidget(add)
        row.addStretch()
        layout.addLayout(row)

    def _add_selected(self):
        pairs = []
        for index in self.table.selectionModel().selectedRows():
            hkl = self.table.item(index.row(), 0).data(
                QtCore.Qt.UserRole)
            if hkl is not None:
                pairs.append((int(hkl[0]), int(hkl[1])))
        if pairs:
            self._on_add_pairs(pairs)


class ProjectionWindow(QtWidgets.QDialog):
    """Modeless image of the data projected along the substrate normal."""

    def __init__(self, projection, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Projection along substrate normal')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(560, 460)
        layout = QtWidgets.QVBoxLayout(self)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except Exception as exc:                       # pragma: no cover
            layout.addWidget(QtWidgets.QLabel(
                f'matplotlib not available for the projection:\n{exc}'))
            return
        figure = Figure(figsize=(5, 4), tight_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot(111)
        image = np.clip(np.nan_to_num(projection['image']), 0, None)
        h_axis, v_axis = projection['h_axis'], projection['v_axis']
        mappable = axes.imshow(
            np.log1p(image), origin='lower', aspect='auto', cmap='inferno',
            extent=[h_axis[0], h_axis[-1], v_axis[0], v_axis[-1]])
        axes.set_xlabel(projection['labels'][0])
        axes.set_ylabel(projection['labels'][1])
        figure.colorbar(mappable, ax=axes, label='log(1 + I)')
        layout.addWidget(canvas)


class RodBoxViewer(QtWidgets.QDialog):
    """Integrate a rod over an (H, K) box, with an optional background box.

    Both boxes are given as numeric H/K min-max ranges (like rsm_viewer's region
    ranges) rather than drawn -- the integration box and a checkable background
    box. "Natural ranges" fills both with the rod's full H-K extent as a starting
    point. The background-subtracted I(L) profile (``integrate_rod``) updates live
    and can be saved as CSV.
    """

    def __init__(self, data, H, K, L, title='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Rod integration box{f" - {title}" if title else ""}')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(560, 560)
        layout = QtWidgets.QVBoxLayout(self)
        try:
            import pyqtgraph as pg
        except Exception as exc:                       # pragma: no cover
            layout.addWidget(QtWidgets.QLabel(
                f'pyqtgraph not available for the box viewer:\n{exc}'))
            return
        self.pg = pg
        self.data = np.asarray(data, float)
        self.H = np.asarray(H, float)
        self.K = np.asarray(K, float)
        self.L = np.asarray(L, float)
        self._last = None                              # (L, I) for Save CSV

        # The rod's full in-plane extent -- the "natural" ranges.
        self._natural = (float(self.H[0]), float(self.H[-1]),
                         float(self.K[0]), float(self.K[-1]))
        h0, h1, k0, k1 = self._natural
        natural = QtWidgets.QLabel(
            f'Natural rod ranges:  H [{h0:.4f}, {h1:.4f}]   '
            f'K [{k0:.4f}, {k1:.4f}]')
        natural.setWordWrap(True)
        layout.addWidget(natural)

        # Integration box defaults to the central half; the background box to a
        # narrow band just outside it on the high-H side.
        boxes_row = QtWidgets.QHBoxLayout()
        self.int_group, self.int_spins = self._box_group(
            'Integration box', (self._frac(self.H, 0.25),
                                self._frac(self.H, 0.75),
                                self._frac(self.K, 0.25),
                                self._frac(self.K, 0.75)))
        self.bkg_group, self.bkg_spins = self._box_group(
            'Background box', (self._frac(self.H, 0.75),
                               self._frac(self.H, 0.95),
                               self._frac(self.K, 0.25),
                               self._frac(self.K, 0.75)), checkable=True)
        boxes_row.addWidget(self.int_group)
        boxes_row.addWidget(self.bkg_group)
        layout.addLayout(boxes_row)

        controls = QtWidgets.QHBoxLayout()
        natural_button = QtWidgets.QPushButton('Natural ranges')
        natural_button.setToolTip("Fill both boxes with the rod's full H-K "
                                  'extent.')
        natural_button.clicked.connect(self._fill_natural)
        self.log_y = QtWidgets.QCheckBox('Log Y')
        self.log_y.setChecked(True)
        self.log_y.toggled.connect(lambda on: self.profile.setLogMode(y=on))
        save = QtWidgets.QPushButton('Save profile CSV')
        save.clicked.connect(self._save)
        controls.addWidget(natural_button)
        controls.addWidget(self.log_y)
        controls.addStretch()
        controls.addWidget(save)
        layout.addLayout(controls)

        self.profile = pg.PlotWidget()
        self.profile.setLabel('bottom', 'L', units='rlu')
        self.profile.setLabel('left', 'Integrated intensity')
        self.profile.showGrid(x=True, y=True, alpha=0.25)
        self.profile.setLogMode(y=True)
        layout.addWidget(self.profile, 1)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self._refresh()

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _frac(axis, fraction):
        return float(axis[0] + fraction * (axis[-1] - axis[0]))

    def _spin(self, value):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(-10000, 10000)
        box.setDecimals(4)
        box.setSingleStep(0.001)
        box.setValue(value)
        box.valueChanged.connect(self._refresh)
        return box

    def _box_group(self, title, defaults, checkable=False):
        """A group of H min/max, K min/max spin boxes. Returns (group, spins)."""
        group = QtWidgets.QGroupBox(title)
        if checkable:
            group.setCheckable(True)
            group.setChecked(True)
            group.toggled.connect(self._refresh)
        form = QtWidgets.QFormLayout(group)
        h_lo, h_hi, k_lo, k_hi = defaults
        spins = {'h_lo': self._spin(h_lo), 'h_hi': self._spin(h_hi),
                 'k_lo': self._spin(k_lo), 'k_hi': self._spin(k_hi)}
        for label, lo, hi in (('H', spins['h_lo'], spins['h_hi']),
                              ('K', spins['k_lo'], spins['k_hi'])):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel('min'))
            row.addWidget(lo)
            row.addWidget(QtWidgets.QLabel('max'))
            row.addWidget(hi)
            form.addRow(f'{label}:', self._wrap(row))
        return group, spins

    @staticmethod
    def _wrap(layout):
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    @staticmethod
    def _box_values(spins):
        return (spins['h_lo'].value(), spins['h_hi'].value(),
                spins['k_lo'].value(), spins['k_hi'].value())

    def _fill_natural(self):
        for spins in (self.int_spins, self.bkg_spins):
            for key, value in zip(('h_lo', 'h_hi', 'k_lo', 'k_hi'),
                                  self._natural):
                spins[key].blockSignals(True)
                spins[key].setValue(value)
                spins[key].blockSignals(False)
        self._refresh()

    def _refresh(self):
        bkg = (self._box_values(self.bkg_spins)
               if self.bkg_group.isChecked() else None)
        try:
            L, intensity = integrate_rod(self.data, self.H, self.K, self.L,
                                         self._box_values(self.int_spins), bkg)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._last = (L, intensity)
        self.profile.clear()
        self.profile.plot(L, intensity, pen=self.pg.mkPen((220, 40, 40),
                                                          width=2))
        h_lo, h_hi, k_lo, k_hi = self._box_values(self.int_spins)
        self.status.setText(
            f'Integration H [{h_lo:.4f}, {h_hi:.4f}], K [{k_lo:.4f}, {k_hi:.4f}]'
            + ('  -  background subtracted' if bkg is not None else ''))

    def _save(self):
        if self._last is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save rod profile', 'rod_profile.csv', 'CSV (*.csv)')
        if not path:
            return
        if not path.endswith('.csv'):
            path += '.csv'
        L, intensity = self._last
        np.savetxt(path, np.column_stack([L, intensity]), delimiter=',',
                   header='L,intensity', comments='')


# ----------------------------------------------------------------------
# Main dialog
# ----------------------------------------------------------------------

class CTRDialog(QtWidgets.QDialog):
    def __init__(self, datasets, substrates, parent=None, lattice_file=None,
                 default_normal=None):
        super().__init__(parent)
        self.setWindowTitle('CTR rods - high-resolution HKL rod reconstruction')
        self.datasets = list(datasets)
        self.lattice_file = lattice_file
        self.metadata = None
        self.peaks = []
        self.projection = None
        self._busy = False
        self._peaks_window = None
        self._projection_window = None
        layout = QtWidgets.QVBoxLayout(self)

        scans = ', '.join(str(ds.scan_number) for ds in self.datasets)
        header = QtWidgets.QLabel(f'<b>{len(self.datasets)} scan(s):</b> {scans}')
        header.setWordWrap(True)
        layout.addWidget(header)

        # --- Step 1: Find UB --------------------------------------------------
        ub_box = QtWidgets.QGroupBox('1. Find UB (re-index against substrate)')
        ub_form = QtWidgets.QFormLayout(ub_box)
        self.substrate = QtWidgets.QComboBox()
        self.substrate.addItems(list(substrates))
        ub_form.addRow('Substrate:', self.substrate)
        normal = default_normal or workflow.DEFAULT_NORMAL
        self.normal = QtWidgets.QLineEdit(' '.join(
            str(int(v)) if float(v).is_integer() else f'{v:g}' for v in normal))
        self.normal.setPlaceholderText('0 0 1')
        self.normal.setToolTip('Substrate surface normal (h k l); pins U for a '
                               'reproducible orientation. Blank = unconstrained.')
        ub_form.addRow('Surface normal:', self.normal)
        button_row = QtWidgets.QHBoxLayout()
        self.find_button = QtWidgets.QPushButton('Find UB')
        self.find_button.clicked.connect(self._find_ub)
        self.peaks_button = QtWidgets.QPushButton('Found peaks...')
        self.peaks_button.clicked.connect(self._show_peaks)
        self.peaks_button.setEnabled(False)
        self.projection_button = QtWidgets.QPushButton('Projection...')
        self.projection_button.clicked.connect(self._show_projection)
        self.projection_button.setEnabled(False)
        for widget in (self.find_button, self.peaks_button,
                       self.projection_button):
            button_row.addWidget(widget)
        button_row.addStretch()
        ub_form.addRow(button_row)
        self.ub_status = QtWidgets.QLabel('No UB yet -- run Find UB.')
        self.ub_status.setWordWrap(True)
        ub_form.addRow(self.ub_status)
        layout.addWidget(ub_box)

        # --- Step 2: rods -----------------------------------------------------
        hk_box = QtWidgets.QGroupBox('2. HKL rods')
        hk_layout = QtWidgets.QVBoxLayout(hk_box)
        range_row = QtWidgets.QHBoxLayout()
        self.h_min, self.h_max = self._int_spin(-2), self._int_spin(2)
        self.k_min, self.k_max = self._int_spin(-2), self._int_spin(2)
        for label, widget in (('H from', self.h_min), ('to', self.h_max),
                              ('  K from', self.k_min), ('to', self.k_max)):
            range_row.addWidget(QtWidgets.QLabel(label))
            range_row.addWidget(widget)
        populate = QtWidgets.QPushButton('Populate pairs')
        populate.clicked.connect(self._populate_pairs)
        range_row.addWidget(populate)
        self.from_peaks = QtWidgets.QPushButton('From indexed peaks')
        self.from_peaks.setToolTip('Add a rod at every distinct (h, k) of the '
                                   'indexed inlier peaks.')
        self.from_peaks.clicked.connect(self._populate_from_peaks)
        self.from_peaks.setEnabled(False)
        range_row.addWidget(self.from_peaks)
        range_row.addStretch()
        hk_layout.addLayout(range_row)

        self.pairs = QtWidgets.QListWidget()
        self.pairs.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.pairs.setFixedHeight(110)
        hk_layout.addWidget(self.pairs)
        remove_row = QtWidgets.QHBoxLayout()
        remove = QtWidgets.QPushButton('Remove selected')
        remove.clicked.connect(self._remove_selected)
        self.pair_count = QtWidgets.QLabel('0 rods')
        remove_row.addWidget(remove)
        remove_row.addStretch()
        remove_row.addWidget(self.pair_count)
        hk_layout.addLayout(remove_row)
        layout.addWidget(hk_box)

        # --- L grid + per-rod windows ----------------------------------------
        grid = QtWidgets.QGridLayout()
        for col, head in enumerate(('', 'Minimum', 'Maximum', 'Points')):
            grid.addWidget(QtWidgets.QLabel(head), 0, col)
        self.l_lo, self.l_hi, self.l_pts = (self._float_spin(0.0),
                                            self._float_spin(6.0),
                                            self._count_spin(2000))
        self.h_lo, self.h_hi, self.h_pts = (self._float_spin(-0.1),
                                            self._float_spin(0.1),
                                            self._count_spin(100))
        self.k_lo, self.k_hi, self.k_pts = (self._float_spin(-0.1),
                                            self._float_spin(0.1),
                                            self._count_spin(100))
        for row, (label, lo, hi, pts) in enumerate(
                (('L (full range)', self.l_lo, self.l_hi, self.l_pts),
                 ('H window (about h)', self.h_lo, self.h_hi, self.h_pts),
                 ('K window (about k)', self.k_lo, self.k_hi, self.k_pts)), 1):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(lo, row, 1)
            grid.addWidget(hi, row, 2)
            grid.addWidget(pts, row, 3)
        layout.addLayout(grid)

        # --- coordinates (rotate to new directions) --------------------------
        coord_box = QtWidgets.QGroupBox(
            'Coordinates (UB; rod axes run along these crystal directions)')
        coord_form = QtWidgets.QFormLayout(coord_box)
        self.directions = {}
        for axis, default in (('H', '1 0 0'), ('K', '0 1 0'), ('L', '0 0 1')):
            edit = QtWidgets.QLineEdit(default)
            edit.setPlaceholderText(default)
            self.directions[axis] = edit
            coord_form.addRow(f'{axis} direction [u v w]:', edit)
        layout.addWidget(coord_box)

        tag_form = QtWidgets.QFormLayout()
        self.tag = QtWidgets.QLineEdit('rods_r01')
        tag_form.addRow('Output tag:', self.tag)
        layout.addLayout(tag_form)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept_checked)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._ok().setEnabled(False)             # need a UB first
        self._populate_pairs()

    # -- widget factories ----------------------------------------------------

    @staticmethod
    def _int_spin(value):
        box = QtWidgets.QSpinBox()
        box.setRange(-50, 50)
        box.setValue(value)
        return box

    @staticmethod
    def _float_spin(value):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(-10000, 10000)
        box.setDecimals(4)
        box.setValue(value)
        return box

    @staticmethod
    def _count_spin(value):
        box = QtWidgets.QSpinBox()
        box.setRange(2, 100000)
        box.setValue(value)
        return box

    def _ok(self):
        return self.buttons.button(
            QtWidgets.QDialogButtonBox.Ok)

    # -- Find UB -------------------------------------------------------------

    def _find_ub(self):
        if self._busy:
            return
        substrate = self.substrate.currentText()
        text = self.normal.text().strip()
        try:
            normal = workflow.parse_direction(text) if text else None
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid normal', str(exc))
            return
        if not self.lattice_file:
            QtWidgets.QMessageBox.warning(self, 'No lattice file',
                                          'Lattice file path was not provided.')
            return
        self._set_busy(True)
        self.ub_status.setText(f'Indexing {len(self.datasets)} scan(s) ...')
        task = _Task(lambda status: index_group(
            self.datasets, self.lattice_file, substrate, normal, status))
        task.signals.status.connect(self.ub_status.setText)
        task.signals.done.connect(self._find_ub_done)
        task.signals.error.connect(self._find_ub_error)
        QtCore.QThreadPool.globalInstance().start(task)

    def _find_ub_done(self, result):
        self.metadata = result['metadata']
        self.peaks = result['peaks']
        self.projection = result['projection']
        n_in = sum(1 for p in self.peaks if p['inlier'])
        self.ub_status.setText(
            f"UB averaged over {self.metadata['n_averaged']} scan(s): "
            f"{self.metadata['substrate']}, {n_in}/{len(self.peaks)} indexed "
            f"peaks.")
        self.peaks_button.setEnabled(bool(self.peaks))
        self.from_peaks.setEnabled(any(p['hkl'] for p in self.peaks))
        self.projection_button.setEnabled(self.projection is not None)
        self._set_busy(False)
        self._ok().setEnabled(True)

    def _find_ub_error(self, detail):
        self.ub_status.setText(detail.strip().splitlines()[-1])
        self._set_busy(False)
        QtWidgets.QMessageBox.critical(self, 'Find UB failed', detail)

    def _set_busy(self, busy):
        self._busy = busy
        self.find_button.setEnabled(not busy)
        self.buttons.button(
            QtWidgets.QDialogButtonBox.Cancel).setEnabled(
                not busy)

    # -- helper windows ------------------------------------------------------

    def _show_peaks(self):
        if self._peaks_window is None:
            self._peaks_window = PeaksWindow(self.peaks, self._add_pairs, self)
            self._peaks_window.destroyed.connect(
                lambda: setattr(self, '_peaks_window', None))
        self._peaks_window.show()
        self._peaks_window.raise_()

    def _show_projection(self):
        if self.projection is None:
            return
        if self._projection_window is None:
            self._projection_window = ProjectionWindow(self.projection, self)
            self._projection_window.destroyed.connect(
                lambda: setattr(self, '_projection_window', None))
        self._projection_window.show()
        self._projection_window.raise_()

    # -- rod list ------------------------------------------------------------

    def _add_pairs(self, pairs):
        existing = set(self._current_pairs())
        for pair in pairs:
            if pair not in existing:
                existing.add(pair)
                item = QtWidgets.QListWidgetItem(f'({pair[0]}, {pair[1]})')
                item.setData(QtCore.Qt.UserRole, pair)
                self.pairs.addItem(item)
        self._update_count()

    def _populate_pairs(self):
        try:
            pairs = workflow.hkl_pairs(
                (self.h_min.value(), self.h_max.value()),
                (self.k_min.value(), self.k_max.value()))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid range', str(exc))
            return
        self.pairs.clear()
        for h, k in pairs:
            item = QtWidgets.QListWidgetItem(f'({h}, {k})')
            item.setData(QtCore.Qt.UserRole, (h, k))
            self.pairs.addItem(item)
        self._update_count()

    def _populate_from_peaks(self):
        seen, pairs = set(), []
        for peak in self.peaks:
            hkl = peak['hkl']
            if hkl is not None and (hkl[0], hkl[1]) not in seen:
                seen.add((hkl[0], hkl[1]))
                pairs.append((int(hkl[0]), int(hkl[1])))
        self._add_pairs(sorted(pairs))

    def _remove_selected(self):
        for item in self.pairs.selectedItems():
            self.pairs.takeItem(self.pairs.row(item))
        self._update_count()

    def _update_count(self):
        self.pair_count.setText(f'{self.pairs.count()} rods')

    def _current_pairs(self):
        return [self.pairs.item(row).data(QtCore.Qt.UserRole)
                for row in range(self.pairs.count())]

    # -- accept --------------------------------------------------------------

    def _accept_checked(self):
        try:
            self.values()
            self.accept()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid CTR setup', str(exc))

    def values(self):
        """Return the validated spec dict consumed by the monitor."""
        if self.metadata is None:
            raise ValueError('run Find UB before reconstructing rods')
        pairs = self._current_pairs()
        if not pairs:
            raise ValueError('no (h, k) rods selected')
        windows = {'L': (self.l_lo.value(), self.l_hi.value()),
                   'H': (self.h_lo.value(), self.h_hi.value()),
                   'K': (self.k_lo.value(), self.k_hi.value())}
        for axis, (lo, hi) in windows.items():
            if lo >= hi:
                raise ValueError(f'{axis} minimum must be below maximum')
        directions = tuple(workflow.parse_direction(self.directions[axis].text())
                           for axis in ('H', 'K', 'L'))
        workflow.orientation_matrix(*directions)     # raises if not orthogonal
        identity = np.allclose(np.column_stack(directions), np.eye(3))
        tag = self.tag.text().strip()
        if not tag:
            raise ValueError('output tag cannot be empty')
        return {
            'metadata': self.metadata,
            'orientation': None if identity else directions,
            'pairs': pairs,
            'l_range': windows['L'], 'l_points': self.l_pts.value(),
            'h_window': windows['H'], 'h_points': self.h_pts.value(),
            'k_window': windows['K'], 'k_points': self.k_pts.value(),
            'output_tag': tag,
        }
