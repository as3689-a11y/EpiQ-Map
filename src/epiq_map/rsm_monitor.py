#!/usr/bin/env python3
"""Qt beamtime monitor with substrate indexing and HKL reconstruction.

Created by Ben Gregory, Timo Fuchs, and Andrej Singer, Cornell University.
"""

import argparse
import ast
import getpass
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from importlib.resources import files

from qtpy import QtCore, QtGui, QtWidgets

from . import rsm_workflow as workflow
from . import rsm_viewer_ctr as ctr


def autorsm_command(opts, config_path):
    """Build the autoRSM invocation for a config file.

    The command is constructed directly from configured paths -- there is no
    intermediate command list. A single autoRSM handles both unindexed maps
    and indexed (U/UB) maps with optional custom ranges; which one runs is
    decided by the keys present in ``config_path``.
    """
    return _python_command(opts['python'], opts['autorsm'], config_path)


def make_log_files_command(opts):
    """Command to (re)build the per-scan config/log files: walk the raw tree
    and read the SPEC file to identify and classify scans. Discovery only --
    it writes log files, it does not convert anything."""
    return _python_command(
        opts['python'], opts['make_log_files'],
        '--base-dir', opts['base_dir'], '--spec-dir', opts['spec_dir'],
        '--output-dir', opts['output_dir'], '--poni-file', opts['poni_file'],
        '--mask-file', opts['mask_file'],
        '--max-intensity', repr(opts['max_intensity']))


def _python_command(python, target, *args):
    """Build a command for either a module target or an external script."""
    if target.startswith('-m '):
        return [python, '-m', target[3:].strip(), *args]
    return [python, target, *args]


def _dialog_exec(dialog):
    """Run a modal dialog across Qt 5 and Qt 6 bindings."""
    return dialog.exec_()


def _application_exec(app):
    """Run the Qt event loop across Qt 5 and Qt 6 bindings."""
    return app.exec_()


# autoRSM reports per-frame progress with tqdm, whose bars look like
#   scan 18:  45%|####5     | 9/20 [00:03<00:04,  2.7it/s]
# refreshed in place with carriage returns. We stream the process output,
# split on either newline or carriage return, and pull the live percentage
# out of each refreshed bar so the GUI can show a progress bar.
_TQDM_RE = re.compile(r'(\d+)%\|.*?(\d+)/(\d+)')


def _iter_segments(stream):
    """Yield text delimited by newline OR carriage return, so tqdm's
    \\r-refreshed progress arrives one update at a time rather than in one
    blob at the end."""
    buf = []
    while True:
        ch = stream.read(1)
        if not ch:
            if buf:
                yield ''.join(buf)
            return
        if ch in '\r\n':
            if buf:
                yield ''.join(buf)
                buf = []
        else:
            buf.append(ch)


def _parse_tqdm(segment):
    """Return ``(percent, 'n/total')`` from a tqdm bar line, else ``None``."""
    match = _TQDM_RE.search(segment)
    if not match:
        return None
    percent, n, total = (int(group) for group in match.groups())
    return min(100, percent), f'{n}/{total}'


def run_autorsm(command, on_progress):
    """Run autoRSM, forwarding live tqdm progress to ``on_progress(percent,
    text)`` as it streams. Returns ``(returncode, non_progress_output)`` --
    the non-progress lines (config echo, 'Saved ...', errors) joined for the
    caller to parse or report."""
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for segment in _iter_segments(proc.stdout):
        parsed = _parse_tqdm(segment)
        if parsed:
            on_progress(*parsed)
        elif segment.strip():
            lines.append(segment.strip())
    proc.wait()
    return proc.returncode, '\n'.join(lines)


def validate_opts(opts):
    """Return a list of human-readable problems with the configured paths.

    Catches the common per-beamtime config mistakes early -- a stale mask or
    poni left pointing at the previous beamtime, a wrong base/spec directory,
    an empty raw tree -- so they surface at launch with a clear message
    instead of deep inside autoRSM at conversion time. Empty list means OK.
    """
    problems = []
    # Files autoRSM and the helper tools must be able to open.
    for key in ('poni_file', 'mask_file', 'autorsm', 'autorsm_rods',
                'make_log_files', 'lattice_file'):
        path = opts.get(key)
        if path and path.startswith('-m '):
            continue
        if path and not os.path.isfile(path):
            problems.append(f'{key}: file not found: {path}')
    # The interpreter: a bare name is resolved on PATH; a path must exist.
    python = opts.get('python')
    if python and os.sep in python and not os.path.isfile(python):
        problems.append(f'python: interpreter not found: {python}')
    elif python and os.sep not in python and shutil.which(python) is None:
        problems.append(f'python: not found on PATH: {python}')
    # Directories that must exist; the raw tree and spec dir must be non-empty.
    for key in ('base_dir', 'spec_dir'):
        path = opts.get(key)
        if path and not os.path.isdir(path):
            problems.append(f'{key}: directory not found: {path}')
        elif path and not os.listdir(path):
            problems.append(f'{key}: directory is empty: {path}')
    # output_dir is created on demand, so only its parent needs to be writable.
    out = opts.get('output_dir')
    if out and not os.path.isdir(out):
        parent = os.path.dirname(os.path.abspath(out.rstrip(os.sep)))
        if not os.path.isdir(parent):
            problems.append(f'output_dir: parent does not exist: {parent}')
        elif not os.access(parent, os.W_OK):
            problems.append(f'output_dir: parent not writable: {parent}')
    return problems


class Dataset:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = self._parse(config_path)

    @staticmethod
    def _parse(path):
        cfg = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if ': ' in line and not line.startswith('#'):
                    key, value = line.split(': ', 1)
                    cfg[key] = value
        return cfg

    @property
    def scans(self):
        return (list(ast.literal_eval(self.cfg.get('Scan List', '[]'))) +
                list(ast.literal_eval(self.cfg.get('Theta Scan List', '[]'))))

    @property
    def scan_number(self):
        return int(self.scans[0]) if self.scans else None

    @property
    def is_theta_only(self):
        return (not ast.literal_eval(self.cfg.get('Scan List', '[]')) and
                bool(ast.literal_eval(self.cfg.get('Theta Scan List', '[]'))))

    @property
    def scan_kind(self):
        """'theta' (rocking) or 'phi' (rotation), read from the SPEC-derived
        scan lists -- the same classification make_log_files writes."""
        return 'theta' if self.is_theta_only else 'phi'

    @property
    def image_count(self):
        """Number of .cbf frames currently in the scan's image directory, or
        None if it can't be read. Counted live (so it grows as a scan runs)
        and memoized per Dataset instance."""
        if not hasattr(self, '_image_count'):
            image_dir = self.cfg.get('Image Directory')
            count = None
            if image_dir and os.path.isdir(image_dir):
                try:
                    count = sum(1 for entry in os.scandir(image_dir)
                                if entry.name.endswith('cbf'))
                except OSError:
                    count = None
            self._image_count = count
        return self._image_count

    @property
    def label(self):
        scans = ','.join(map(str, self.scans))
        images = self.image_count
        img_text = f'{images} imgs' if images is not None else '? imgs'
        return (f"{self.cfg['Material']} / {self.cfg['Sample Name']} / "
                f"{self.cfg.get('Temperature', '?')} / scan {scans} "
                f"({self.scan_kind}, {img_text})")

    def _output_candidates(self):
        """The transformed-map output paths to look for, newest naming first:
        autoRSM now writes the unindexed map as '_full.nxs'; '_out.nxs' is the
        legacy name, kept so conversions from before the rename still resolve.
        """
        scan_text = '_'.join(map(str, self.scans))
        stem = (f"{self.cfg['Material']}_{self.cfg['Sample Name']}_scans_"
                f"{scan_text}")
        directory = os.path.join(self.cfg['Output Directory'],
                                 'transformed_objects')
        return [os.path.join(directory, stem + '_full.nxs'),
                os.path.join(directory, stem + '_out.nxs')]

    def base_output(self):
        return self._output_candidates()[0]

    def resolved_output(self):
        for path in self._output_candidates():
            if os.path.exists(path):
                return path
            stem = path[:-4]
            for _ in range(50):
                stem += '_more'
                if os.path.exists(stem + '.nxs'):
                    return stem + '.nxs'
        return None

    def u_base(self):
        output = self.resolved_output()
        return output[:-4] + '_U_S' if output else None

    def u_path(self):
        """Newest complete substrate-derived U_S text record, if present."""
        metadata = self.metadata_path()
        return metadata[:-5] + '.txt' if metadata else None

    def next_u_path(self):
        """A new versioned U_S path; existing U and U_S files are untouched."""
        base = self.u_base()
        if base is None:
            return None
        version = 1
        while True:
            suffix = '' if version == 1 else f'_{version:02d}'
            text = base + suffix + '.txt'
            record = base + suffix + '.json'
            ub_text = text.replace('_U_S', '_UB_S')
            if not any(os.path.exists(path)
                       for path in (text, record, ub_text)):
                return text
            version += 1

    def metadata_path(self):
        records = self.metadata_records()
        return records[-1][1] if records else None

    def metadata_records(self):
        """All complete U_S versions as ``(version, json_path, metadata)``."""
        base = self.u_base()
        if base is None:
            return []
        pattern = re.compile(re.escape(base) + r'(?:_(\d+))?\.json$')
        records = []
        for path in glob.glob(base + '*.json'):
            match = pattern.fullmatch(path)
            text_path = path[:-5] + '.txt'
            if match and os.path.exists(text_path):
                try:
                    with open(path) as fh:
                        metadata = json.load(fh)
                    records.append((int(match.group(1) or 1), path, metadata))
                except (OSError, ValueError):
                    continue
        return sorted(records)

    def attempt_path(self):
        output = self.resolved_output()
        return output[:-4] + '_U_S_attempt.json' if output else None

    def metadata(self):
        records = self.metadata_records()
        return records[-1][2] if records else None


class TaskSignals(QtCore.QObject):
    status = QtCore.Signal(str)
    success = QtCore.Signal(object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()


class FunctionTask(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    @QtCore.Slot()
    def run(self):
        try:
            result = self.function(self.signals.status.emit)
            self.signals.success.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class IndexDialog(QtWidgets.QDialog):
    def __init__(self, entries, parent=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle('Find U from substrate')
        form = QtWidgets.QFormLayout(self)
        self.substrate = QtWidgets.QComboBox()
        self.substrate.addItems(entries)
        if initial and initial.get('substrate') in entries:
            self.substrate.setCurrentText(initial['substrate'])
        form.addRow('Substrate:', self.substrate)
        # Same finder as rsm_viewer's "Calculate U": a substrate surface normal
        # (h k l) pins the indexed frame reproducibly; two in-plane axes are
        # derived from it. Blank leaves the orientation unconstrained.
        normal = (initial or {}).get('normal') or workflow.DEFAULT_NORMAL
        self.normal = QtWidgets.QLineEdit(' '.join(
            str(int(v)) if float(v).is_integer() else f'{v:g}' for v in normal))
        self.normal.setPlaceholderText('0 0 1')
        self.normal.setToolTip('Substrate surface normal (h k l); pins U for a '
                               'reproducible orientation. Blank = unconstrained.')
        form.addRow('Surface normal:', self.normal)
        self.save_ub = QtWidgets.QCheckBox(
            'Also save scaled UB_S matrix (r.l.u.)')
        self.save_ub.setChecked((initial or {}).get('save_scaled_ub', True))
        form.addRow('', self.save_ub)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        text = self.normal.text().strip()
        normal = workflow.parse_direction(text) if text else None
        return self.substrate.currentText(), normal, self.save_ub.isChecked()


class DimensionsDialog(QtWidgets.QDialog):
    """Choose U/UB and optional grids for the existing compatible autoRSM."""

    def __init__(self, records, parent=None, batch=False):
        super().__init__(parent)
        self.batch = batch
        self.setWindowTitle('Batch U/UB on selected scans' if batch
                            else 'Run existing autoRSM with U or UB')
        layout = QtWidgets.QVBoxLayout(self)
        if batch:
            note = QtWidgets.QLabel(
                'Batch mode: each selected scan uses its own latest U_S record. '
                'The transfer matrix, grid, orientation, and tag below apply to '
                'all selected scans (the orientation shown is the first scan\'s, '
                'for reference).')
            note.setWordWrap(True)
            note.setStyleSheet('color: #c8a200;')
            layout.addWidget(note)
        form = QtWidgets.QFormLayout()

        self.u_selector = QtWidgets.QComboBox()
        for version, path, metadata in reversed(records):
            substrate = metadata.get('substrate', '?')
            inliers = metadata.get('n_inliers', '?')
            total = metadata.get('n_peaks', '?')
            rms = metadata.get('rms_A^-1')
            rms_text = f'{rms:.4g}' if isinstance(rms, (int, float)) else '?'
            label = (f'{os.path.basename(path)[:-5]} | {substrate} | '
                     f'{inliers}/{total} inliers | RMS {rms_text}')
            self.u_selector.addItem(label, (version, path, metadata))
        if batch:
            self.u_selector.setEnabled(False)
        form.addRow('Orientation (scan 1):' if batch else 'Saved orientation:',
                    self.u_selector)

        self.matrix_type = QtWidgets.QComboBox()
        self.matrix_type.addItem(
            'UB - scaled reciprocal basis; ranges are in r.l.u.', 'UB')
        self.matrix_type.addItem(
            'U - orientation only; ranges are in inverse angstrom', 'U')
        form.addRow('Transfer matrix:', self.matrix_type)

        self.custom_grid = QtWidgets.QCheckBox(
            'Use custom H/K/L ranges and grid shape')
        # Default to editable ranges, like rsm_viewer's oriented-axis grid;
        # uncheck to let autoRSM pick its automatic ranges and 1000^3 grid.
        self.custom_grid.setChecked(True)
        form.addRow('', self.custom_grid)
        layout.addLayout(form)

        # Oriented Q axes, like rsm_viewer: each output axis carries a crystal
        # direction [u v w] plus its range cut. Default Q1=[100], Q2=[010],
        # Q3=[001] reproduces the unrotated H/K/L frame.
        grid = QtWidgets.QGridLayout()
        for col, header in enumerate(
                ('Axis', 'Direction [u v w]', 'Minimum', 'Maximum', 'Points')):
            grid.addWidget(QtWidgets.QLabel(header), 0, col)
        defaults = {'H': ('1 0 0', -4, 4, 300),
                    'K': ('0 1 0', -4, 4, 300),
                    'L': ('0 0 1', 0, 6, 300)}
        labels = {'H': 'Q1 (H)', 'K': 'Q2 (K)', 'L': 'Q3 (L)'}
        self.axes = {}
        self.directions = {}
        for row, axis in enumerate(('H', 'K', 'L'), 1):
            direction = QtWidgets.QLineEdit(defaults[axis][0])
            direction.setPlaceholderText('1 0 0')
            lo = QtWidgets.QDoubleSpinBox()
            hi = QtWidgets.QDoubleSpinBox()
            for box in (lo, hi):
                box.setRange(-10000, 10000)
                box.setDecimals(5)
            count = QtWidgets.QSpinBox()
            count.setRange(2, 4000)
            lo.setValue(defaults[axis][1])
            hi.setValue(defaults[axis][2])
            count.setValue(defaults[axis][3])
            self.axes[axis] = (lo, hi, count)
            self.directions[axis] = direction
            grid.addWidget(QtWidgets.QLabel(labels[axis]), row, 0)
            grid.addWidget(direction, row, 1)
            grid.addWidget(lo, row, 2)
            grid.addWidget(hi, row, 3)
            grid.addWidget(count, row, 4)
        layout.addLayout(grid)

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        tag_form = QtWidgets.QFormLayout()
        self.tag = QtWidgets.QLineEdit()
        tag_form.addRow('Audit tag:', self.tag)
        layout.addLayout(tag_form)

        self.u_selector.currentIndexChanged.connect(self._set_default_tag)
        self.matrix_type.currentIndexChanged.connect(self._update_state)
        self.custom_grid.toggled.connect(self._update_state)
        self._update_state()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_default_tag(self):
        if self.u_selector.count() == 0:
            return
        version, _path, metadata = self.u_selector.currentData()
        substrate = metadata.get('substrate', 'substrate')
        matrix = self.matrix_type.currentData()
        self.tag.setText(f'{matrix}{version:02d}_{substrate}_r01')

    def _update_state(self):
        custom = self.custom_grid.isChecked()
        for widgets in self.axes.values():
            for widget in widgets:
                widget.setEnabled(custom)
        units = ('r.l.u.' if self.matrix_type.currentData() == 'UB'
                 else 'inverse angstrom')
        if custom:
            self.info.setText(f'Custom H/K/L ranges are in {units}.')
        else:
            self.info.setText(
                'No grid data will be added: existing autoRSM uses its '
                'original automatic ranges and 1000 x 1000 x 1000 grid.')
        self._set_default_tag()

    def _accept_checked(self):
        try:
            self.values()
            self.accept()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid selection', str(exc))

    def values(self):
        version, path, metadata = self.u_selector.currentData()
        selected = dict(metadata, u_s_record=os.path.abspath(path),
                        u_s_version=version)
        directions = tuple(workflow.parse_direction(self.directions[axis].text())
                           for axis in ('H', 'K', 'L'))
        # Validate orthogonality up front (raises on a bad Q1/Q2/Q3 triple).
        workflow.orientation_matrix(*directions)
        ranges = {}
        shape = []
        for axis, (lo, hi, count) in self.axes.items():
            if lo.value() >= hi.value():
                raise ValueError(f'{axis} minimum must be below maximum')
            ranges[axis] = (lo.value(), hi.value())
            shape.append(count.value())
        tag = self.tag.text().strip()
        if not tag:
            raise ValueError('audit tag cannot be empty')
        return (selected, self.matrix_type.currentData(),
                self.custom_grid.isChecked(), ranges, tuple(shape), tag,
                directions)


class WatcherWorker(QtCore.QObject):
    datasets_updated = QtCore.Signal(list)
    status = QtCore.Signal(str)
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal()

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.stopped = False

    @QtCore.Slot()
    def stop(self):
        self.stopped = True

    def _datasets(self):
        log_dir = os.path.join(self.opts['output_dir'], 'logs')
        result = []
        if os.path.isdir(log_dir):
            for name in sorted(os.listdir(log_dir)):
                is_ledger = (name.startswith('command_list') or
                             name.startswith('processed_commands'))
                if name.endswith('.txt') and not is_ledger:
                    try:
                        result.append(Dataset(os.path.join(log_dir, name)))
                    except Exception as exc:
                        self.status.emit(f'Bad config {name}: {exc}')
        return result

    def _command_for(self, dataset):
        return autorsm_command(self.opts, dataset.config_path)

    def _auto_index(self, datasets):
        by_scan = {ds.scan_number: ds for ds in datasets}
        for ds in datasets:
            if self.stopped or not ds.resolved_output() or ds.metadata():
                continue
            if ds.attempt_path() and os.path.exists(ds.attempt_path()):
                continue
            # Isolate each scan: a manual index can run concurrently and win
            # the race to write a U_S record (FileExistsError), and a single
            # unreadable .nxs should not take the whole watcher down.
            try:
                self._auto_index_one(ds, by_scan)
            except Exception as exc:
                self.status.emit(
                    f'Auto-index skipped scan {ds.scan_number}: {exc}')

    def _auto_index_one(self, ds, by_scan):
        if ds.is_theta_only:
            source = by_scan.get(ds.scan_number - 1)
            if source and source.metadata():
                metadata = source.metadata()
                metadata = dict(metadata, method='copied',
                                copied_from=source.resolved_output(),
                                source_nxs=ds.resolved_output())
                workflow.save_index_metadata(ds.next_u_path(), metadata)
                self.status.emit(f'Copied U into theta scan {ds.scan_number}')
            return
        self.status.emit(f'Auto-indexing {ds.label} ...')
        data, H, K, L = workflow.rl.load_rsm(ds.resolved_output())
        try:
            match = workflow.auto_match_substrate(
                data, H, K, L, self.opts['lattice_file'])
        finally:
            del data
        if match['accepted']:
            name, lattice, result = match['best']
            metadata = workflow.build_index_metadata(
                result, name, lattice,
                ds.resolved_output(), method='automatic')
            metadata['save_scaled_ub'] = True
            workflow.save_index_metadata(ds.next_u_path(), metadata)
            self.status.emit(
                f'Auto-indexed scan {ds.scan_number}: {name}, '
                f"{result['n_inliers']} inliers, RMS {result['rms']:.4g}")
        else:
            summary = {
                'reason': match['reason'],
                'candidates': [
                    {'substrate': name, 'n_inliers': res['n_inliers'],
                     'rms_A^-1': res['rms']}
                    for name, _, res in match['ranked'][:10]],
            }
            with open(ds.attempt_path(), 'w') as fh:
                json.dump(summary, fh, indent=2)
            self.status.emit(
                f'Auto-index scan {ds.scan_number}: {match["reason"]}; '
                'manual choice required')

    @QtCore.Slot()
    def run(self):
        opts = self.opts
        while not self.stopped:
            self.status.emit('Scanning for new datasets ...')
            scan = subprocess.run(make_log_files_command(opts),
                                  capture_output=True, text=True, check=False)
            if scan.returncode:
                self.status.emit(
                    f'make_log_files failed: {scan.stderr.strip()[-300:]}')
            datasets = self._datasets()
            self.datasets_updated.emit(datasets)
            for ds in datasets:
                if self.stopped:
                    break
                if ds.resolved_output():
                    continue
                command = self._command_for(ds)
                if command:
                    workflow.sync_config_paths(
                        ds.config_path, poni_file=opts['poni_file'],
                        mask_file=opts['mask_file'],
                        output_dir=opts['output_dir'], spec_dir=opts['spec_dir'],
                        max_intensity=opts['max_intensity'])
                    self.status.emit(f'Processing {ds.label} ...')
                    label = f'scan {ds.scan_number}'
                    returncode, output = run_autorsm(
                        command,
                        lambda pct, text, lbl=label:
                            self.progress.emit(pct, f'{lbl}  {text}'))
                    self.progress.emit(0, '')
                    if returncode:
                        self.status.emit(
                            f'autoRSM failed for scan {ds.scan_number}: '
                            f'{output[-300:]}')
                    else:
                        note = next((l for l in output.splitlines()
                                     if l.startswith('Total overloaded')), '')
                        if note:
                            self.status.emit(f'scan {ds.scan_number}: {note}')
            if not self.stopped:
                self._auto_index(datasets)
                self.datasets_updated.emit(self._datasets())
            for _ in range(opts['interval'] * 10):
                if self.stopped:
                    break
                QtCore.QThread.msleep(100)
        self.finished.emit()


class MonitorWindow(QtWidgets.QMainWindow):
    stop_watcher = QtCore.Signal()
    # Progress from a manual reconstruction (runs on the thread pool); the
    # watcher has its own progress signal. Both drive the bottom progress bar.
    conversion_progress = QtCore.Signal(int, str)
    COLUMNS = ('Dataset', 'Scan done', 'Config', 'Output', 'Convert',
               'Index / U', 'Reconstruct')

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.datasets = []
        self.watcher_thread = None
        self.watcher = None
        self.busy = False
        self.pool = QtCore.QThreadPool.globalInstance()
        self.lattices = workflow.load_lattice_entries(opts['lattice_file'])
        self._build_ui()
        self.conversion_progress.connect(self.update_progress)
        self.refresh()
        self.message('EpiQ-Map monitor -- created by Ben Gregory, Timo Fuchs, and Andrej Singer, Cornell University')
        self.message(f"autoRSM: {opts['python']} {opts['autorsm']}")
        self.message(f"Scanning logs in: {os.path.join(opts['output_dir'], 'logs')}")
        self._check_config()

    def _check_config(self):
        """Preflight the configured paths and surface any problems at launch."""
        problems = validate_opts(self.opts)
        if not problems:
            self.message('Config check: all data paths present.')
            return
        for problem in problems:
            self.message(f'CONFIG PROBLEM -- {problem}')
        QtWidgets.QMessageBox.warning(
            self, 'Configuration problems',
            'These configured paths look wrong. Fix epiq_monitor.toml (or '
            'override on the command line) and restart:\n\n  - '
            + '\n  - '.join(problems))

    def _build_ui(self):
        self.setWindowTitle('EpiQ-Map_monitor')
        self.resize(1250, 650)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        bar = QtWidgets.QHBoxLayout()
        self.watch_button = QtWidgets.QPushButton('Start watching')
        self.watch_button.setCheckable(True)
        self.watch_button.clicked.connect(self.toggle_watcher)
        bar.addWidget(self.watch_button)
        bar.addWidget(QtWidgets.QLabel('Interval (s):'))
        self.interval = QtWidgets.QSpinBox()
        self.interval.setRange(5, 3600)
        self.interval.setValue(self.opts['interval'])
        bar.addWidget(self.interval)
        self.find_button = QtWidgets.QPushButton('Find scans')
        self.find_button.setToolTip(
            'Walk the base folder and read the SPEC file to (re)build the '
            'per-scan config/log files. Discovery only -- nothing is '
            'converted. Use Refresh to just re-read existing logs.')
        self.find_button.clicked.connect(self.discover)
        bar.addWidget(self.find_button)
        refresh = QtWidgets.QPushButton('Refresh')
        refresh.setToolTip('Re-read the existing log files from disk (no walk, '
                           'no conversion).')
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        self.auto_index_button = QtWidgets.QPushButton('Auto-index missing')
        self.auto_index_button.clicked.connect(self.auto_index_missing)
        bar.addWidget(self.auto_index_button)
        self.batch_convert_button = QtWidgets.QPushButton('Batch Convert')
        self.batch_convert_button.setToolTip(
            'Convert every selected, not-yet-converted scan row in turn -- the '
            'same as clicking Convert on each, but as one queued pass.')
        self.batch_convert_button.clicked.connect(self.batch_convert)
        bar.addWidget(self.batch_convert_button)
        self.batch_reconstruct_button = QtWidgets.QPushButton('Batch U/UB...')
        self.batch_reconstruct_button.setToolTip(
            'Run U/UB on every selected scan row in turn: each uses its own '
            'latest U_S record, with grid/matrix/tag settings chosen once. '
            'Requires every selected scan to be converted and indexed.')
        self.batch_reconstruct_button.clicked.connect(self.batch_reconstruct)
        bar.addWidget(self.batch_reconstruct_button)
        self.ctr_button = QtWidgets.QPushButton('CTR rods...')
        self.ctr_button.setToolTip(
            'Reconstruct high-resolution HKL rods (CTR) for the selected group '
            'of phi scan rows: average their U, define (h, k) rods and the L '
            'grid, then run a single multi-rod conversion.')
        self.ctr_button.clicked.connect(self.ctr_rods)
        bar.addWidget(self.ctr_button)
        bar.addStretch()
        self.count = QtWidgets.QLabel()
        bar.addWidget(self.count)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        for column in range(1, len(self.COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        # Row multi-select so a group of scans can be picked for CTR rods
        # (click the Dataset column; the action columns hold buttons).
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.table)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setFixedHeight(160)
        self.log.setStyleSheet('font-family: monospace; font-size: 11px;')
        layout.addWidget(self.log)
        # Live autoRSM conversion progress (parsed from its tqdm output).
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat('idle')
        layout.addWidget(self.progress)
        self.statusBar().showMessage('Idle')

    def update_progress(self, percent, text):
        """Drive the bottom progress bar from a conversion's tqdm output."""
        self.progress.setValue(percent)
        self.progress.setFormat(f'{text}  %p%' if text else 'idle')

    def message(self, text):
        from datetime import datetime
        self.statusBar().showMessage(text)
        self.log.appendPlainText(f'{datetime.now():%H:%M:%S}  {text}')

    def _load_datasets(self):
        log_dir = os.path.join(self.opts['output_dir'], 'logs')
        datasets = []
        if os.path.isdir(log_dir):
            for name in sorted(os.listdir(log_dir)):
                is_ledger = (name.startswith('command_list') or
                             name.startswith('processed_commands'))
                if name.endswith('.txt') and not is_ledger:
                    try:
                        datasets.append(Dataset(os.path.join(log_dir, name)))
                    except Exception as exc:
                        self.message(f'Bad config {name}: {exc}')
        return datasets

    def refresh(self):
        self.set_datasets(self._load_datasets())

    def discover(self):
        """Run make_log_files once to (re)build all per-scan config/log files
        from the base folder + SPEC file, then refresh -- no conversion."""
        command = make_log_files_command(self.opts)

        def work(status):
            status('Walking base folder and reading SPEC file for scans ...')
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
            found = sum(1 for line in proc.stdout.splitlines()
                        if line.startswith('Processing '))
            written = sum(1 for line in proc.stdout.splitlines()
                          if 'Log file written' in line)
            return (f'Discovery complete: {found} scan(s) found, '
                    f'{written} new config(s) written. Use Convert (or Start '
                    f'watching) to process them.')

        self._start_task(work, 'Finding scans / building config files ...')

    def auto_index_missing(self):
        datasets = self._load_datasets()

        def work(status):
            helper = WatcherWorker(self.opts)
            helper.status.connect(status)
            helper._auto_index(datasets)
            return 'Automatic indexing pass complete'

        self._start_task(work, 'Starting automatic substrate matching ...')

    def _convert_one(self, ds, status):
        """Convert a single scan synchronously (config sync + autoRSM), streaming
        progress through ``status``. Returns a summary string; raises on failure.
        Runs inside a task's worker thread -- shared by Convert and Batch Convert."""
        changed = workflow.sync_config_paths(
            ds.config_path, poni_file=self.opts['poni_file'],
            mask_file=self.opts['mask_file'],
            output_dir=self.opts['output_dir'], spec_dir=self.opts['spec_dir'],
            max_intensity=self.opts['max_intensity'])
        if changed:
            status(f"Updated {', '.join(changed)} in "
                   f'{os.path.basename(ds.config_path)} from current config')
        command = autorsm_command(self.opts, ds.config_path)
        command_text = shlex.join(command)
        status(f'Converting {ds.label} ...')
        status('RUN: ' + command_text)
        returncode, output = run_autorsm(
            command,
            lambda pct, text: self.conversion_progress.emit(
                pct, f'scan {ds.scan_number}  {text}'))
        self.conversion_progress.emit(0, '')
        if returncode:
            raise RuntimeError(output[-2000:])
        saved = [line[6:] for line in output.splitlines()
                 if line.startswith('Saved ')]
        note = next((l for l in output.splitlines()
                     if l.startswith('Total overloaded')), '')
        if note:
            status(note)
        return (f'Converted scan {ds.scan_number}: '
                f'{saved[-1] if saved else "done"}'
                + (f' -- {note}' if note else ''))

    def convert(self, ds):
        """Run autoRSM on a single scan now, without the watcher and without
        waiting for the queue. Streams progress to the bottom bar."""
        if ds.resolved_output():
            self.message(f'Scan {ds.scan_number} is already converted.')
            return
        self._start_task(lambda status: self._convert_one(ds, status),
                         f'Converting {ds.label} ...')

    def batch_convert(self):
        """Convert every selected, not-yet-converted scan in turn -- the same as
        clicking Convert on each row, but as one queued pass. A scan that fails
        is reported and skipped so the rest still run."""
        datasets = [ds for ds in self.selected_datasets()
                    if not ds.resolved_output()]
        if not datasets:
            self.message('Select one or more unconverted scans to batch convert.')
            return
        total = len(datasets)

        def work(status):
            converted, failed = [], []
            for index, ds in enumerate(datasets, 1):
                status(f'[{index}/{total}] {ds.label} ...')
                try:
                    status(f'[{index}/{total}] {self._convert_one(ds, status)}')
                    converted.append(ds.scan_number)
                except Exception as exc:
                    failed.append(ds.scan_number)
                    status(f'Scan {ds.scan_number} failed: '
                           f'{str(exc).splitlines()[-1]}')
            summary = f'Batch convert: {len(converted)}/{total} scan(s) converted'
            if failed:
                summary += f'; failed: {failed}'
            return summary

        self._start_task(work, f'Batch converting {total} scan(s) ...')

    def _check_item(self, yes):
        item = QtWidgets.QTableWidgetItem('yes' if yes else '-')
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setForeground(QtGui.QColor('#27ae60' if yes else '#888888'))
        return item

    def set_datasets(self, datasets):
        self.datasets = datasets
        self.table.setRowCount(len(datasets))
        for row, ds in enumerate(datasets):
            label = QtWidgets.QTableWidgetItem(ds.label)
            label.setToolTip(ds.config_path)
            self.table.setItem(row, 0, label)
            self.table.setItem(row, 1, self._check_item(True))
            self.table.setItem(row, 2, self._check_item(True))
            self.table.setItem(row, 3, self._check_item(bool(ds.resolved_output())))
            convert = QtWidgets.QPushButton('Convert')
            # Convert one scan on demand -- no need to start the watcher or wait
            # for the queue. Disabled once the scan already has an output .nxs.
            convert.setEnabled(not ds.resolved_output() and not self.busy)
            convert.clicked.connect(
                lambda _checked=False, dataset=ds: self.convert(dataset))
            self.table.setCellWidget(row, 4, convert)
            self.table.setCellWidget(row, 5, self._index_widget(ds))
            reconstruct = QtWidgets.QPushButton('Run U / UB...')
            # Enabled for any converted scan, even while the watcher runs:
            # reconstruction acts only on an existing .nxs and writes uniquely
            # named outputs, so it does not collide with the watcher.
            reconstruct.setEnabled(bool(ds.resolved_output()) and not self.busy)
            reconstruct.clicked.connect(
                lambda _checked=False, dataset=ds: self.reconstruct(dataset))
            self.table.setCellWidget(row, 6, reconstruct)
        self.count.setText(f'{len(datasets)} datasets')

    def _index_widget(self, ds):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        metadata = ds.metadata()
        if metadata:
            status = (f"{metadata['substrate']}  {metadata['n_inliers']}/"
                      f"{metadata['n_peaks']}")
            color = '#27ae60'
        elif ds.attempt_path() and os.path.exists(ds.attempt_path()):
            status, color = 'ambiguous', '#c8a200'
        else:
            status, color = '-', '#888888'
        label = QtWidgets.QLabel(status)
        label.setStyleSheet(f'color: {color};')
        layout.addWidget(label)
        button = QtWidgets.QToolButton()
        button.setText('Actions')
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # Per-scan indexing stays available while the watcher runs (see
        # _start_task): it only touches already-converted scans.
        button.setEnabled(bool(ds.resolved_output()) and not self.busy)
        menu = QtWidgets.QMenu(button)
        find_action = menu.addAction('Find new U_S...')
        find_action.triggered.connect(
            lambda _checked=False, dataset=ds: self.find_u(dataset))
        copy_action = menu.addAction('U same as scan...')
        copy_action.triggered.connect(
            lambda _checked=False, dataset=ds: self.copy_u(dataset))
        same_substrate = menu.addAction('Substrate same as scan...')
        same_substrate.triggered.connect(
            lambda _checked=False, dataset=ds: self.same_substrate(dataset))
        button.setMenu(menu)
        layout.addWidget(button)
        return widget

    def _choose_source(self, target, require_metadata=True):
        choices = [ds for ds in self.datasets if ds is not target and
                   (ds.metadata() is not None if require_metadata else True)]
        if not choices:
            QtWidgets.QMessageBox.information(
                self, 'No source scan', 'No other indexed scan is available.')
            return None
        labels = [ds.label for ds in choices]
        selected, ok = QtWidgets.QInputDialog.getItem(
            self, 'Choose source scan', 'Source:', labels, 0, False)
        return choices[labels.index(selected)] if ok else None

    def _start_task(self, function, start_message):
        # A manual index/reconstruct may run alongside the watcher -- it acts
        # only on already-converted scans, and U_S records are written with
        # exclusive create, so a rare overlap with the watcher's auto-index
        # surfaces as an error instead of corrupting a record. Only one manual
        # task at a time, though.
        if self.busy:
            self.message('Wait for the current manual task to finish.')
            return
        self.busy = True
        self.auto_index_button.setEnabled(False)
        self.find_button.setEnabled(False)
        self.message(start_message)
        self.set_datasets(self.datasets)
        task = FunctionTask(function)
        task.signals.status.connect(self.message)
        task.signals.success.connect(
            lambda result: self.message(str(result)) if result else None)
        task.signals.error.connect(self._task_error)
        task.signals.finished.connect(self._task_finished)
        self.pool.start(task)

    def _task_error(self, detail):
        self.message(detail.splitlines()[-1])
        QtWidgets.QMessageBox.critical(self, 'Task failed', detail)

    def _task_finished(self):
        self.busy = False
        # The bulk "Auto-index missing" pass and discovery conflict with the
        # watcher (which does both itself), so keep them disabled while it runs.
        self.auto_index_button.setEnabled(self.watcher is None)
        self.find_button.setEnabled(self.watcher is None)
        self.refresh()

    def find_u(self, ds, initial=None):
        dialog = IndexDialog(list(self.lattices), self, initial=initial)
        if _dialog_exec(dialog) != QtWidgets.QDialog.Accepted:
            return
        try:
            substrate, normal, save_ub = dialog.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid indexing setup', str(exc))
            return

        def work(status):
            status(f'Loading {os.path.basename(ds.resolved_output())} ...')
            data, H, K, L = workflow.rl.load_rsm(ds.resolved_output())
            try:
                # The exact rsm_viewer finder: find peaks, then index against
                # the substrate cell with the surface normal pinning U.
                result = workflow.rl.compute_U_from_substrate(
                    data, H, K, L, self.opts['lattice_file'], substrate,
                    normal=normal, verbose=False)
            finally:
                del data
            if result is None:
                raise RuntimeError(f'No consistent {substrate} indexing found')
            metadata = workflow.build_index_metadata(
                result, substrate, self.lattices[substrate],
                ds.resolved_output(), method='manual', normal=normal)
            metadata['save_scaled_ub'] = save_ub
            saved_path = ds.next_u_path()
            workflow.save_index_metadata(saved_path, metadata)
            if ds.attempt_path() and os.path.exists(ds.attempt_path()):
                os.remove(ds.attempt_path())
            return (f"Saved {os.path.basename(saved_path)}: {substrate}, "
                    f"{result['n_inliers']}/"
                    f"{len(result['peaks'])} inliers, RMS {result['rms']:.5f}")

        self._start_task(work, f'Finding U for {ds.label} ...')

    def copy_u(self, ds):
        source = self._choose_source(ds)
        if source is None:
            return
        metadata = source.metadata()
        metadata = dict(metadata, method='copied',
                        copied_from=source.resolved_output(),
                        source_nxs=ds.resolved_output())
        saved_path = ds.next_u_path()
        try:
            workflow.save_index_metadata(saved_path, metadata)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, 'Copy U failed', str(exc))
            return
        self.message(f'Copied U from scan {source.scan_number} to {ds.scan_number}')
        self.refresh()

    def same_substrate(self, ds):
        source = self._choose_source(ds)
        if source is not None:
            self.find_u(ds, initial=source.metadata())

    def reconstruct(self, ds):
        records = ds.metadata_records()
        if not records:
            answer = QtWidgets.QMessageBox.question(
                self, 'U_S required',
                'Dimensions / Run uses a substrate-derived U_S record. '
                'No U_S exists for this scan. Find one now?')
            if answer == QtWidgets.QMessageBox.Yes:
                self.find_u(ds)
            return
        dialog = DimensionsDialog(records, self)
        if _dialog_exec(dialog) != QtWidgets.QDialog.Accepted:
            return
        try:
            (metadata, matrix_type, custom_grid, ranges, shape, tag,
             orientation) = dialog.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid selection', str(exc))
            return
        spec = (matrix_type, custom_grid, ranges, shape, tag, orientation)
        self._start_task(
            lambda status: self._reconstruct_one(ds, metadata, spec, status),
            f'Starting indexed reconstruction for {ds.label}')

    def _reconstruct_one(self, ds, metadata, spec, status):
        """Write a per-scan reconstruction config for ``metadata`` (a chosen U_S
        record) and run the existing autoRSM, streaming progress through
        ``status``. Returns a summary; raises on failure. Shared by Run U/UB and
        Batch U/UB. ``spec`` = (matrix_type, custom_grid, ranges, shape, tag,
        orientation)."""
        matrix_type, custom_grid, ranges, shape, tag, orientation = spec
        run_label = self.opts['run_label']
        recon_dir = os.path.join(self.opts['output_dir'], 'logs',
                                 'reconstructions', run_label)
        config_name = (os.path.splitext(os.path.basename(ds.config_path))[0] +
                       f'_{tag}.txt')
        config_path = workflow.next_available_path(
            os.path.join(recon_dir, config_name))
        workflow.write_reconstruction_config(
            ds.config_path, config_path, metadata, ranges, shape, tag,
            custom_grid=custom_grid, matrix_type=matrix_type,
            orientation=orientation, max_intensity=self.opts['max_intensity'])

        command = autorsm_command(self.opts, config_path)
        command_text = shlex.join(command)
        log_dir = os.path.join(self.opts['output_dir'], 'logs')
        command_list = os.path.join(
            log_dir, f'command_list_indexed_{run_label}.txt')
        processed_list = os.path.join(
            log_dir, f'processed_commands_indexed_{run_label}.txt')
        workflow.append_unique_line(command_list, command_text)
        status(f'Reconstructing scan {ds.scan_number} as {tag} with '
               f'the existing autoRSM ({matrix_type}) ...')
        status('RUN: ' + command_text)
        returncode, output = run_autorsm(
            command,
            lambda pct, text: self.conversion_progress.emit(
                pct, f'{tag}  {text}'))
        self.conversion_progress.emit(0, '')
        if returncode:
            raise RuntimeError(output[-2000:])
        workflow.append_unique_line(processed_list, command_text)
        saved = [line[6:] for line in output.splitlines()
                 if line.startswith('Saved ')]
        note = next((l for l in output.splitlines()
                     if l.startswith('Total overloaded')), '')
        if note:
            status(note)
        return (f'Reconstruction saved: {saved[-1] if saved else tag}'
                + (f' -- {note}' if note else ''))

    def batch_reconstruct(self):
        """Run U/UB on every selected scan in one queued pass: each scan uses its
        own latest U_S record, with the grid/matrix/tag settings chosen once in
        the dialog applied to all. Refuses if any selected scan is not converted
        or has no U_S record yet. A scan that fails is reported and skipped."""
        datasets = self.selected_datasets()
        if not datasets:
            self.message('Select one or more scans for batch U/UB.')
            return
        not_converted = [ds.label for ds in datasets if not ds.resolved_output()]
        no_u = [ds.label for ds in datasets if not ds.metadata_records()]
        if not_converted or no_u:
            problems = []
            if not_converted:
                problems.append('not converted: ' + ', '.join(not_converted))
            if no_u:
                problems.append('no U_S record (Find U first): '
                                + ', '.join(no_u))
            QtWidgets.QMessageBox.warning(
                self, 'Batch U/UB not ready',
                'Every selected scan needs a reconstructed volume and a U_S '
                'record before a batch run:\n\n' + '\n'.join(problems))
            return
        dialog = DimensionsDialog(datasets[0].metadata_records(), self,
                                  batch=True)
        if _dialog_exec(dialog) != QtWidgets.QDialog.Accepted:
            return
        try:
            (_metadata, matrix_type, custom_grid, ranges, shape, tag,
             orientation) = dialog.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid selection', str(exc))
            return
        spec = (matrix_type, custom_grid, ranges, shape, tag, orientation)
        total = len(datasets)

        def work(status):
            done, failed = [], []
            for index, ds in enumerate(datasets, 1):
                version, path, metadata = ds.metadata_records()[-1]
                selected = dict(metadata, u_s_record=os.path.abspath(path),
                                u_s_version=version)
                status(f'[{index}/{total}] {ds.label} ...')
                try:
                    status(f'[{index}/{total}] '
                           f'{self._reconstruct_one(ds, selected, spec, status)}')
                    done.append(ds.scan_number)
                except Exception as exc:
                    failed.append(ds.scan_number)
                    status(f'Scan {ds.scan_number} failed: '
                           f'{str(exc).splitlines()[-1]}')
            summary = f'Batch U/UB: {len(done)}/{total} scan(s) reconstructed'
            if failed:
                summary += f'; failed: {failed}'
            return summary

        self._start_task(work, f'Batch U/UB for {total} scan(s) ...')

    def selected_datasets(self):
        """The datasets for the currently selected table rows, in order."""
        rows = sorted({index.row()
                       for index in self.table.selectionModel().selectedRows()})
        return [self.datasets[row] for row in rows if row < len(self.datasets)]

    def ctr_rods(self):
        """Reconstruct high-resolution HKL rods for the selected phi scans:
        average their U, define the (h, k) rods and L grid in a dialog, then
        run autoRSM_rods once over the merged group."""
        datasets = [ds for ds in self.selected_datasets()
                    if not ds.is_theta_only]
        if not datasets:
            QtWidgets.QMessageBox.information(
                self, 'Select scans',
                'Select one or more phi scan rows (click the Dataset column) '
                'for the rod reconstruction.')
            return
        missing = [ds for ds in datasets if not ds.resolved_output()]
        if missing:
            QtWidgets.QMessageBox.information(
                self, 'Convert first',
                'These selected scans are not converted yet: '
                + ', '.join(str(ds.scan_number) for ds in missing))
            return
        dialog = ctr.CTRDialog(datasets, list(self.lattices), self,
                               lattice_file=self.opts['lattice_file'],
                               default_normal=workflow.DEFAULT_NORMAL)
        if _dialog_exec(dialog) != QtWidgets.QDialog.Accepted:
            return
        spec = dialog.values()

        scan_list = sorted({scan for ds in datasets for scan in ds.scans})
        run_label = self.opts['run_label']
        recon_dir = os.path.join(self.opts['output_dir'], 'logs',
                                 'reconstructions', run_label)
        source = datasets[0].config_path
        config_name = (os.path.splitext(os.path.basename(source))[0]
                       + f"_{spec['output_tag']}_rods.txt")
        config_path = workflow.next_available_path(
            os.path.join(recon_dir, config_name))

        def work(status):
            workflow.write_rod_config(
                source, config_path, spec['metadata'], scan_list, spec['pairs'],
                spec['h_window'], spec['h_points'], spec['k_window'],
                spec['k_points'], spec['l_range'], spec['l_points'],
                spec['output_tag'], max_intensity=self.opts['max_intensity'],
                orientation=spec['orientation'])
            command = _python_command(
                self.opts['python'], self.opts['autorsm_rods'], config_path)
            command_text = shlex.join(command)
            status(f"Reconstructing {len(spec['pairs'])} rod(s) from scans "
                   f"{scan_list} ...")
            status('RUN: ' + command_text)
            returncode, output = run_autorsm(
                command,
                lambda pct, text: self.conversion_progress.emit(
                    pct, f"{spec['output_tag']}  {text}"))
            self.conversion_progress.emit(0, '')
            if returncode:
                raise RuntimeError(output[-2000:])
            saved = [line[6:] for line in output.splitlines()
                     if line.startswith('Saved ')]
            note = next((l for l in output.splitlines()
                         if l.startswith('Total overloaded')), '')
            if note:
                status(note)
            return (f"Rod reconstruction saved: "
                    f"{saved[-1] if saved else spec['output_tag']}"
                    + (f' -- {note}' if note else ''))

        self._start_task(
            work, f'Starting CTR rod reconstruction for scans {scan_list}')

    def toggle_watcher(self, checked):
        if checked:
            if self.busy:
                self.watch_button.setChecked(False)
                self.message('Wait for the current manual task to finish.')
                return
            self.opts['interval'] = self.interval.value()
            self.watch_button.setText('Stop watching')
            self.auto_index_button.setEnabled(False)
            self.find_button.setEnabled(False)
            self.watcher_thread = QtCore.QThread()
            self.watcher = WatcherWorker(self.opts)
            self.watcher.moveToThread(self.watcher_thread)
            self.watcher_thread.started.connect(self.watcher.run)
            self.watcher.datasets_updated.connect(self.set_datasets)
            self.watcher.status.connect(self.message)
            self.watcher.progress.connect(self.update_progress)
            self.watcher.finished.connect(self.watcher_thread.quit)
            self.stop_watcher.connect(self.watcher.stop)
            self.watcher_thread.start()
        else:
            self.message('Stopping after the current operation ...')
            if self.watcher is not None:
                self.watcher.stopped = True
            self.stop_watcher.emit()
            self.watch_button.setEnabled(False)
            self.watcher_thread.finished.connect(self._watcher_finished)

    def _watcher_finished(self):
        self.watcher_thread = None
        self.watcher = None
        self.watch_button.setEnabled(True)
        self.watch_button.setChecked(False)
        self.watch_button.setText('Start watching')
        self.auto_index_button.setEnabled(True)
        self.find_button.setEnabled(True)
        self.refresh()

    def closeEvent(self, event):
        if self.watcher is not None:
            self.watcher.stopped = True
            self.stop_watcher.emit()
            self.watcher_thread.quit()
            self.watcher_thread.wait(5000)
        self.pool.waitForDone(5000)
        event.accept()


def default_opts():
    run_label = re.sub(r'[^A-Za-z0-9_.-]+', '-', getpass.getuser())
    return {
        'base_dir': '',
        'spec_dir': '',
        'output_dir': '',
        'poni_file': '',
        'mask_file': '',
        'interval': 60,
        # Frames whose peak (unmasked) intensity exceeds this are dropped from
        # the reconstruction -- a detector overload would otherwise smear a
        # saturation halo across the map.
        'max_intensity': 1e5,
        'run_label': run_label,
        'python': sys.executable,
        'make_log_files': '-m epiq_map.make_log_files',
        # autoRSM ships as a package module so the monitor is self-contained
        # wherever the repo is deployed; override with --autorsm if it lives
        # elsewhere on the beamtime server. The python interpreter is separate.
        'autorsm': '-m epiq_map.hkl_convert.auto_rsm',
        # CTR multi-rod driver, bundled alongside autoRSM.
        'autorsm_rods': '-m epiq_map.hkl_convert.auto_rsm_rods',
        'lattice_file': str(files("epiq_map.substrates").joinpath(
            "substrate_lattice_constants.txt")),
    }


def _load_toml(path):
    """Parse a TOML file, using stdlib tomllib (3.11+) or the tomli backport."""
    try:
        import tomllib as toml          # Python 3.11+
    except ModuleNotFoundError:
        import tomli as toml             # pip install tomli (3.10 and earlier)
    with open(path, 'rb') as handle:
        return toml.load(handle)


def load_config(path=None):
    """Read monitor settings from a TOML file (defaults to epiq_monitor.toml
    in the current directory). Returns {} if no config is present and none was
    explicitly requested. Relative paths resolve from the config directory."""
    if path is None:
        path = os.path.abspath('epiq_monitor.toml')
        if not os.path.exists(path):
            return {}
    else:
        path = os.path.abspath(path)
    here = os.path.dirname(path)
    data = _load_toml(path)
    unknown = set(data) - set(default_opts())
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    # Resolve relative path-like values against the config directory. A bare
    # command name (no path separator, e.g. "python") is left alone so it is
    # found on PATH -- only things that look like relative paths are joined.
    for key in ('python', 'autorsm', 'autorsm_rods', 'make_log_files',
                'lattice_file', 'base_dir', 'spec_dir', 'output_dir',
                'poni_file', 'mask_file'):
        val = data.get(key)
        if (isinstance(val, str) and val and not os.path.isabs(val)
                and ('/' in val or '\\' in val)):
            data[key] = os.path.join(here, val)
    return data


def parse_args(argv=None):
    # Precedence: built-in defaults < config file < command-line flags.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None,
                     help='TOML config file (default: epiq_monitor.toml in '
                          'the current directory)')
    pre_args, _ = pre.parse_known_args(argv)

    opts = default_opts()
    opts.update(load_config(pre_args.config))

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    for key in ('base_dir', 'spec_dir', 'output_dir', 'poni_file', 'mask_file',
                'python', 'make_log_files', 'autorsm', 'autorsm_rods',
                'lattice_file'):
        parser.add_argument('--' + key.replace('_', '-'), default=opts[key])
    parser.add_argument('--interval', type=int, default=opts['interval'])
    parser.add_argument('--max-intensity', type=float,
                        default=opts['max_intensity'])
    parser.add_argument('--run-label', default=opts['run_label'])
    args = parser.parse_args(argv)
    config_path = args.config
    opts.update({k: v for k, v in vars(args).items() if k != 'config'})
    opts['config'] = config_path
    return opts


def main(argv=None):
    opts = parse_args(argv)
    app = QtWidgets.QApplication([sys.argv[0]])
    window = MonitorWindow(opts)
    window.show()
    return _application_exec(app)


if __name__ == '__main__':
    raise SystemExit(main())
