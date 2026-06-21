#!/usr/bin/env python3
"""PyQt6 beamtime monitor with substrate indexing and HKL reconstruction."""

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

from PyQt6 import QtCore, QtGui, QtWidgets

import rsm_workflow as workflow


def autorsm_command(opts, config_path):
    """Build the autoRSM invocation for a config file.

    The command is constructed directly from configured paths -- there is no
    intermediate command list. A single autoRSM handles both unindexed maps
    and indexed (U/UB) maps with optional custom ranges; which one runs is
    decided by the keys present in ``config_path``.
    """
    return [opts['python'], opts['autorsm'], config_path]


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
    def label(self):
        scans = ','.join(map(str, self.scans))
        suffix = ' [theta]' if self.is_theta_only else ''
        return (f"{self.cfg['Material']} / {self.cfg['Sample Name']} / "
                f"{self.cfg.get('Temperature', '?')} / scan {scans}{suffix}")

    def base_output(self):
        scan_text = '_'.join(map(str, self.scans))
        name = (f"{self.cfg['Material']}_{self.cfg['Sample Name']}_scans_"
                f"{scan_text}_out.nxs")
        return os.path.join(self.cfg['Output Directory'],
                            'transformed_objects', name)

    def resolved_output(self):
        path = self.base_output()
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
    status = QtCore.pyqtSignal(str)
    success = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()


class FunctionTask(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    @QtCore.pyqtSlot()
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
        self.directions = {}
        defaults = (initial or {}).get('directions', workflow.DEFAULT_DIRECTIONS)
        for key in ('x', 'y', 'z'):
            edit = QtWidgets.QLineEdit(' '.join(map(str, defaults[key])))
            edit.setPlaceholderText('1 0 0')
            self.directions[key] = edit
            form.addRow(f'{key} direction:', edit)
        self.save_ub = QtWidgets.QCheckBox(
            'Also save scaled UB_S matrix (inverse angstrom)')
        self.save_ub.setChecked((initial or {}).get('save_scaled_ub', True))
        form.addRow('', self.save_ub)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return self.substrate.currentText(), {
            key: workflow.parse_direction(edit.text())
            for key, edit in self.directions.items()
        }, self.save_ub.isChecked()


class DimensionsDialog(QtWidgets.QDialog):
    """Choose U/UB and optional grids for the existing compatible autoRSM."""

    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Run existing autoRSM with U or UB')
        layout = QtWidgets.QVBoxLayout(self)
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
        form.addRow('Saved orientation:', self.u_selector)

        self.matrix_type = QtWidgets.QComboBox()
        self.matrix_type.addItem(
            'UB - scaled reciprocal basis; ranges are in r.l.u.', 'UB')
        self.matrix_type.addItem(
            'U - orientation only; ranges are in inverse angstrom', 'U')
        form.addRow('Transfer matrix:', self.matrix_type)

        self.custom_grid = QtWidgets.QCheckBox(
            'Use custom H/K/L ranges and grid shape')
        form.addRow('', self.custom_grid)
        layout.addLayout(form)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel('Axis'), 0, 0)
        grid.addWidget(QtWidgets.QLabel('Minimum'), 0, 1)
        grid.addWidget(QtWidgets.QLabel('Maximum'), 0, 2)
        grid.addWidget(QtWidgets.QLabel('Points'), 0, 3)
        defaults = {'H': (-4, 4, 300), 'K': (-4, 4, 300), 'L': (0, 6, 300)}
        self.axes = {}
        for row, axis in enumerate(('H', 'K', 'L'), 1):
            lo = QtWidgets.QDoubleSpinBox()
            hi = QtWidgets.QDoubleSpinBox()
            for box in (lo, hi):
                box.setRange(-10000, 10000)
                box.setDecimals(5)
            count = QtWidgets.QSpinBox()
            count.setRange(2, 4000)
            lo.setValue(defaults[axis][0])
            hi.setValue(defaults[axis][1])
            count.setValue(defaults[axis][2])
            self.axes[axis] = (lo, hi, count)
            grid.addWidget(QtWidgets.QLabel(axis), row, 0)
            grid.addWidget(lo, row, 1)
            grid.addWidget(hi, row, 2)
            grid.addWidget(count, row, 3)
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
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
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
                self.custom_grid.isChecked(), ranges, tuple(shape), tag)


class WatcherWorker(QtCore.QObject):
    datasets_updated = QtCore.pyqtSignal(list)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.stopped = False

    @QtCore.pyqtSlot()
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
            if ds.is_theta_only:
                source = by_scan.get(ds.scan_number - 1)
                if source and source.metadata():
                    metadata = source.metadata()
                    metadata = dict(metadata, method='copied',
                                    copied_from=source.resolved_output(),
                                    source_nxs=ds.resolved_output())
                    workflow.save_index_metadata(ds.next_u_path(), metadata)
                    self.status.emit(f'Copied U into theta scan {ds.scan_number}')
                continue
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
                    result, name, lattice, workflow.DEFAULT_DIRECTIONS,
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

    @QtCore.pyqtSlot()
    def run(self):
        opts = self.opts
        while not self.stopped:
            self.status.emit('Scanning for new datasets ...')
            subprocess.run([
                opts['python'], opts['make_log_files'],
                '--base-dir', opts['base_dir'], '--spec-dir', opts['spec_dir'],
                '--output-dir', opts['output_dir'], '--poni-file',
                opts['poni_file'], '--mask-file', opts['mask_file']],
                capture_output=True, text=True, check=False)
            datasets = self._datasets()
            self.datasets_updated.emit(datasets)
            for ds in datasets:
                if self.stopped:
                    break
                if ds.resolved_output():
                    continue
                command = self._command_for(ds)
                if command:
                    self.status.emit(f'Processing {ds.label} ...')
                    proc = subprocess.run(command, capture_output=True, text=True)
                    if proc.returncode:
                        self.status.emit(
                            f'autoRSM failed for scan {ds.scan_number}: '
                            f'{proc.stderr[-300:]}')
            if not self.stopped:
                self._auto_index(datasets)
                self.datasets_updated.emit(self._datasets())
            for _ in range(opts['interval'] * 10):
                if self.stopped:
                    break
                QtCore.QThread.msleep(100)
        self.finished.emit()


class MonitorWindow(QtWidgets.QMainWindow):
    stop_watcher = QtCore.pyqtSignal()
    COLUMNS = ('Dataset', 'Scan done', 'Config', 'Output', 'Index / U',
               'Reconstruct')

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
        self.refresh()
        self.message(f"autoRSM: {opts['python']} {opts['autorsm']}")
        self.message(f"Scanning logs in: {os.path.join(opts['output_dir'], 'logs')}")

    def _build_ui(self):
        self.setWindowTitle('autoRSM monitor - wrapper3')
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
        refresh = QtWidgets.QPushButton('Refresh')
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        self.auto_index_button = QtWidgets.QPushButton('Auto-index missing')
        self.auto_index_button.clicked.connect(self.auto_index_missing)
        bar.addWidget(self.auto_index_button)
        bar.addStretch()
        self.count = QtWidgets.QLabel()
        bar.addWidget(self.count)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(self.COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setFixedHeight(160)
        self.log.setStyleSheet('font-family: monospace; font-size: 11px;')
        layout.addWidget(self.log)
        self.statusBar().showMessage('Idle')

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

    def auto_index_missing(self):
        datasets = self._load_datasets()

        def work(status):
            helper = WatcherWorker(self.opts)
            helper.status.connect(status)
            helper._auto_index(datasets)
            return 'Automatic indexing pass complete'

        self._start_task(work, 'Starting automatic substrate matching ...')

    def _check_item(self, yes):
        item = QtWidgets.QTableWidgetItem('yes' if yes else '-')
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
            self.table.setCellWidget(row, 4, self._index_widget(ds))
            reconstruct = QtWidgets.QPushButton('Run U / UB...')
            reconstruct.setEnabled(bool(ds.resolved_output()) and not self.busy
                                   and self.watcher is None)
            reconstruct.clicked.connect(
                lambda _checked=False, dataset=ds: self.reconstruct(dataset))
            self.table.setCellWidget(row, 5, reconstruct)
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
        button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setEnabled(bool(ds.resolved_output()) and not self.busy
                          and self.watcher is None)
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
        if self.busy or self.watcher is not None:
            self.message('Stop the watcher and wait for the current task first.')
            return
        self.busy = True
        self.auto_index_button.setEnabled(False)
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
        self.auto_index_button.setEnabled(True)
        self.refresh()

    def find_u(self, ds, initial=None):
        dialog = IndexDialog(list(self.lattices), self, initial=initial)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            substrate, directions, save_ub = dialog.values()
            workflow.validate_directions(self.lattices[substrate], directions)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid indexing setup', str(exc))
            return

        def work(status):
            status(f'Loading {os.path.basename(ds.resolved_output())} ...')
            data, H, K, L = workflow.rl.load_rsm(ds.resolved_output())
            try:
                result = workflow.index_with_substrate(
                    data, H, K, L, self.opts['lattice_file'], substrate,
                    directions=directions, verbose=False)
            finally:
                del data
            if result is None:
                raise RuntimeError(f'No consistent {substrate} indexing found')
            metadata = workflow.build_index_metadata(
                result, substrate, self.lattices[substrate], directions,
                ds.resolved_output(), method='manual')
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
        workflow.save_index_metadata(saved_path, metadata)
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
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self.find_u(ds)
            return
        dialog = DimensionsDialog(records, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            metadata, matrix_type, custom_grid, ranges, shape, tag = dialog.values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid selection', str(exc))
            return
        run_label = self.opts['run_label']
        recon_dir = os.path.join(self.opts['output_dir'], 'logs',
                                 'reconstructions', run_label)
        config_name = (os.path.splitext(os.path.basename(ds.config_path))[0] +
                       f'_{tag}.txt')
        config_path = workflow.next_available_path(
            os.path.join(recon_dir, config_name))
        workflow.write_reconstruction_config(
            ds.config_path, config_path, metadata, ranges, shape, tag,
            custom_grid=custom_grid, matrix_type=matrix_type)

        command = autorsm_command(self.opts, config_path)
        command_text = shlex.join(command)
        log_dir = os.path.join(self.opts['output_dir'], 'logs')
        command_list = os.path.join(
            log_dir, f'command_list_indexed_{run_label}.txt')
        processed_list = os.path.join(
            log_dir, f'processed_commands_indexed_{run_label}.txt')
        workflow.append_unique_line(command_list, command_text)

        def work(status):
            status(f'Reconstructing scan {ds.scan_number} as {tag} with '
                   f'the existing autoRSM ({matrix_type}) ...')
            status('RUN: ' + command_text)
            proc = subprocess.run(
                command,
                capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
            workflow.append_unique_line(processed_list, command_text)
            saved = [line[6:] for line in proc.stdout.splitlines()
                     if line.startswith('Saved ')]
            return f'Reconstruction saved: {saved[-1] if saved else tag}'

        self._start_task(work, f'Starting indexed reconstruction for {ds.label}')

    def toggle_watcher(self, checked):
        if checked:
            if self.busy:
                self.watch_button.setChecked(False)
                self.message('Wait for the current manual task to finish.')
                return
            self.opts['interval'] = self.interval.value()
            self.watch_button.setText('Stop watching')
            self.auto_index_button.setEnabled(False)
            self.watcher_thread = QtCore.QThread()
            self.watcher = WatcherWorker(self.opts)
            self.watcher.moveToThread(self.watcher_thread)
            self.watcher_thread.started.connect(self.watcher.run)
            self.watcher.datasets_updated.connect(self.set_datasets)
            self.watcher.status.connect(self.message)
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
    here = os.path.dirname(os.path.abspath(__file__))
    run_label = re.sub(r'[^A-Za-z0-9_.-]+', '-', getpass.getuser())
    return {
        'base_dir': '/nfs/chess/id4b/2026-2/sarker-4910-a/raw6M/',
        'spec_dir': '/nfs/chess/id4b/2026-2/sarker-4910-a/',
        'output_dir': '/nfs/chess/id4baux/2026-2/sarker-4910-a/processed/output/',
        'poni_file': '/nfs/chess/id4baux/2026-2/sarker-4910-a/calibrations/ceO2_15keV.poni',
        'mask_file': '/nfs/chess/id4baux/2026-2/sarker-4910-a/calibrations/mask.edf',
        'interval': 60,
        'run_label': run_label,
        'python': '/nfs/chess/user/ss3428/anaconda3_jpcr/bin/python',
        'make_log_files': os.path.join(here, 'make_log_files.py'),
        # autoRSM ships bundled in HKL_Convert/ so the monitor is self-contained
        # wherever the repo is deployed; override with --autorsm if it lives
        # elsewhere on the beamtime server. The python interpreter is separate.
        'autorsm': os.path.join(here, 'HKL_Convert', 'autoRSM.py'),
        'lattice_file': os.path.join(here, 'substrate_lattice_constants.txt'),
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
    next to this script). Returns {} if no config is present and none was
    explicitly requested. Relative path values resolve from the repo dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(here, 'epiq_monitor.toml')
        if not os.path.exists(path):
            return {}
    data = _load_toml(path)
    unknown = set(data) - set(default_opts())
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    # Resolve relative path-like values against the repo directory. A bare
    # command name (no path separator, e.g. "python") is left alone so it is
    # found on PATH -- only things that look like relative paths are joined.
    for key in ('python', 'autorsm', 'make_log_files', 'lattice_file',
                'base_dir', 'spec_dir', 'output_dir', 'poni_file', 'mask_file'):
        val = data.get(key)
        if (isinstance(val, str) and val and not os.path.isabs(val)
                and os.sep in val):
            data[key] = os.path.join(here, val)
    return data


def parse_args(argv=None):
    # Precedence: built-in defaults < config file < command-line flags.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None,
                     help='TOML config file (default: epiq_monitor.toml beside '
                          'this script)')
    pre_args, _ = pre.parse_known_args(argv)

    opts = default_opts()
    opts.update(load_config(pre_args.config))

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    for key in ('base_dir', 'spec_dir', 'output_dir', 'poni_file', 'mask_file',
                'python', 'make_log_files', 'autorsm',
                'lattice_file'):
        parser.add_argument('--' + key.replace('_', '-'), default=opts[key])
    parser.add_argument('--interval', type=int, default=opts['interval'])
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
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
