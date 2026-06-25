# rsm_monitor: autoRSM monitor, substrate indexing, and reconstruction

`rsm_monitor` (formerly the `wrapper3` tool) is a self-contained beamtime
wrapper bundled in the EpiQ-Map suite. It discovers scans, runs
lab-frame autoRSM reconstruction, finds substrate orientation matrices, and
can rerun a scan directly into a user-selected indexed HKL volume.

## Run

After installing EpiQ-Map:

```bash
export OMP_NUM_THREADS=12
rsm_monitor --config epiq_monitor.toml
```

## Configuration

Per-beamtime settings live in a user-created `epiq_monitor.toml`. Copy
`examples/epiq_monitor.toml` and edit
it once per beamtime instead of changing the code -- paths, the Python
interpreter that runs autoRSM, the polling interval, etc.:

```toml
base_dir   = "/nfs/chess/id4b/2024-2/gregory-3864-b/raw6M/"
output_dir = "/nfs/chess/id4baux/2024-2/gregory-3864-b/processed/output/"
python     = "/nfs/chess/user/YOURUSER/anaconda3/bin/python"
# The packaged autoRSM module is used unless --autorsm overrides it.
interval   = 60
```

Settings resolve in this order (later wins):

```
built-in defaults  <  epiq_monitor.toml  <  command-line flags
```

Point at a different config with `--config /path/to/file.toml`. Override any
single value at launch without touching the file:

```bash
rsm_monitor \
  --base-dir /path/to/raw6M \
  --spec-dir /path/to/spec/files \
  --output-dir /path/to/output \
  --poni-file /path/to/calibration.poni \
  --mask-file /path/to/mask.edf
```

TOML parsing uses the stdlib `tomllib` on Python 3.11+, or the `tomli`
backport on 3.10 and earlier (`conda install -n viz tomli`).

## How reconstruction is invoked

The monitor builds each reconstruction command directly as
`[python, autorsm, config_file]` -- there is no intermediate command list.
`make_log_files` writes one self-contained config (`.txt`) per scan into
`<output_dir>/logs`; the monitor runs autoRSM on each. The configs double as
manual rerun recipes: `python autoRSM.py <output_dir>/logs/<scan>.txt`.

A single `autoRSM.py` handles every case: an unindexed lab-frame Q map, or an
indexed (U/UB) map, with optional custom H/K/L ranges -- decided by the keys
present in the config (UB / Substrate Lattice Params / H,K,L Range + Grid
Shape). Identity UB and detector-derived ranges are the defaults when those
keys are absent. Point at a different interpreter or autoRSM with `--python`
and `--autorsm` (or set them in `epiq_monitor.toml`).

The defaults are in `rsm_monitor.py:default_opts`.

## Monitor columns

- **Dataset**: material, sample, temperature, and scan number.
- **Scan done / Config / Output**: acquisition and lab-frame processing state.
- **Index / U**: substrate, inlier count, and an Actions menu.
- **Reconstruct**: choose indexed H/K/L ranges and point counts, then run.

The **Index / U** menu provides:

- **Find new U_S**: choose a substrate and three crystallographic output
  directions. Directions accept forms such as `1 0 0` or `[1, -1, 0]` and
  must form a right-handed, physically orthogonal set. An option also saves
  `_UB_S.txt`, the full scaled reciprocal matrix in inverse angstrom.
- **U same as scan**: copy the complete orientation/indexing record.
- **Substrate same as scan**: reuse another scan's substrate and directions,
  but fit a new orientation from this scan's own peaks.

All long work runs outside the GUI thread.
Use **Auto-index missing** to run one pass over existing outputs without
starting the continuous watcher.

## Automatic substrate matching

After producing a lab-frame `.nxs`, the watcher finds peaks once and tests all
entries in `substrate_lattice_constants.txt`. Candidates are ranked by inlier
count and RMS residual. A result is saved only if it has:

- at least 25% of detected peaks (and at least six) indexed;
- RMS no greater than `0.03 A^-1`; and
- a clear advantage over the second-ranked lattice.

Weak or tied results show as **ambiguous** and are not silently accepted. The
candidate summary is saved as `<stem>_U_attempt.json`; a manual fit replaces
that status. Theta-only scans inherit U from scan `N-1` when available.

## U files

Substrate-derived matrices are deliberately separate from legacy `_U.txt`
files. The first result is:

```text
<lab-frame-output-stem>_U_S.txt
```

The authoritative record is:

```text
<lab-frame-output-stem>_U_S.json
```

Nothing is overwritten and file permissions are never changed. Repeated fits
or copies create `_U_S_02`, `_U_S_03`, and so on. **Dimensions / Run always
uses the newest complete U_S record**, never a legacy `_U.txt`.

It contains the substrate and nominal cell, selected x/y/z directions,
viewing `U`, fitted orientation, reciprocal basis, full reconstruction `UB`,
inliers/RMS, refined cell with uncertainties, source scan, and provenance.

The reconstruction matrix follows:

```text
q_measured = UB @ [H, K, L]
UB = orientation_U @ Bstar @ [x_direction, y_direction, z_direction]
```

This is important for non-cubic substrates: a rotation matrix alone does not
contain reciprocal-lattice scale.

## Indexed reconstruction

**Dimensions / Run** first lists every complete U_S version for the scan with
its substrate, inliers, RMS, and x/y/z directions. The selected U_S is then
used to create an immutable derived config under:

```text
<output>/logs/reconstructions/
```

The original scan config is never modified. Output is written under
`indexed_objects` with a concise name such as:

```text
La3Ni2O7_standard_scans_57_out_U_S_LaAlO3_r01.nxs
```

Exact ranges, shape, UB, substrate cell, source config, output tag, and UTC
creation time are stored inside the NeXus file as
`reconstruction_metadata`. If a name exists, autoRSM appends `_more`.
The metadata also records the exact selected `_U_S*.json` path.

The dialog offers two engines:

- **Original server autoRSM**: unchanged automatic bounds and `1000^3` grid.
- **U_S autoRSM**: custom H/K/L bounds and dimensions.

Every click writes a persistent derived config under
`logs/reconstructions/<username>/` and appends the exact command to
`logs/command_list_U_S_<username>.txt`. Successful commands are appended to
`logs/processed_commands_U_S_<username>.txt`. Shared command lists and ledgers
are never edited. Override the identity suffix with `--run-label NAME`.

## Lattice database

Edit `substrate_lattice_constants.txt`, one material per line:

```text
Formula  a  b  c  alpha  beta  gamma
```

Lengths are in angstrom and angles in degrees.

## Dependencies

- Python 3.10+
- NumPy, SciPy, scikit-learn
- QtPy plus one of PyQt5, PyQt6, PySide2 (Qt 5), or PySide6
- nexusformat
- pyFAI and fabio
- silx

The monitor itself does not require napari. A minimal Qt 6 installation is:

```bash
python -m pip install "EpiQ-Map[qt6]"
```

Equivalent binding extras are `qt5`, `pyside2`, and `pyside6`. PySide2 is
the Qt 5 package name; there is no package named PySide5.

The native `hklBen` kernel is included in platform wheels. For a source
checkout, build it with `python -m build` or `python -m pip install -e .`.

## Tests

```bash
cd wrapper3
python -m pytest tests/integration/test_monitor_workflow.py
QT_QPA_PLATFORM=offscreen rsm_monitor --help
```
