"""Scientific and persistence helpers for the wrapper3 RSM monitor."""

import json
import os
import re
from datetime import datetime, timezone

import numpy as np

import Visualize_RSM_Lib as rl


DEFAULT_DIRECTIONS = {
    'x': [1, 0, 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1],
}

# Default substrate surface normal for the rsm_viewer-style U finder: a
# (001)-oriented cell. The normal pins the indexed frame reproducibly.
DEFAULT_NORMAL = [0, 0, 1]


def load_lattice_entries(path):
    """Return ordered ``{name: (a,b,c,alpha,beta,gamma)}`` entries."""
    names = []
    with open(path) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            match = re.match(r'([A-Za-z0-9_.\-()]+)', line)
            if match and match.group(1) not in names:
                names.append(match.group(1))
    return {name: rl.load_lattice(path, name) for name in names}


def parse_direction(text):
    """Parse ``1 0 0``, ``[1,0,0]``, or ``1,0,0`` as a 3-vector."""
    cleaned = text.strip().strip('[]()').replace(',', ' ')
    values = [float(value) for value in cleaned.split()]
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError("direction must contain three finite numbers")
    if np.linalg.norm(values) == 0:
        raise ValueError("direction cannot be zero")
    return values


def orientation_matrix(d1, d2, d3):
    """Output directions Q1, Q2, Q3 as the columns of a rotation matrix.

    Each ``di`` is a crystal direction ``[u v w]`` in the U-aligned frame; a
    unit step along output axis Qi then runs along ``di``. The three must be
    mutually orthogonal. Mirrors ``rsm_viewer.orientation_matrix`` so oriented
    reconstructions match the viewer's oriented Q axes exactly.
    """
    directions = np.asarray((d1, d2, d3), dtype=float)
    if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
        raise ValueError("orientation directions must be finite 3-vectors")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("orientation directions cannot be zero")
    directions = directions / lengths[:, None]
    if not np.allclose(directions @ directions.T, np.eye(3), atol=1e-6):
        raise ValueError("Q1, Q2, Q3 directions must be mutually orthogonal")
    return directions.T


def validate_directions(lattice, directions):
    Bstar = rl.reciprocal_matrix(*lattice, with_2pi=True)
    x, y, z = (directions[key] for key in ('x', 'y', 'z'))
    rl.target_orientation_matrix(Bstar, x, y, z)
    D = np.column_stack((x, y, z))
    if np.linalg.det(D) <= 0:
        raise ValueError("x, y, z directions must be right-handed")
    return D


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()
                if key not in ('residuals', 'covariance',
                               'reciprocal_covariance')}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def build_index_metadata(result, substrate, lattice, source_nxs,
                         method='manual', copied_from=None, normal=None):
    """Build the persisted U/UB record from a successful index result.

    The orientation is already pinned inside ``result['U']`` -- the
    rsm_viewer finder (``compute_U_from_substrate``) bakes the chosen surface
    normal into U, so q_measured = U @ B* @ hkl directly. Hence:

      * ``UB = U @ B*`` maps hkl (reciprocal lattice units) to measured q.
      * the saved ``U`` is the orientation-only frame (unit crystal-axis
        directions) for autoRSM runs whose ranges are in inverse angstrom.
    """
    U0 = np.asarray(result['U'], dtype=float)
    Bstar = np.asarray(result['Bstar'], dtype=float)
    UB = U0 @ Bstar
    view_U = UB / np.linalg.norm(UB, axis=0)
    view_U = rl.orthonormalize(view_U)

    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_nxs': os.path.abspath(source_nxs),
        'method': method,
        'copied_from': copied_from,
        'substrate': substrate,
        'lattice': list(map(float, lattice)),
        'normal': None if normal is None else [float(v) for v in normal],
        'U': view_U.tolist(),
        'orientation_U': U0.tolist(),
        'Bstar': Bstar.tolist(),
        'UB': UB.tolist(),
        'n_peaks': int(len(result['peaks'])),
        'n_inliers': int(result['n_inliers']),
        'inlier_fraction': float(result['n_inliers'] / len(result['peaks'])),
        'rms_A^-1': float(result['rms']),
        'refined_lattice': _json_value(result.get('refined_lattice')),
    }


def save_index_metadata(u_path, metadata):
    """Save U_S text/JSON records, refusing to overwrite either file."""
    json_path = u_path[:-4] + '.json' if u_path.endswith('.txt') \
        else u_path + '.json'
    marker = u_path.rfind('_U_S')
    ub_path = (u_path[:marker] + '_UB_S' + u_path[marker + 4:]
               if marker >= 0 else u_path[:-4] + '_UB.txt')
    save_ub = bool(metadata.get('save_scaled_ub', False))
    targets = [u_path, json_path] + ([ub_path] if save_ub else [])
    if any(os.path.exists(path) for path in targets):
        raise FileExistsError(f'U_S record already exists: {u_path}')
    written = []
    try:
        with open(u_path, 'x') as fh:
            np.savetxt(fh, np.asarray(metadata['U'], dtype=float))
        written.append(u_path)
        if save_ub:
            with open(ub_path, 'x') as fh:
                np.savetxt(fh, np.asarray(metadata['UB'], dtype=float))
            written.append(ub_path)
        with open(json_path, 'x') as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
            fh.write('\n')
        written.append(json_path)
    except Exception:
        for path in written:
            if os.path.exists(path):
                os.remove(path)
        raise
    return json_path


def next_available_path(path):
    """Return path or a `_02`, `_03`, ... variant that does not exist."""
    root, extension = os.path.splitext(path)
    candidate = path
    version = 2
    while os.path.exists(candidate):
        candidate = f'{root}_{version:02d}{extension}'
        version += 1
    return candidate


def append_unique_line(path, line):
    """Append one audit-ledger line unless it is already present."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    existing = set()
    if os.path.exists(path):
        with open(path) as fh:
            existing = {item.rstrip('\n') for item in fh}
    if line not in existing:
        with open(path, 'a') as fh:
            fh.write(line + '\n')


def sync_config_paths(config_path, poni_file=None, mask_file=None,
                      output_dir=None, spec_dir=None, max_intensity=None):
    """Update a per-scan log's global beamtime settings in place from the
    current config, then return the list of keys that changed.

    make_log_files bakes PONI/Mask/Output/Specfile/Max Intensity into each log
    when it is first written and never rewrites an existing log, so a corrected
    epiq_monitor.toml otherwise has no effect until the logs are deleted. This
    reconciles those global fields (the per-scan Image/Temperature directories
    are left alone -- they are tied to base_dir, which renames the log) so a
    toml change takes effect on the next conversion. Atomic replace; no-op when
    nothing differs.
    """
    with open(config_path) as fh:
        lines = fh.readlines()
    material = next((line.split(': ', 1)[1].strip() for line in lines
                     if line.startswith('Material: ')), None)
    path_keys = {'PONI File', 'Mask File', 'Output Directory', 'Specfile'}
    updates = {}
    if poni_file is not None:
        updates['PONI File'] = poni_file
    if mask_file is not None:
        updates['Mask File'] = mask_file
    if output_dir is not None:
        updates['Output Directory'] = output_dir
    if spec_dir is not None and material is not None:
        updates['Specfile'] = os.path.join(spec_dir, material)
    if max_intensity is not None:
        updates['Max Intensity'] = repr(float(max_intensity))

    changed, new_lines, seen = [], [], set()
    for line in lines:
        key = line.split(': ', 1)[0] if ': ' in line else None
        if key in updates:
            seen.add(key)
            value = line.split(': ', 1)[1].strip()
            new = updates[key]
            differs = (os.path.normpath(value) != os.path.normpath(new)
                       if key in path_keys else value != new)
            if differs:
                new_lines.append(f'{key}: {new}\n')
                changed.append(key)
                continue
        new_lines.append(line)
    # Append any configured key the log predates (e.g. Max Intensity in logs
    # written before that key existed) -- replacing alone would never add it.
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'
    for key in updates:
        if key not in seen:
            new_lines.append(f'{key}: {updates[key]}\n')
            changed.append(key)
    if changed:
        tmp = config_path + '.tmp'
        with open(tmp, 'w') as fh:
            fh.writelines(new_lines)
        os.replace(tmp, config_path)
    return changed


def load_index_metadata(u_path):
    json_path = u_path[:-4] + '.json' if u_path.endswith('.txt') \
        else u_path + '.json'
    with open(json_path) as fh:
        return json.load(fh)


def index_with_substrate(data, H, K, L, lattice_file, substrate,
                         directions=None, peaks=None, verbose=False):
    lattice = rl.load_lattice(lattice_file, substrate)
    if peaks is None:
        threshold = np.nanmax(data) / 25
        peaks, _ = rl.find_peaks_in_RSM(
            data, H, K, L, dQ=0.3, threshold=threshold, peak_width=0.05)
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 2:
        return None
    result = rl.index_against_lattice(
        peaks, lattice, q_with_2pi=True, hkl_max=6, mod_tol=0.04,
        ang_tol=2.0, inlier_tol=0.05, ortho=True, verbose=verbose)
    if result is not None:
        result['peaks'] = peaks
    return result


def auto_match_substrate(data, H, K, L, lattice_file, verbose=False):
    """Try every lattice and return a confidence-gated ranked result."""
    threshold = np.nanmax(data) / 25
    peaks, _ = rl.find_peaks_in_RSM(
        data, H, K, L, dQ=0.3, threshold=threshold, peak_width=0.05)
    peaks = np.asarray(peaks, dtype=float)
    ranked = []
    for name, lattice in load_lattice_entries(lattice_file).items():
        try:
            validate_directions(lattice, DEFAULT_DIRECTIONS)
        except ValueError:
            # The default output axes are not a valid orthogonal view for
            # this cell; it can still be indexed manually with chosen axes.
            continue
        result = index_with_substrate(
            data, H, K, L, lattice_file, name, peaks=peaks, verbose=verbose)
        if result is not None:
            ranked.append((name, lattice, result))
    ranked.sort(key=lambda item: (-item[2]['n_inliers'], item[2]['rms']))
    if not ranked:
        return {'accepted': False, 'reason': 'no lattice matched',
                'ranked': [], 'peaks': peaks}

    best = ranked[0]
    n_peaks = max(len(peaks), 1)
    enough = (best[2]['n_inliers'] >= max(6, int(np.ceil(0.25 * n_peaks)))
              and best[2]['rms'] <= 0.03)
    clear = True
    if len(ranked) > 1:
        second = ranked[1]
        clear = (best[2]['n_inliers'] >= second[2]['n_inliers'] + 3 or
                 (best[2]['n_inliers'] > second[2]['n_inliers'] and
                  best[2]['rms'] < 0.75 * second[2]['rms']))
    reason = 'accepted' if enough and clear else (
        'weak match' if not enough else 'ambiguous best and second match')
    return {'accepted': enough and clear, 'reason': reason, 'best': best,
            'ranked': ranked, 'peaks': peaks}


def write_reconstruction_config(source_config, destination, metadata, ranges,
                                shape, output_tag, custom_grid=True,
                                matrix_type="UB", orientation=None,
                                max_intensity=None):
    """Create an indexed autoRSM config without modifying the source log.

    ``orientation``, if given, is a ``(Q1, Q2, Q3)`` triple of crystal
    directions; the transfer matrix is re-oriented so each output axis runs
    along its direction (``transfer @ orientation_matrix``), exactly like
    rsm_viewer's oriented Q axes.
    """
    # Max Intensity is stripped and re-written from the current threshold so a
    # reconstruction always uses the overload cutoff configured now, not a
    # stale value (or none) baked into the source log.
    remove = {'UB', 'Substrate Lattice Params', 'H Range', 'K Range',
              'L Range', 'Grid Shape', 'Output Tag', 'U_S Record',
              'Max Intensity'}
    kept = []
    with open(source_config) as fh:
        for line in fh:
            if ': ' in line and line.split(': ', 1)[0].strip() in remove:
                continue
            kept.append(line.rstrip('\n'))
    if matrix_type not in ('U', 'UB'):
        raise ValueError("matrix_type must be 'U' or 'UB'")
    transfer = np.asarray(metadata[matrix_type], dtype=float)
    if orientation is not None:
        transfer = transfer @ orientation_matrix(*orientation)
    effective_lengths = tuple(
        float(2 * np.pi / np.linalg.norm(transfer[:, axis]))
        for axis in range(3))
    # autoRSM names the output Material_Sample_scans_N_<Output Tag>.nxs, so the
    # tag carries the audit label plus the H/K/L range -- the audit tag's UB/U
    # prefix marks r.l.u. vs inverse angstrom -- keeping the many indexed maps
    # of a single scan distinguishable in indexed_objects/.
    if custom_grid:
        range_desc = '_'.join(
            f'{axis}{ranges[axis][0]:g}to{ranges[axis][1]:g}'
            for axis in ('H', 'K', 'L'))
        out_tag = f'{output_tag}_{range_desc}'
    else:
        out_tag = f'{output_tag}_auto'
    values = [
        f"# Transfer Matrix Type: {matrix_type}",
        f"Output Tag: {out_tag}",
        f"UB: {transfer.tolist()}",
    ]
    if max_intensity is not None:
        values.append(f"Max Intensity: {repr(float(max_intensity))}")
    if custom_grid:
        values.extend([
            f"Substrate Lattice Params: {tuple(metadata['lattice'])}",
            f"H Range: {tuple(ranges['H'])}",
            f"K Range: {tuple(ranges['K'])}",
            f"L Range: {tuple(ranges['L'])}",
            f"Grid Shape: {tuple(int(n) for n in shape)}",
        ])
    else:
        # The production server parser accepts exactly three lengths and
        # chooses its own 1000^3 bounds. These effective lengths correspond
        # to the selected orthogonal reciprocal output directions.
        values.append(f"Substrate Lattice Params: {effective_lengths}")
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, 'x') as fh:
        fh.write('\n'.join(kept + values) + '\n')
    return destination


# ----------------------------------------------------------------------
# CTR rods: average several scans' U, write a multi-rod reconstruction config
# ----------------------------------------------------------------------

def average_U(records):
    """Average several substrate-index records into one orientation.

    Each record is a saved U_S metadata dict (``build_index_metadata`` output)
    carrying ``orientation_U`` (a proper rotation, q ~= U @ B* @ hkl),
    ``Bstar``, ``lattice``, and ``substrate``. The orientation matrices are
    averaged elementwise and snapped back to the nearest exact rotation via
    polar decomposition (``rl.orthonormalize``) -- the standard way to mean
    rotations that are already close. All records must share a substrate, so a
    common B* is well defined; ``UB = U_avg @ B*`` maps hkl (r.l.u.) to
    measured q for the rod reconstruction. Returns an averaged metadata dict in
    the ``build_index_metadata`` shape that ``write_rod_config`` consumes.
    """
    records = list(records)
    if not records:
        raise ValueError("no U_S records to average")
    substrates = {r.get('substrate') for r in records}
    if len(substrates) > 1:
        raise ValueError("cannot average U across different substrates: "
                         f"{sorted(str(s) for s in substrates)}")
    Us = [np.asarray(r['orientation_U'], dtype=float) for r in records]
    if any(U.shape != (3, 3) or not np.all(np.isfinite(U)) for U in Us):
        raise ValueError("each record needs a finite 3x3 orientation_U")
    U_avg = rl.orthonormalize(np.mean(Us, axis=0))
    Bstar = np.asarray(records[0]['Bstar'], dtype=float)
    UB = U_avg @ Bstar
    base = records[0]
    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'averaged',
        'substrate': base.get('substrate'),
        'lattice': list(map(float, base['lattice'])),
        'normal': base.get('normal'),
        'orientation_U': U_avg.tolist(),
        'Bstar': Bstar.tolist(),
        'UB': UB.tolist(),
        'n_averaged': len(records),
        'sources': [r.get('source_nxs') for r in records],
    }


def write_rod_config(source_config, destination, metadata, scan_list,
                     rod_hk_pairs, h_window, h_points, k_window, k_points,
                     l_range, l_points, output_tag, max_intensity=None,
                     orientation=None):
    """Create an ``autoRSM_rods`` config for a multi-rod (CTR) reconstruction.

    Mirrors ``write_reconstruction_config`` but for the rod driver: the source
    log's per-scan grid/scan keys are stripped and replaced so the whole group
    of phi scans (``scan_list``) is merged in one pass. ``metadata['UB']``
    (= U_avg @ B*) is the transfer matrix, so the rod centers in ``rod_hk_pairs``
    are in r.l.u. (h, k). Each rod is a tight grid
    ``h0 + linspace(*h_window, h_points)`` x ``k0 + linspace(*k_window, k_points)``
    x ``linspace(*l_range, l_points)``.

    ``orientation``, if given, is a ``(Q1, Q2, Q3)`` triple of crystal
    directions; the transfer matrix is re-oriented (``UB @ orientation_matrix``)
    so the rod (h, k) and L axes run along those directions -- the rod
    counterpart of rsm_viewer's oriented Q axes.
    """
    remove = {'UB', 'Substrate Lattice Params', 'H Range', 'K Range',
              'L Range', 'Grid Shape', 'Output Tag', 'U_S Record',
              'Max Intensity', 'Scan List', 'Scan Number',
              'Theta Scan List', 'Theta Scan Number', 'L Points',
              'Rod HK List', 'Rod H Window', 'Rod H Points',
              'Rod K Window', 'Rod K Points'}
    kept = []
    with open(source_config) as fh:
        for line in fh:
            if ': ' in line and line.split(': ', 1)[0].strip() in remove:
                continue
            kept.append(line.rstrip('\n'))

    UB = np.asarray(metadata['UB'], dtype=float)
    if orientation is not None:
        UB = UB @ orientation_matrix(*orientation)
    pairs = [(int(h), int(k)) for h, k in rod_hk_pairs]
    if not pairs:
        raise ValueError("no (h, k) rods selected")
    h_lo, h_hi = (float(v) for v in h_window)
    k_lo, k_hi = (float(v) for v in k_window)
    l_lo, l_hi = (float(v) for v in l_range)
    if not (h_lo < h_hi and k_lo < k_hi and l_lo < l_hi):
        raise ValueError("windows and L range must be (minimum, maximum)")
    if min(int(h_points), int(k_points), int(l_points)) < 2:
        raise ValueError("rod point counts must be >= 2")
    scans = [int(s) for s in scan_list]
    if not scans:
        raise ValueError("no scans to reconstruct")

    values = [
        "# Transfer Matrix Type: UB",
        f"Output Tag: {output_tag}",
        f"Scan List: {scans}",
        "Theta Scan List: []",
        f"UB: {UB.tolist()}",
        f"Substrate Lattice Params: {tuple(metadata['lattice'])}",
        f"Rod HK List: {pairs}",
        f"Rod H Window: {(h_lo, h_hi)}",
        f"Rod H Points: {int(h_points)}",
        f"Rod K Window: {(k_lo, k_hi)}",
        f"Rod K Points: {int(k_points)}",
        f"L Range: {(l_lo, l_hi)}",
        f"L Points: {int(l_points)}",
    ]
    if max_intensity is not None:
        values.append(f"Max Intensity: {repr(float(max_intensity))}")
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, 'x') as fh:
        fh.write('\n'.join(kept + values) + '\n')
    return destination


def hkl_pairs(h_range, k_range):
    """All integer (h, k) pairs in the inclusive H and K ranges.

    ``h_range``/``k_range`` are (min, max) integer bounds. Returns the pairs in
    row-major order ((h0,k0), (h0,k1), ...), the order the CTR dialog populates
    its editable rod list with before the user trims it.
    """
    h_lo, h_hi = (int(round(v)) for v in h_range)
    k_lo, k_hi = (int(round(v)) for v in k_range)
    if h_lo > h_hi or k_lo > k_hi:
        raise ValueError("range minimum must not exceed maximum")
    return [(h, k) for h in range(h_lo, h_hi + 1)
            for k in range(k_lo, k_hi + 1)]
