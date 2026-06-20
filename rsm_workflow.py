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


def build_index_metadata(result, substrate, lattice, directions, source_nxs,
                         method='manual', copied_from=None):
    """Build the persisted U/UB record from a successful index result."""
    U0 = np.asarray(result['U'], dtype=float)
    Bstar = np.asarray(result['Bstar'], dtype=float)
    D = validate_directions(lattice, directions)

    # One output coordinate unit follows the selected crystallographic
    # direction. q_measured = UB @ [H,K,L] in the selected output frame.
    UB = U0 @ Bstar @ D
    physical = U0 @ Bstar @ D
    view_U = physical / np.linalg.norm(physical, axis=0)
    view_U = rl.orthonormalize(view_U)

    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_nxs': os.path.abspath(source_nxs),
        'method': method,
        'copied_from': copied_from,
        'substrate': substrate,
        'lattice': list(map(float, lattice)),
        'directions': _json_value(directions),
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
                                matrix_type="UB"):
    """Create an indexed autoRSM config without modifying the source log."""
    remove = {'UB', 'Substrate Lattice Params', 'H Range', 'K Range',
              'L Range', 'Grid Shape', 'Output Tag', 'U_S Record'}
    kept = []
    with open(source_config) as fh:
        for line in fh:
            if ': ' in line and line.split(': ', 1)[0].strip() in remove:
                continue
            kept.append(line.rstrip('\n'))
    if matrix_type not in ('U', 'UB'):
        raise ValueError("matrix_type must be 'U' or 'UB'")
    transfer = np.asarray(metadata[matrix_type], dtype=float)
    effective_lengths = tuple(
        float(2 * np.pi / np.linalg.norm(transfer[:, axis]))
        for axis in range(3))
    values = [
        f"# Transfer Matrix Type: {matrix_type}",
        f"UB: {transfer.tolist()}",
    ]
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
