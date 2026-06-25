#!/usr/bin/env python
"""
autoRSM_rods.py -- High-resolution HKL-rod (CTR) reconstruction.

Reconstruct many narrow reciprocal-space rods in a single pass over the
detector frames. Each rod is a tight box in (H, K) around an integer (h0, k0)
peak that spans the full L range at high resolution -- the small,
high-resolution counterpart of the big coarse cube that autoRSM.py builds.

Usage:
    python autoRSM_rods.py config.txt

The config is the same "Key: value" format as autoRSM.py (and shares its
detector/scan keys), plus the rod keys written by
``rsm_workflow.write_rod_config``:

    UB: [[...], [...], [...]]          # averaged U @ B*  (hkl -> measured q)
    Substrate Lattice Params: (a, b, c)
    Rod HK List: [(-1, -1), (-1, 0), ...]   # rod centers in r.l.u.
    Rod H Window: (-0.1, 0.1)          # per-rod H half-window (about h0)
    Rod H Points: 100
    Rod K Window: (-0.1, 0.1)
    Rod K Points: 100
    L Range: (0.0, 6.0)                # shared across rods
    L Points: 2000

Why one pass: the only expensive step is reading/decompressing the CBF frames.
Each frame is read once (prefetched), then histogrammed into every rod with a
separate, tight ``HKLHIST`` call. The C kernel bins by binary search and drops
pixels outside each rod's bounds, so a rod is self-clipping -- no contamination
between rods and no wasted off-rod voxels.

Output: a single NeXus file under ``{Output Directory}/rod_objects/`` whose
``entry`` holds one ``NXdata`` group per rod (``rod_<h0>_<k0>`` with its own
counts/H/K/L/norm), plus the averaged UB, substrate lattice, and scan list.
"""

import argparse
import ast
import os

import numpy as np
import fabio
import pyFAI
import tqdm
from nexusformat.nexus import NXdata, NXentry, NXfield, NXroot, nxsetmemory
from spec2nexus.spec import SpecDataFile

from . import hkl_ben
from . import auto_rsm

nxsetmemory(8000)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def parse_config(path):
    """Parse the rod "Key: value" config. Reuses autoRSM's detector/scan keys
    and adds the rod-grid keys. Unknown keys raise so typos do not vanish."""
    converters = {
        'PONI File': str,
        'Material': str,
        'Sample Name': str,
        'Scan Number': int,
        'Scan List': ast.literal_eval,
        'Theta Scan List': ast.literal_eval,
        'Theta Scan Number': lambda v: None if v == 'None' else int(v),
        'Temperature': int,
        'Mask File': str,
        'Specfile': str,
        'Temperature Directory': str,
        'Image Directory': str,
        'Output Directory': str,
        'UB': lambda v: np.array(ast.literal_eval(v)),
        'Substrate Lattice Params': ast.literal_eval,
        'Output Tag': str,
        'Max Intensity': lambda v: None if v == 'None' else float(v),
        'Rod HK List': ast.literal_eval,
        'Rod H Window': ast.literal_eval,
        'Rod H Points': int,
        'Rod K Window': ast.literal_eval,
        'Rod K Points': int,
        'L Range': ast.literal_eval,
        'L Points': int,
    }

    cfg = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ': ' not in line:
                raise ValueError(f"{path}:{lineno}: expected 'Key: value', got {line!r}")
            key, value = line.split(': ', 1)
            if key not in converters:
                raise ValueError(f"{path}:{lineno}: unknown key {key!r}")
            cfg[key] = converters[key](value)

    required = ['PONI File', 'Material', 'Sample Name', 'Scan List',
                'Mask File', 'Specfile', 'Temperature Directory',
                'Output Directory', 'UB', 'Substrate Lattice Params',
                'Rod HK List', 'Rod H Window', 'Rod H Points',
                'Rod K Window', 'Rod K Points', 'L Range', 'L Points']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")

    if not cfg['Rod HK List']:
        raise ValueError("'Rod HK List' is empty: nothing to reconstruct")
    for key in ('Rod H Window', 'Rod K Window', 'L Range'):
        limits = tuple(float(x) for x in cfg[key])
        if len(limits) != 2 or not limits[0] < limits[1]:
            raise ValueError(f"{key} must be (minimum, maximum)")
        cfg[key] = limits
    for key in ('Rod H Points', 'Rod K Points', 'L Points'):
        if int(cfg[key]) < 2:
            raise ValueError(f"{key} must be >= 2")

    cfg.setdefault('Theta Scan List', [])
    cfg.setdefault('Theta Scan Number', None)
    return cfg


def build_rods(cfg):
    """Return ``(rods, L)``: one accumulator dict per (h0, k0) and the shared L
    grid. Each rod carries its own sorted float64 H/K bin grids (offset from the
    integer center) and zeroed float32 ``data``/``norm`` accumulators."""
    L = np.linspace(*cfg['L Range'], int(cfg['L Points'])).astype(np.float64)
    h_lo, h_hi = cfg['Rod H Window']
    k_lo, k_hi = cfg['Rod K Window']
    nH, nK, nL = int(cfg['Rod H Points']), int(cfg['Rod K Points']), len(L)
    rods = []
    for h0, k0 in cfg['Rod HK List']:
        h0, k0 = int(h0), int(k0)
        H = np.linspace(h0 + h_lo, h0 + h_hi, nH).astype(np.float64)
        K = np.linspace(k0 + k_lo, k0 + k_hi, nK).astype(np.float64)
        data = np.zeros(nH * nK * nL, dtype=np.float32)
        rods.append({'h0': h0, 'k0': k0, 'H': H, 'K': K,
                     'data': data, 'norm': np.zeros_like(data)})
    return rods, L


def output_filename(cfg):
    scans = '_'.join(str(s) for s in cfg.get('Scan List', []))
    base = f"{cfg['Material']}_{cfg['Sample Name']}_scans_{scans}"
    tag = cfg.get('Output Tag') or 'rods'
    return f"{base}_{tag}.nxs"


def _rod_group_name(h0, k0):
    """HDF5-safe group name for a rod, with 'm' for a minus sign so the name
    has no characters that confuse a NeXus path (h0, k0 are also stored as
    fields in the group for unambiguous read-back)."""
    fmt = lambda v: ('m' if v < 0 else '') + str(abs(int(v)))
    return f"rod_{fmt(h0)}_{fmt(k0)}"


# ----------------------------------------------------------------------
# Frame processing
# ----------------------------------------------------------------------

def transform_scan_rods(scan_num, image_dir, geom, cfg, rods, L, UB):
    """Histogram one phi scan into every rod (eta, chi fixed; phi per frame).

    Frames are read once (prefetched) and binned into each rod with its own
    ``HKLHIST`` call -- mirrors ``autoRSM.transform_scan`` but fans out to the
    rod list instead of one grid."""
    q, inv_solidangle, _ = geom
    mask_bool = (fabio.open(cfg['Mask File']).data > 0.5).ravel()
    unmasked = ~mask_bool
    max_intensity = cfg.get('Max Intensity')

    imgfiles = auto_rsm.list_images(image_dir)
    scan = SpecDataFile(cfg['Specfile']).getScan(scan_num)
    phi = np.asarray(scan.data['phi'])
    chi = float(scan.positioner['chi'])
    eta = float(scan.positioner['th'])
    icnorm = np.asarray(scan.data['ic2'])
    icnorm = icnorm / np.average(icnorm)

    if len(imgfiles) != len(phi):
        print(f"WARNING scan {scan_num}: {len(imgfiles)} images vs "
              f"{len(phi)} spec points; using the first "
              f"{min(len(imgfiles), len(phi))}.")
    nframes = min(len(imgfiles), len(phi))

    overloaded = 0
    frames = auto_rsm.iter_frames(image_dir, imgfiles[:nframes])
    for i, img in enumerate(tqdm.tqdm(frames, total=nframes,
                                      desc=f"scan {scan_num}")):
        if icnorm[i] <= 0.0:
            continue
        if auto_rsm.is_overloaded(img, unmasked, max_intensity):
            overloaded += 1
            continue
        counts = auto_rsm.make_counts(img, mask_bool, inv_solidangle, icnorm[i])
        M = hkl_ben.rotation_matrix(eta, chi, phi[i], UB)
        for rod in rods:
            hkl_ben.HKLHIST(q, M, counts, rod['H'], rod['K'], L,
                            rod['data'], rod['norm'])
    if overloaded:
        print(f"scan {scan_num}: skipped {overloaded} overloaded frame(s) "
              f"(peak > {max_intensity:g})")
    return overloaded


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def save_rods(fout, rods, L, UB, cfg):
    """Write one NeXus file with an ``NXdata`` per rod plus run metadata."""
    nL = len(L)
    Lf32 = L.astype('float32')
    root = NXroot()
    entry = NXentry()
    root['entry'] = entry
    for rod in rods:
        nH, nK = len(rod['H']), len(rod['K'])
        counts = (rod['data'].clip(0.0) / rod['norm'].clip(0.9)).reshape(
            nH, nK, nL)
        Hf = NXfield(rod['H'].astype('float32'), name='H', long_name='H')
        Kf = NXfield(rod['K'].astype('float32'), name='K', long_name='K')
        Lf = NXfield(Lf32, name='L', long_name='L')
        group = NXdata(NXfield(counts, name='counts', long_name='counts'),
                       (Hf, Kf, Lf))
        group.norm = NXfield(rod['norm'].reshape(nH, nK, nL), name='norm')
        group.h0 = NXfield(np.int32(rod['h0']), name='h0')
        group.k0 = NXfield(np.int32(rod['k0']), name='k0')
        entry[_rod_group_name(rod['h0'], rod['k0'])] = group
    # Run-level metadata: enough to re-derive the rods and the orientation.
    entry['UB'] = NXfield(np.asarray(UB, dtype='float64'), name='UB')
    entry['substrate_lattice'] = NXfield(
        np.asarray(cfg['Substrate Lattice Params'], dtype='float64'),
        name='substrate_lattice')
    entry['scan_list'] = NXfield(
        np.asarray(list(cfg.get('Scan List', [])), dtype='int32'),
        name='scan_list')
    entry['hk_centers'] = NXfield(
        np.asarray([[r['h0'], r['k0']] for r in rods], dtype='int32'),
        name='hk_centers')
    root.save(fout)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct high-resolution HKL rods (CTR) in one pass')
    parser.add_argument('data_file', help='Path to the rod config file')
    args = parser.parse_args()

    cfg = parse_config(args.data_file)
    poni = pyFAI.load(cfg['PONI File'])
    UB = cfg['UB']

    rods, L = build_rods(cfg)
    out_dir = os.path.join(cfg['Output Directory'], 'rod_objects')
    os.makedirs(out_dir, exist_ok=True)
    fout = os.path.join(out_dir, output_filename(cfg))
    while os.path.exists(fout):
        fout = fout[:-4] + '_more.nxs'

    nH, nK, nL = (int(cfg['Rod H Points']), int(cfg['Rod K Points']), len(L))
    voxels = len(rods) * nH * nK * nL
    print(f"UB:\n{UB}")
    print(f"Output file: {fout}")
    print(f"{len(rods)} rod(s), each {nH} x {nK} x {nL} "
          f"({voxels:,} voxels, ~{voxels * 8 / 1e9:.2f} GB accumulators)")
    print(f"L: [{L[0]:.4f}, {L[-1]:.4f}]")

    # Frame-independent detector geometry, computed once.
    q = hkl_ben.detector_q(poni)
    inv_solidangle = (1.0 / poni.solidAngleArray()).astype(np.float32).ravel()
    geom = (q, inv_solidangle, poni.twoThetaArray())

    phi_scans = list(cfg.get('Scan List', []))
    if not phi_scans:
        raise ValueError("Nothing to process: 'Scan List' is empty.")

    skipped = 0
    for scan in phi_scans:
        image_dir = os.path.join(cfg['Temperature Directory'],
                                 f"{cfg['Material']}_{scan:03d}") + os.sep
        skipped += transform_scan_rods(scan, image_dir, geom, cfg, rods, L, UB)

    max_intensity = cfg.get('Max Intensity')
    if max_intensity is not None:
        print(f"Total overloaded frames skipped: {skipped} "
              f"(peak > {max_intensity:g})")

    save_rods(fout, rods, L, UB, cfg)
    print(f"Saved {fout}")


if __name__ == '__main__':
    main()
