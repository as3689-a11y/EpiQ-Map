#!/usr/bin/env python
"""
autoRSM.py -- Reconstruct a 3D reciprocal-space map (RSM) from area-detector
images collected during phi scans (and optionally theta scans) at QM2/CHESS.

Usage:
    python autoRSM.py config.txt

The config file is a plain-text "Key: value" file; see config_example.txt
and README.md for the full list of keys.

Pipeline per frame:
    1. read CBF image (prefetched on a background thread while the previous
       frame is being histogrammed)
    2. apply the static mask (same mask for phi and theta scans)
    3. normalize by ion chamber and solid angle
    4. fused C kernel (hklBen.HKLHIST): rotate detector q into HKL with the
       per-frame goniometer matrix and accumulate into the 3D histogram

Output: a NeXus file containing counts (= data/norm) plus the raw data and
norm arrays, on the (H, K, L) grid.
"""

import argparse
import ast
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import fabio
import pyFAI
import tqdm
from nexusformat.nexus import NXdata, NXfield, nxsetmemory
from spec2nexus.spec import SpecDataFile

import hklBen

nxsetmemory(8000)

GRID_POINTS = 1000  # points per H, K, L axis


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def parse_config(path):
    """Parse the 'Key: value' config file into a dict.

    Blank lines and lines starting with '#' are ignored.
    Unknown keys raise an error so that typos do not silently disappear.
    """
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
        'H Range': ast.literal_eval,
        'K Range': ast.literal_eval,
        'L Range': ast.literal_eval,
        'Grid Shape': ast.literal_eval,
        'Output Tag': str,
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
                'Output Directory']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")

    if 'UB' in cfg and 'Substrate Lattice Params' not in cfg:
        # The old script silently fell back to UB = identity here.
        raise ValueError("'UB' given without 'Substrate Lattice Params'; "
                         "both are needed for an indexed (HKL) map.")

    grid_keys = ('H Range', 'K Range', 'L Range', 'Grid Shape')
    supplied = [key in cfg for key in grid_keys]
    if any(supplied) and not all(supplied):
        raise ValueError("custom grids require H Range, K Range, L Range, and Grid Shape together")
    if all(supplied):
        shape = tuple(int(n) for n in cfg['Grid Shape'])
        if len(shape) != 3 or any(n < 2 for n in shape):
            raise ValueError("Grid Shape must contain three integers >= 2")
        for key in grid_keys[:3]:
            limits = tuple(float(x) for x in cfg[key])
            if len(limits) != 2 or not limits[0] < limits[1]:
                raise ValueError(f"{key} must be (minimum, maximum)")
            cfg[key] = limits
        cfg['Grid Shape'] = shape

    cfg.setdefault('Theta Scan Number', None)
    cfg.setdefault('Theta Scan List', [])
    return cfg


def build_grids(cfg, poni):
    """Return (H, K, L, UB, fout_dir) for indexed or unindexed maps."""
    if all(key in cfg for key in ('H Range', 'K Range', 'L Range', 'Grid Shape')):
        nH, nK, nL = cfg['Grid Shape']
        H = np.linspace(*cfg['H Range'], nH)
        K = np.linspace(*cfg['K Range'], nK)
        L = np.linspace(*cfg['L Range'], nL)
        UB = cfg.get('UB', np.identity(3))
        subdir = 'indexed_objects' if 'UB' in cfg else 'transformed_objects'
        return H, K, L, UB, os.path.join(cfg['Output Directory'], subdir)

    qmag = poni.qArray()
    out_max = qmag[0, round(poni.poni2 / poni.detector.pixel2)] / 10.0
    in_max = qmag[-1, 0] / 10.0
    qmax = np.max(qmag) / 10.0

    if 'UB' in cfg:
        UB = cfg['UB']
        a, b, c = cfg['Substrate Lattice Params']
        H = np.linspace(-qmax * a / (2 * np.pi), qmax * a / (2 * np.pi), GRID_POINTS)
        K = np.linspace(-qmax * b / (2 * np.pi), qmax * b / (2 * np.pi), GRID_POINTS)
        L = np.linspace(0.0, qmax * c / (2 * np.pi), GRID_POINTS)
        subdir = 'indexed_objects'
    else:
        UB = np.identity(3)
        H = np.linspace(-in_max, in_max, GRID_POINTS)
        K = np.linspace(-in_max, in_max, GRID_POINTS)
        L = np.linspace(0.0, out_max, GRID_POINTS)
        subdir = 'transformed_objects'

    return H, K, L, UB, os.path.join(cfg['Output Directory'], subdir)


def output_filename(cfg):
    """Output filename. The 'scans_' list contains the phi scans followed
    by the theta scans, so a phi+theta (or theta-only) run is
    distinguishable from a phi-only run. Works with an empty phi list.

    An explicit 'Output Tag' (set per indexed reconstruction, e.g. the
    transfer-matrix type plus H/K/L range) names the file so the many indexed
    maps of a single scan stay distinguishable. Without a tag, an unindexed
    lab-frame-Q map is suffixed '_full' and a bare indexed map '_out'.
    """
    scan_nums = list(cfg.get('Scan List', [])) + \
        list(cfg.get('Theta Scan List', []))
    scans = '_'.join(str(s) for s in scan_nums)
    base = f"{cfg['Material']}_{cfg['Sample Name']}_scans_{scans}"
    tag = cfg.get('Output Tag')
    if tag:
        return f"{base}_{tag}.nxs"
    return f"{base}_{'out' if 'UB' in cfg else 'full'}.nxs"


# ----------------------------------------------------------------------
# Frame processing
# ----------------------------------------------------------------------

def list_images(image_dir):
    return sorted(f for f in os.listdir(image_dir) if f.endswith('cbf'))


def iter_frames(image_dir, imgfiles):
    """Yield image data arrays, prefetching the next file on a worker thread
    so that CBF read/decompression overlaps with histogramming."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda p: fabio.open(p).data,
                             os.path.join(image_dir, imgfiles[0]))
        for i in range(len(imgfiles)):
            img = future.result()
            if i + 1 < len(imgfiles):
                future = pool.submit(lambda p: fabio.open(p).data,
                                     os.path.join(image_dir, imgfiles[i + 1]))
            yield img


def make_counts(img, mask_bool, inv_solidangle, icnorm_i):
    """Normalized float32 intensity array; masked pixels set negative."""
    counts = np.multiply(img.ravel(), inv_solidangle,
                         dtype=np.float32) * np.float32(1.0 / icnorm_i)
    counts[mask_bool] = -2.0
    return counts


def transform_scan(scan_num, image_dir, geom, cfg, H, K, L, UB, data, norm):
    """Histogram one phi scan (eta, chi fixed; phi varies per frame)."""
    q, inv_solidangle, _ = geom
    mask_bool = (fabio.open(cfg['Mask File']).data > 0.5).ravel()

    imgfiles = list_images(image_dir)
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

    frames = iter_frames(image_dir, imgfiles[:nframes])
    for i, img in enumerate(tqdm.tqdm(frames, total=nframes,
                                      desc=f"scan {scan_num}")):
        if icnorm[i] <= 0.0:
            continue
        counts = make_counts(img, mask_bool, inv_solidangle, icnorm[i])
        M = hklBen.rotation_matrix(eta, chi, phi[i], UB)
        hklBen.HKLHIST(q, M, counts, H, K, L, data, norm)


def theta_scan(scan_num, image_dir, geom, cfg, H, K, L, UB, data, norm):
    """Histogram one theta scan (phi, chi fixed; eta varies per frame).

    Uses the same static mask as the phi-scan path -- no horizon cut.
    (The original thetaRSM contained horizon-mask code, but it computed the
    cutoff row from a flattened tth array, so the slice was always empty
    and the except: pass swallowed the error -- the mask was never applied.
    Reproducing that here means: static mask only.)
    """
    q, inv_solidangle, _ = geom
    mask_bool = (fabio.open(cfg['Mask File']).data > 0.5).ravel()

    imgfiles = list_images(image_dir)
    scan = SpecDataFile(cfg['Specfile']).getScan(scan_num)
    eta = np.asarray(scan.data['th'])
    phi = float(scan.positioner['phi'])
    chi = float(scan.positioner['chi'])
    icnorm = np.asarray(scan.data['ic2'])
    icnorm = icnorm / np.average(icnorm)

    if len(imgfiles) != len(eta):
        print(f"WARNING theta scan {scan_num}: {len(imgfiles)} images vs "
              f"{len(eta)} spec points; using the first "
              f"{min(len(imgfiles), len(eta))}.")
    nframes = min(len(imgfiles), len(eta))

    frames = iter_frames(image_dir, imgfiles[:nframes])
    for i, img in enumerate(tqdm.tqdm(frames, total=nframes,
                                      desc=f"theta scan {scan_num}")):
        if icnorm[i] <= 0.0:
            continue
        counts = make_counts(img, mask_bool, inv_solidangle, icnorm[i])
        M = hklBen.rotation_matrix(eta[i], chi, phi, UB)
        hklBen.HKLHIST(q, M, counts, H, K, L, data, norm)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct a 3D RSM from a scan-log config file')
    parser.add_argument('data_file', help='Path to the config file')
    args = parser.parse_args()

    cfg = parse_config(args.data_file)
    poni = pyFAI.load(cfg['PONI File'])

    H, K, L, UB, out_dir = build_grids(cfg, poni)
    os.makedirs(out_dir, exist_ok=True)
    fout = os.path.join(out_dir, output_filename(cfg))
    while os.path.exists(fout):
        fout = fout[:-4] + '_more.nxs'

    for key in ('Material', 'Sample Name', 'Scan Number', 'Scan List',
                'Theta Scan Number', 'Theta Scan List', 'Temperature',
                'PONI File', 'Mask File', 'Specfile',
                'Temperature Directory', 'Image Directory'):
        if key in cfg:
            print(f"{key}: {cfg[key]}")
    print(f"UB:\n{UB}")
    print(f"Output file: {fout}")
    print(f"H: [{H[0]:.4f}, {H[-1]:.4f}]  K: [{K[0]:.4f}, {K[-1]:.4f}]  "
          f"L: [{L[0]:.4f}, {L[-1]:.4f}]  ({len(H)} x {len(K)} x {len(L)} voxels)")

    # Frame-independent detector geometry, computed once.
    q = hklBen.detector_q(poni)
    inv_solidangle = (1.0 / poni.solidAngleArray()).astype(np.float32).ravel()
    tth2d = poni.twoThetaArray()
    geom = (q, inv_solidangle, tth2d)

    data = np.zeros(len(H) * len(K) * len(L), dtype=np.float32)
    norm = np.zeros_like(data)

    phi_scans = list(cfg.get('Scan List', []))
    theta_scans = list(cfg.get('Theta Scan List', []))
    if not phi_scans and not theta_scans:
        raise ValueError("Nothing to process: both 'Scan List' and "
                         "'Theta Scan List' are empty.")

    for scan in phi_scans:
        image_dir = os.path.join(cfg['Temperature Directory'],
                                 f"{cfg['Material']}_{scan:03d}") + os.sep
        transform_scan(scan, image_dir, geom, cfg, H, K, L, UB, data, norm)

    for scan in theta_scans:
        image_dir = os.path.join(cfg['Temperature Directory'],
                                 f"{cfg['Material']}_{scan:03d}") + os.sep
        theta_scan(scan, image_dir, geom, cfg, H, K, L, UB, data, norm)

    dataout = (data.clip(0.0) / norm.clip(0.9)).reshape(len(H), len(K), len(L))

    Hf = NXfield(H.astype('float32'), name='H', long_name='H')
    Kf = NXfield(K.astype('float32'), name='K', long_name='K')
    Lf = NXfield(L.astype('float32'), name='L', long_name='L')
    counts = NXfield(dataout, name='counts', long_name='counts')

    G = NXdata(counts, (Hf, Kf, Lf))
    # Keep norm so that empty voxels (norm == 0) can be distinguished from
    # voxels that genuinely measured zero intensity.
    G.norm = NXfield(norm.reshape(len(H), len(K), len(L)), name='norm')
    G.save(fout)
    print(f"Saved {fout}")


if __name__ == '__main__':
    main()
