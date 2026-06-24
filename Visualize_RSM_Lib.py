"""
Visualize_RSM_Lib.py -- analysis library for 3D reciprocal-space maps
(autoRSM.py output): automated peak finding, U-matrix determination from
peak-pair geometry, resampling on rotated grids, slicing, and line cuts.

Workflow (see Visualize_RSM_clean.ipynb):

    load 3D data
      -> find peaks                      find_peaks_in_RSM
      -> align with predefined basis     compute_U_matrix (DBSCAN peaks ->
         vectors, i.e. find U            peak-pair vectors -> categorize by
                                         angle to v1,v2,v3 -> orthonormal
                                         triple with unit volume)
      -> optional rotation / new axes    rotate_U, view_axes_transform
      -> interpolate                     transform_slab (fast) or
                                         main_transformation (legacy RGI)

Conventions:
    * U columns are the measured unit vectors of the nominal basis:
      U @ e_i = measured direction of basis vector i.
    * Resampling: the value at new-grid point x is taken from the data at
      U @ x (so the new axes are the nominal basis).

Changes vs the previous version (results differ where these applied):
    * fi() returns the nearest grid index (argmin). The old version
      returned "first index above value, plus one", shifting every slice,
      rod, and line-cut anchor by +1..+2 bins and crashing for values
      beyond the grid.
    * Missing scipy imports fixed: compute_U_matrix and rotate_* work from
      the library again (no need for inline copies in the notebook).
    * angle_between actually clips: exactly (anti)parallel peak pairs are
      no longer silently dropped as NaN.
    * Gaussian width filter uses |sigma|: broad fits with negative fitted
      width no longer pass the peak filter.
    * get_rod: fixed the dL typo (L[10]-[9]) that collapsed the L
      integration window for H- and K-rods.
"""

import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


# ======================================================================
# Basics
# ======================================================================

def fi(axis, value):
    """Index of the grid point nearest to value.

    NOTE: the previous version returned (first index above value) + 1,
    a systematic +1..+2 bin shift; all positions move slightly with this
    fix.
    """
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


# A single rod inside a multi-rod CTR file (autoRSM_rods) is addressed as
# "file.nxs::rod_<h>_<k>" so the whole viewer pipeline (load_source, caching,
# interpolation) can treat one rod exactly like a standalone .nxs.
ROD_TOKEN = '::'


def load_rsm(path):
    """Load an autoRSM NeXus file fully into memory.

    Returns (data, H, K, L). Loading once makes every projection, slice,
    and interpolator construction run at memory speed; keeping the lazy
    NXfield re-reads the volume from disk on each access.

    A ``"file.nxs::rod_<h>_<k>"`` token loads that single rod from a multi-rod
    CTR file (see ``load_rod``) instead of the default ``entry/data`` group.
    """
    file_path, _, group = str(path).partition(ROD_TOKEN)
    if group:
        return load_rod(file_path, group)
    from nexusformat.nexus import nxload, nxsetconfig
    nxsetconfig(memory=8000)
    a = nxload(file_path)
    H = np.array(a.entry.data.H, dtype=float)
    K = np.array(a.entry.data.K, dtype=float)
    L = np.array(a.entry.data.L, dtype=float)
    data = np.asarray(a.entry.data.counts, dtype=np.float32)
    return data, H, K, L


def list_rods(path):
    """List the rods in a multi-rod CTR file (``autoRSM_rods`` output).

    Returns ``[(h0, k0, group_name), ...]`` sorted by (h0, k0), or ``[]`` if
    ``path`` is an ordinary single-volume .nxs. Each ``group_name`` pairs with
    ``path`` as ``f"{path}{ROD_TOKEN}{group_name}"`` for ``load_rsm``.
    """
    from nexusformat.nexus import nxload
    file_path = str(path).partition(ROD_TOKEN)[0]
    entry = nxload(file_path).entry
    rods = []
    for name in entry.entries:
        if not name.startswith('rod_'):
            continue
        node = entry[name]
        try:
            rods.append((int(node['h0']), int(node['k0']), name))
        except Exception:
            continue
    return sorted(rods)


def load_rod(path, group):
    """Load one rod's volume ``(data, H, K, L)`` from a multi-rod CTR file."""
    from nexusformat.nexus import nxload, nxsetconfig
    nxsetconfig(memory=8000)
    node = nxload(str(path).partition(ROD_TOKEN)[0]).entry[group]
    H = np.array(node.H, dtype=float)
    K = np.array(node.K, dtype=float)
    L = np.array(node.L, dtype=float)
    data = np.asarray(node.counts, dtype=np.float32)
    return data, H, K, L


# ======================================================================
# Peak centering and rods (explicit data arguments; no globals)
# ======================================================================

def fix_HKL(data, H, K, L, hkl, Q_range=(0.05, 0.05, 0.05), verbose=True):
    """Refine a nominal (h, k, l) to the intensity maximum in a small box.

    The box has total size Q_range per axis. Returns np.array([h, k, l])
    of the maxima of the axis-summed profiles (bin resolution).
    """
    h, k, l = hkl
    sH = slice(fi(H, h - Q_range[0] / 2), fi(H, h + Q_range[0] / 2) + 1)
    sK = slice(fi(K, k - Q_range[1] / 2), fi(K, k + Q_range[1] / 2) + 1)
    sL = slice(fi(L, l - Q_range[2] / 2), fi(L, l + Q_range[2] / 2) + 1)

    y = np.asarray(data[sH, sK, sL])          # read the ROI once
    h = H[sH.start + int(np.argmax(y.sum(axis=(1, 2))))]
    k = K[sK.start + int(np.argmax(y.sum(axis=(0, 2))))]
    l = L[sL.start + int(np.argmax(y.sum(axis=(0, 1))))]
    if verbose:
        print(f"fix_HKL: {tuple(hkl)} -> ({h:.4f}, {k:.4f}, {l:.4f})")
    return np.array([h, k, l])


def get_rod(data, H, K, L, hkl, delta=0.01, film_norm=2, center=True,
            Q_range=(0.05, 0.05, 0.05)):
    """Integrated intensity rod through a peak along one axis.

    film_norm selects the rod axis: 0 = H, 1 = K, 2 = L. delta is the
    total integration width (axis units) perpendicular to the rod.
    Returns the 1D rod (length of the chosen axis).
    """
    if center:
        hkl = fix_HKL(data, H, K, L, hkl, Q_range=Q_range, verbose=False)

    half = lambda step: max(int(np.ceil((delta / step - 1) / 2)), 0)
    dH, dK, dL = half(H[1] - H[0]), half(K[1] - K[0]), half(L[1] - L[0])
    i_h, i_k, i_l = fi(H, hkl[0]), fi(K, hkl[1]), fi(L, hkl[2])

    rods = [
        lambda: data[:, i_k - dK:i_k + dK + 1,
                     i_l - dL:i_l + dL + 1].sum(axis=(1, 2)),
        lambda: data[i_h - dH:i_h + dH + 1, :,
                     i_l - dL:i_l + dL + 1].sum(axis=(0, 2)),
        lambda: data[i_h - dH:i_h + dH + 1,
                     i_k - dK:i_k + dK + 1, :].sum(axis=(0, 1)),
    ]
    return np.asarray(rods[film_norm]())


# ======================================================================
# Automated peak finding
# ======================================================================

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-(x - mean) ** 2
                              / (2 * standard_deviation ** 2))


def fit_gaussian_and_find_peak(data1d, axis_range):
    """Fit a Gaussian to a 1D profile; returns (mean, amplitude, |sigma|).

    Returns NaNs if the fit fails. |sigma| guards against the
    sign-degenerate fitted width passing 'width < peak_width' filters.
    """
    try:
        popt, _ = curve_fit(
            gaussian, axis_range, data1d,
            p0=[axis_range[np.argmax(data1d)], np.max(data1d), 5e-2])
        return popt[0], popt[1], abs(popt[2])
    except RuntimeError:
        return np.nan, np.nan, np.nan


def find_peaks_in_RSM(data, H, K, L, dQ=0.1, threshold=1e5,
                      peak_width=0.01):
    """Detect Bragg peaks in the volume.

    Voxels above `threshold` are clustered with DBSCAN (in index space);
    each cluster is boxed with a margin of dQ (axis units) and its three
    axis projections are fit with Gaussians. A peak is accepted when all
    three fits converge with width < peak_width.

    Returns (peaks, heights): lists of (H, K, L) positions and fitted
    amplitudes.
    """
    dH, dK, dL = H[1] - H[0], K[1] - K[0], L[1] - L[0]
    margin = np.array([round(dQ / dH), round(dQ / dK), round(dQ / dL)],
                      dtype=int)

    indices = np.argwhere(data > threshold)
    if len(indices) == 0:
        return [], []
    labels = DBSCAN(eps=3, min_samples=2).fit(indices).labels_

    peaks, heights = [], []
    for label in np.unique(labels):
        if label == -1:                       # DBSCAN noise
            continue
        pts = indices[labels == label]
        lo = np.maximum(pts.min(axis=0) - margin, 0)
        hi = np.minimum(pts.max(axis=0) + margin, data.shape)

        box = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        fits = [
            fit_gaussian_and_find_peak(np.nanmean(box, axis=(1, 2)),
                                       H[lo[0]:hi[0]]),
            fit_gaussian_and_find_peak(np.nanmean(box, axis=(0, 2)),
                                       K[lo[1]:hi[1]]),
            fit_gaussian_and_find_peak(np.nanmean(box, axis=(0, 1)),
                                       L[lo[2]:hi[2]]),
        ]
        pos = [f[0] for f in fits]
        amp = [f[1] for f in fits]
        wid = [f[2] for f in fits]
        if not np.any(np.isnan(pos)) and all(w < peak_width for w in wid):
            peaks.append(tuple(pos))
            heights.append(tuple(amp))
    return peaks, heights


def filter_peaks(peaks, Hi, Hf, Ki, Kf, Li, Lf):
    """Keep peaks inside the box [Hi,Hf] x [Ki,Kf] x [Li,Lf]."""
    return [p for p in peaks
            if Hi <= p[0] <= Hf and Ki <= p[1] <= Kf and Li <= p[2] <= Lf]


def plot_3d_view(ax, peaks, elev=90, azim=0):
    """Scatter the found peaks on a 3D axis (use add_subplot(projection='3d'))."""
    p = np.asarray(peaks)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='r', marker='o')
    ax.set_xlabel('H'); ax.set_ylabel('K'); ax.set_zlabel('L')
    ax.view_init(elev=elev, azim=azim)


# ======================================================================
# U matrix from peak-pair geometry
# ======================================================================

def angle_between(v1, v2):
    """Angle in degrees between two vectors (clipped: parallel pairs give
    exactly 0/180 instead of NaN)."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def categorize_peak_pairs(peaks, v1, v2, v3, difference=10):
    """Difference vectors between all peak pairs, sorted by direction.

    Every ordered pair (i != j) gives a difference vector; vectors within
    `difference` degrees of a reference direction are normalized and
    collected into that category. Returns (cat_v1, cat_v2, cat_v3).
    """
    cats = ([], [], [])
    refs = (v1, v2, v3)
    peaks = [np.asarray(p, dtype=float) for p in peaks]
    for i, j in itertools.permutations(range(len(peaks)), 2):
        diff = peaks[j] - peaks[i]
        for ref, cat in zip(refs, cats):
            if angle_between(diff, ref) <= difference:
                cat.append(diff / np.linalg.norm(diff))
    return cats


def are_approx_orthogonal(vectors, threshold):
    """True if all mutual dot products are below threshold in magnitude."""
    return all(abs(np.dot(a, b)) < threshold
               for a, b in itertools.combinations(vectors, 2))


def find_orthogonal_subset(cat_v1, cat_v2, cat_v3, threshold=0.001):
    """First triple (one vector per category) that is pairwise orthogonal
    within threshold. Returns (vectors, found)."""
    for vectors in itertools.product(cat_v1, cat_v2, cat_v3):
        if are_approx_orthogonal(vectors, threshold):
            print("Approximately orthogonal subset found")
            return vectors, True
    return [], False


def find_closest_basis_with_volume(cat_v1, cat_v2, cat_v3, desired_volume,
                                   threshold=0.001):
    """Triple (one unit vector per category) whose scalar triple product is
    closest to desired_volume (within threshold). For unit vectors,
    |triple product| = 1 only for an orthonormal set, so desired_volume=1
    selects the most orthonormal basis. Returns (best_basis, found).

    Vectorized: all |cat_v1| x |cat_v2| x |cat_v3| triple products are
    computed in one einsum (the per-triple Python loop took seconds for
    ~50 peaks). Tie-breaking matches the original iteration order.
    """
    if not (cat_v1 and cat_v2 and cat_v3):
        return [], False
    A = np.asarray(cat_v1)                     # (n1, 3)
    B = np.asarray(cat_v2)                     # (n2, 3)
    C = np.asarray(cat_v3)                     # (n3, 3)
    cross = np.cross(A[:, None, :], B[None, :, :])        # (n1, n2, 3)
    triple = np.einsum('ijx,kx->ijk', cross, C)           # (n1, n2, n3)
    deviation = np.abs(np.abs(triple) - desired_volume)
    flat = int(np.argmin(deviation))           # first minimum = product order
    if deviation.ravel()[flat] > threshold:
        return [], False
    i, j, k = np.unravel_index(flat, deviation.shape)
    print(f"Basis found with {deviation[i, j, k]:f} difference in volume")
    return (A[i], B[j], C[k]), True


def orthonormalize(U):
    """Nearest rotation matrix to U (polar decomposition via SVD).

    The basis from find_closest_basis_with_volume is orthonormal only to
    ~threshold; using U directly applies a slight shear in the transform.
    """
    W, _, Vt = np.linalg.svd(U)
    R = W @ Vt
    if np.linalg.det(R) < 0:                  # keep a proper rotation
        W[:, -1] *= -1
        R = W @ Vt
    return R


def compute_U_matrix(data, H, K, L, angle_deg=0, peak_width=0.005,
                     dQ=None, difference=5, threshold_V=1e-6,
                     max_iterations=3, ortho=False, verbose=True):
    """Find the orientation matrix U automatically from the data.

    1. Threshold the volume at max/25 and find peaks (find_peaks_in_RSM);
       if no valid basis is found, lower the threshold 5x and retry, up to
       max_iterations times.
    2. Build all peak-pair difference vectors and keep those within
       `difference` degrees of the predefined basis vectors
       (x, y, z rotated in-plane by angle_deg).
    3. Pick the triple closest to unit volume (orthonormal) within
       threshold_V.

    Returns U (3x3, columns = measured basis directions), or None if no
    basis was found. ortho=True replaces U with the nearest exact rotation
    (recommended; default False preserves previous behavior).

    dQ defaults to 6 * peak_width (the value the notebook converged on).
    """
    if dQ is None:
        dQ = 6 * peak_width
    R = Rotation.from_euler('z', np.radians(angle_deg)).as_matrix()
    v1, v2, v3 = R.T            # columns of R = rotated x, y, z

    threshold_I = np.nanmax(data) / 25
    vectors, found, peaks = [], False, []
    for _ in range(max_iterations):
        peaks, _ = find_peaks_in_RSM(data, H, K, L, dQ=dQ,
                                     threshold=threshold_I,
                                     peak_width=peak_width)
        if verbose:
            print(f"{len(peaks)} peaks found "
                  f"(threshold {threshold_I:.3g}). Searching for U ...")
        if peaks:
            cats = categorize_peak_pairs(peaks, v1, v2, v3,
                                         difference=difference)
            vectors, found = find_closest_basis_with_volume(
                *cats, 1, threshold=threshold_V)
            if found:
                break
        threshold_I /= 5

    if not found:
        print("No U matrix found! Try reducing the peak width or "
              "threshold.")
        return None

    vv1, vv2, vv3 = vectors
    U = np.column_stack((vv1, vv2, vv3))
    if verbose:
        print(vv1, vv2, vv3)
        print(f"Using {len(peaks)} peaks\n Mutual dot products: "
              f"{np.dot(vv1, vv2):.4f}, {np.dot(vv2, vv3):.4f}, "
              f"{np.dot(vv3, vv1):.4f}")
    return orthonormalize(U) if ortho else U


# ======================================================================
# Substrate-lattice indexing (find U by matching peaks to a known cell)
# ======================================================================

def reciprocal_matrix(a, b, c, alpha, beta, gamma, with_2pi=True):
    """Reciprocal-lattice matrix B* (columns = a*, b*, c*) for a unit cell.

    a, b, c in Angstrom; alpha, beta, gamma in degrees. Real-space vectors
    are built in the standard crystallographic setting (a along x, b in the
    xy-plane), giving B with real lattice vectors as columns; then
    B* = 2*pi * (B^-1)^T so that q = B* . (h,k,l) is in A^-1 with the 2*pi
    convention used by autoRSM (q = 2*pi/d). Set with_2pi=False for q = 1/d.

    A reflection (h,k,l) sits at q = B* . [h,k,l]; |q| = 2*pi/d_hkl.
    """
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = a
    bx, by = b * np.cos(ga), b * np.sin(ga)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    B = np.array([[ax, bx, cx],
                  [0.0, by, cy],
                  [0.0, 0.0, cz]])           # columns = a, b, c vectors
    Bstar = np.linalg.inv(B).T
    if with_2pi:
        Bstar = 2 * np.pi * Bstar
    return Bstar


def load_lattice(filepath, material):
    """Look up a material's lattice constants in a simple text file.

    File format (one material per line, '#' comments allowed); the key is
    the chemical formula, followed by a b c alpha beta gamma. Separators
    may be commas, '=', or whitespace, so all of these parse:

        LaAlO3  3.79 3.79 3.79 90 90 90
        LaAlO3, a=3.79, b=3.79, c=3.79, alpha=90, beta=90, gamma=90
        SrTiO3 = 3.905 3.905 3.905 90 90 90

    Returns (a, b, c, alpha, beta, gamma) as floats.
    """
    import re
    want = material.strip().lower()
    with open(filepath) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            # split off the key (first token before space/comma/equals)
            m = re.match(r'\s*([A-Za-z0-9_.\-()]+)\s*[,=:\s]\s*(.*)', line)
            if not m:
                continue
            key, rest = m.group(1), m.group(2)
            if key.strip().lower() != want:
                continue
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', rest)
            if len(nums) < 6:
                raise ValueError(
                    f"{material}: found {len(nums)} numbers, need 6 "
                    f"(a b c alpha beta gamma) in line: {line!r}")
            return tuple(float(x) for x in nums[:6])
    raise KeyError(f"{material!r} not found in {filepath}")


def _hkl_candidates(Bstar, hkl_max):
    """All (h,k,l) within +-hkl_max (excluding 0,0,0), with their q-vectors
    and |q|. Returns (hkls (N,3) int, qs (N,3), qmags (N,))."""
    rng = range(-hkl_max, hkl_max + 1)
    hkls = np.array([(h, k, l) for h in rng for k in rng for l in rng
                     if (h, k, l) != (0, 0, 0)], dtype=float)
    qs = (Bstar @ hkls.T).T
    qmags = np.linalg.norm(qs, axis=1)
    return hkls, qs, qmags


def refine_lattice_from_indexed_peaks(peaks, hkl, inliers=None,
                                      q_with_2pi=True):
    """Refine a general unit cell from indexed peak centers.

    Fits the unconstrained reciprocal basis ``q = M @ [h,k,l]`` by linear
    least squares.  The direct basis is ``2*pi * inv(M).T`` (or without
    ``2*pi`` for the ``q = 1/d`` convention), from which a, b, c and the
    direct-cell angles alpha, beta, gamma are calculated.

    Reported uncertainties are one standard deviation from the ordinary
    least-squares covariance, using the observed q residual variance and
    propagating it to the six cell parameters with a numerical Jacobian.
    They quantify peak-to-peak scatter; they do not include an absolute
    q-axis calibration uncertainty.

    Returns a dict with ``values`` and ``uncertainties`` in the order
    (a, b, c, alpha, beta, gamma), plus the fitted reciprocal basis, its
    covariance, residuals, RMS error, number of peaks, and degrees of
    freedom.
    """
    P = np.atleast_2d(np.asarray(peaks, dtype=float))
    H = np.atleast_2d(np.asarray(hkl, dtype=float))
    if P.shape != H.shape or P.shape[1] != 3:
        raise ValueError("peaks and hkl must both have shape (N, 3)")

    valid = np.all(np.isfinite(P), axis=1) & np.all(np.isfinite(H), axis=1)
    if inliers is not None:
        mask = np.asarray(inliers, dtype=bool)
        if mask.shape != (len(P),):
            raise ValueError("inliers must have shape (N,)")
        valid &= mask
    P, H = P[valid], H[valid]
    if len(P) < 4 or np.linalg.matrix_rank(H) < 3:
        raise ValueError("need at least four indexed peaks spanning 3D hkl")

    # np.linalg.lstsq gives P = H @ X; M = X.T in q = M @ hkl.
    X, _, _, _ = np.linalg.lstsq(H, P, rcond=None)
    M = X.T
    predicted = H @ X
    residuals = P - predicted
    dof = 3 * len(P) - 9
    rss = float(np.sum(residuals ** 2))
    sigma2 = rss / dof if dof > 0 else np.nan

    # Jacobian of flattened predictions with respect to row-major M.
    J = np.zeros((3 * len(P), 9), dtype=float)
    for i, indices in enumerate(H):
        for component in range(3):
            J[3 * i + component, 3 * component:3 * component + 3] = indices
    cov_M = sigma2 * np.linalg.pinv(J.T @ J)

    scale = 2 * np.pi if q_with_2pi else 1.0

    def cell_from_reciprocal(matrix):
        direct = scale * np.linalg.inv(matrix).T
        av, bv, cv = direct.T
        lengths = [np.linalg.norm(v) for v in (av, bv, cv)]

        def angle(u, v):
            cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        return np.array([*lengths, angle(bv, cv), angle(av, cv),
                         angle(av, bv)])

    values = cell_from_reciprocal(M)
    cell_jac = np.empty((6, 9), dtype=float)
    flat = M.ravel()
    for j in range(9):
        step = 1e-6 * max(abs(flat[j]), 1.0)
        plus, minus = flat.copy(), flat.copy()
        plus[j] += step
        minus[j] -= step
        cell_jac[:, j] = (cell_from_reciprocal(plus.reshape(3, 3)) -
                          cell_from_reciprocal(minus.reshape(3, 3))) / (2 * step)
    cov_cell = cell_jac @ cov_M @ cell_jac.T
    uncertainties = np.sqrt(np.maximum(np.diag(cov_cell), 0.0))

    return {
        'values': values,
        'uncertainties': uncertainties,
        'covariance': cov_cell,
        'reciprocal_basis': M,
        'reciprocal_covariance': cov_M,
        'residuals': residuals,
        'rms': float(np.sqrt(rss / len(P))),
        'n_peaks': len(P),
        'dof': dof,
    }


def target_orientation_matrix(Bstar, out_x, out_y, out_z, ortho_tol=2.0):
    """Rotation R that puts chosen lattice directions onto the output axes.

    out_x, out_y, out_z are (h,k,l) directions (reciprocal-lattice index
    space) that should point along output +x, +y, +z after transforming.

    The three directions MUST be mutually orthogonal (in cartesian q-space)
    to within ortho_tol degrees of 90 -- a slanted/degenerate choice has no
    well-defined orthonormal frame and raises ValueError. This is the
    constraint that makes the output axes a proper orthonormal basis.

    Returns R with R @ (B*.dir_i normalized) = e_i.
    """
    dirs = [np.asarray(v, dtype=float) for v in (out_x, out_y, out_z)]
    cart = [Bstar @ d for d in dirs]
    cart = [v / np.linalg.norm(v) for v in cart]
    cart = np.array(cart)

    # check mutual orthogonality
    for (a, na), (b, nb) in ((( cart[0], 'x'), (cart[1], 'y')),
                             ((cart[0], 'x'), (cart[2], 'z')),
                             ((cart[1], 'y'), (cart[2], 'z'))):
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))
        if abs(90.0 - ang) > ortho_tol:
            raise ValueError(
                f"target directions for {na} and {nb} are {ang:.1f} deg "
                f"apart, not 90; choose three mutually orthogonal "
                f"directions (e.g. [110],[1-10],[001]).")

    targets = np.eye(3)
    # The three directions must form a right-handed set; a rotation cannot
    # map a left-handed set onto the (right-handed) +x,+y,+z axes.
    if np.linalg.det(cart) < 0:
        print("WARNING: target directions are LEFT-handed "
              f"({out_x}, {out_y}, {out_z}); a pure rotation cannot put all "
              "three on the positive axes. Swap two of them, or negate one "
              "(e.g. use [-1,1,0] instead of [1,-1,0]).")
    rot, _ = Rotation.align_vectors(targets, cart)
    return rot.as_matrix()


def axes_from_normal(Bstar, normal, prefer=None):
    """Build an orthonormal (out_x, out_y, out_z) hkl triple from one normal.

    Given just the substrate surface normal as an (h,k,l) direction, return
    three mutually orthogonal hkl directions suitable for out_x/out_y/out_z
    so that U is fully pinned and reproducible. ``normal`` lands on out_z
    (out-of-plane); two in-plane directions are chosen orthogonal to it.

    Orthogonality is enforced in cartesian q-space using ``Bstar`` (so this
    is correct for non-cubic cells, where hkl-orthogonality differs from
    q-space orthogonality). The in-plane azimuth is picked deterministically:
    the cartesian axis least parallel to the normal seeds out_x, giving a
    stable, right-handed frame. ``prefer`` (an hkl direction) may be passed
    to bias the in-plane out_x toward a preferred crystal direction.

    Returns (out_x, out_y, out_z) as float hkl 3-vectors.
    """
    Bstar = np.asarray(Bstar, dtype=float)
    n_hkl = np.asarray(normal, dtype=float)
    if n_hkl.shape != (3,) or not np.all(np.isfinite(n_hkl)) or np.linalg.norm(n_hkl) == 0:
        raise ValueError("normal must be a finite, non-zero (h,k,l) direction")
    Binv = np.linalg.inv(Bstar)

    # Work in cartesian q-space; convert directions back to hkl at the end.
    z = Bstar @ n_hkl
    z = z / np.linalg.norm(z)
    if prefer is not None:
        seed = Bstar @ np.asarray(prefer, dtype=float)
        if np.linalg.norm(seed) == 0 or np.linalg.norm(np.cross(z, seed)) < 1e-8:
            prefer = None
    if prefer is None:
        # Seed with the cartesian axis least aligned with the normal.
        seed = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = seed - np.dot(seed, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                       # right-handed: x cross y = z

    out = []
    for v in (x, y, z):
        hkl = Binv @ v
        # Rescale so the smallest nonzero component is ~1, then round, so the
        # frame reads as clean integer directions (e.g. [110], [001]). Only
        # the direction matters downstream (target_orientation_matrix
        # re-normalizes), so the overall scale is free to choose.
        nonzero = np.abs(hkl[np.abs(hkl) > 1e-6])
        if len(nonzero):
            hkl = hkl / nonzero.min()
        rounded = np.round(hkl)
        if np.allclose(hkl, rounded, atol=1e-4):
            hkl = rounded
        out.append(hkl)
    return tuple(out)


def _cubic_rotations():
    """The 24 proper rotation matrices of the cubic point group."""
    import itertools
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if round(np.linalg.det(M)) == 1:
                ops.append(M)
    return ops


def index_against_lattice(peaks, lattice, q_with_2pi=True, hkl_max=6,
                          mod_tol=0.04, ang_tol=2.0, inlier_tol=0.05,
                          ortho=True, seed=0, max_pairs=20000,
                          out_x=None, out_y=None, out_z=None, verbose=True):
    """Find U by indexing measured peaks against a known substrate cell.

    RANSAC over peak pairs:
      1. Build B* from the lattice; enumerate candidate (h,k,l) and their
         |q|. For each measured peak, keep hkl candidates whose |q| matches
         |q_peak| within mod_tol (A^-1). This is the modulus filter.
      2. For each pair of measured peaks, the angle between their q-vectors
         must match the angle between a candidate hkl-pair within ang_tol
         (degrees). Each surviving (peak-pair, hkl-pair) defines a rotation
         (Kabsch on the two vectors).
      3. Score each candidate U by how many peaks have ANY candidate hkl
         with |U.(B*.hkl) - q_peak| < inlier_tol. Keep the U with the most
         inliers, then refine U by Kabsch on all inliers (their best hkl).

    peaks   : (N,3) measured q-vectors (same convention as autoRSM).
    lattice : (a,b,c,alpha,beta,gamma).
    Returns dict with:
       'U'        3x3 rotation (q_measured ~= U . B* . hkl)
       'Bstar'    the reciprocal matrix used
       'hkl'      (N,3) assigned integer indices per peak (nan if outlier)
       'inliers'  boolean mask over peaks
       'rms'      RMS |residual| over inliers (A^-1)
       'n_inliers'
    or None if no consistent U is found.
    """
    rngsd = np.random.default_rng(seed)
    P = np.atleast_2d(np.asarray(peaks, dtype=float))
    N = len(P)
    pmag = np.linalg.norm(P, axis=1)

    Bstar = reciprocal_matrix(*lattice, with_2pi=q_with_2pi)
    hkls, qs, qmags = _hkl_candidates(Bstar, hkl_max)

    # 1. per-peak candidate hkl by modulus
    cand = []
    for i in range(N):
        idx = np.where(np.abs(qmags - pmag[i]) < mod_tol)[0]
        cand.append(idx)
    have = [i for i in range(N) if len(cand[i]) > 0]
    if len(have) < 2:
        if verbose:
            print("index: fewer than 2 peaks have a lattice match; "
                  "check formula, q convention, or mod_tol.")
        return None

    # precompute candidate q-directions
    def best_rotation(meas, ref):
        rot, _ = Rotation.align_vectors(meas, ref)
        return rot.as_matrix()

    best = None  # (n_inliers, U)
    # 2. RANSAC over peak pairs (limit total trials)
    pair_list = [(i, j) for a, i in enumerate(have) for j in have[a + 1:]]
    rngsd.shuffle(pair_list)
    trials = 0
    for (i, j) in pair_list:
        ang_meas = angle_between(P[i], P[j])
        ci, cj = cand[i], cand[j]
        # candidate hkl-pairs whose mutual angle matches
        for hi in ci:
            ai = qs[hi]
            for hj in cj:
                trials += 1
                if trials > max_pairs:
                    break
                ang_ref = angle_between(ai, qs[hj])
                if abs(ang_ref - ang_meas) > ang_tol:
                    continue
                # rotation mapping the two reference q's onto the measured
                try:
                    U = best_rotation(np.array([P[i], P[j]]),
                                      np.array([ai, qs[hj]]))
                except Exception:
                    continue
                # 3. score over all peaks
                pred = (U @ qs.T).T            # all candidate q's rotated
                n_in = 0
                for k in have:
                    d = np.linalg.norm(pred[cand[k]] - P[k], axis=1)
                    if d.min() < inlier_tol:
                        n_in += 1
                if best is None or n_in > best[0]:
                    best = (n_in, U)
            if trials > max_pairs:
                break
        if trials > max_pairs:
            break

    if best is None or best[0] < 2:
        if verbose:
            print("index: no consistent orientation found.")
        return None

    U = best[1]
    # assign best hkl per peak and refine on inliers
    assigned = np.full((N, 3), np.nan)
    inliers = np.zeros(N, dtype=bool)
    pred_all = (U @ qs.T).T
    meas_in, ref_in = [], []
    for k in range(N):
        if len(cand[k]) == 0:
            continue
        d = np.linalg.norm(pred_all[cand[k]] - P[k], axis=1)
        jbest = int(np.argmin(d))
        if d[jbest] < inlier_tol:
            h = hkls[cand[k][jbest]]
            assigned[k] = h
            inliers[k] = True
            meas_in.append(P[k])
            ref_in.append(qs[cand[k][jbest]])

    if len(meas_in) >= 2:
        Uref, _ = Rotation.align_vectors(np.array(meas_in),
                                         np.array(ref_in))
        U = Uref.as_matrix()
        res = (U @ np.array(ref_in).T).T - np.array(meas_in)
        rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    else:
        rms = np.nan

    if ortho:
        U = orthonormalize(U)

    # Optional: reorient so chosen lattice directions land on output axes.
    # U maps measured q -> lattice cartesian frame (U^-1 @ q_meas = B*.hkl).
    # R rotates the chosen lattice directions onto the target axes; the
    # reoriented orientation is U_final = U @ R^T.
    # For a cubic cell the indexed frame is only defined up to the 24 cubic
    # rotations, so we additionally pick the equivalent that lands the named
    # directions on the POSITIVE axes (deterministic, reproducible).
    if out_x is not None and out_y is not None and out_z is not None:
        R = target_orientation_matrix(Bstar, out_x, out_y, out_z)
        U = U @ R.T

        a, b, c, al, be, ga = lattice
        is_cubic = (abs(a - b) < 1e-3 and abs(b - c) < 1e-3
                    and abs(al - 90) < 1e-2 and abs(be - 90) < 1e-2
                    and abs(ga - 90) < 1e-2)
        if is_cubic:
            want = np.array([np.asarray(out_x, float),
                             np.asarray(out_y, float),
                             np.asarray(out_z, float)])
            want = (want.T / np.linalg.norm(want, axis=1)).T
            best, best_score = U, -np.inf
            for S in _cubic_rotations():
                Ucand = U @ S.T
                # where do the named directions land in the output frame?
                score = 0.0
                for d, axis in zip(want, np.eye(3)):
                    out = np.linalg.inv(Ucand) @ (Bstar @ d)
                    out = out / np.linalg.norm(out)
                    score += np.dot(out, axis)     # reward +axis alignment
                if score > best_score:
                    best_score, best = score, Ucand
            U = best
        if verbose:
            print(f"reoriented: {out_x}->x, {out_y}->y, {out_z}->z")

    if verbose:
        print(f"index_against_lattice: {int(inliers.sum())}/{N} peaks "
              f"indexed, RMS {rms:.4f} A^-1")
    refined = None
    if inliers.sum() >= 4 and np.linalg.matrix_rank(assigned[inliers]) == 3:
        refined = refine_lattice_from_indexed_peaks(
            P, assigned, inliers=inliers, q_with_2pi=q_with_2pi)
    if verbose and refined is not None:
        names = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
        units = ('A', 'A', 'A', 'deg', 'deg', 'deg')
        print("refined unit cell (1 sigma from peak scatter):")
        for name, value, error, unit in zip(
                names, refined['values'], refined['uncertainties'], units):
            print(f"  {name:5s} = {value:.6f} +/- {error:.6f} {unit}")
    return {'U': U, 'Bstar': Bstar, 'hkl': assigned, 'inliers': inliers,
            'rms': rms, 'n_inliers': int(inliers.sum()),
            'refined_lattice': refined}


def compute_U_from_substrate(data, H, K, L, lattice_file, material,
                             dQ=0.3, peak_width=0.05, threshold=None,
                             q_with_2pi=True, hkl_max=6, mod_tol=0.04,
                             ang_tol=2.0, inlier_tol=0.05, ortho=True,
                             out_x=None, out_y=None, out_z=None,
                             normal=None, verbose=True):
    """Full pipeline: find peaks in the volume, then index them against a
    substrate cell from the lattice file to get U.

    Pass ``normal`` (e.g. [0,0,1]) to pin U reproducibly from a single
    substrate surface normal: it becomes out_z and two in-plane directions
    are chosen orthogonal to it (see ``axes_from_normal``). Explicit
    out_x/out_y/out_z take precedence if all three are given.

    Returns the same dict as index_against_lattice (plus 'peaks'), or None.
    """
    lattice = load_lattice(lattice_file, material)
    if verbose:
        print(f"{material} lattice: a,b,c={lattice[:3]} "
              f"alpha,beta,gamma={lattice[3:]}")
    if normal is not None and not all(v is not None for v in (out_x, out_y, out_z)):
        Bstar = reciprocal_matrix(*lattice, with_2pi=q_with_2pi)
        out_x, out_y, out_z = axes_from_normal(Bstar, normal)
        if verbose:
            print(f"normal {np.asarray(normal, float).tolist()} -> out_z; "
                  f"in-plane out_x={out_x.tolist()}, out_y={out_y.tolist()}")
    thr = threshold if threshold is not None else np.nanmax(data) / 25
    peaks, _ = find_peaks_in_RSM(data, H, K, L, dQ=dQ, threshold=thr,
                                 peak_width=peak_width)
    if verbose:
        print(f"{len(peaks)} peaks found.")
    if len(peaks) < 2:
        return None
    result = index_against_lattice(
        np.array(peaks), lattice, q_with_2pi=q_with_2pi, hkl_max=hkl_max,
        mod_tol=mod_tol, ang_tol=ang_tol, inlier_tol=inlier_tol,
        ortho=ortho, out_x=out_x, out_y=out_y, out_z=out_z, verbose=verbose)
    if result is not None:
        result['peaks'] = np.array(peaks)
    return result


def save_U_matrix(U, filepath):
    np.savetxt(filepath, U)


def load_U_matrix(filepath):
    return np.loadtxt(filepath)


def rotate_U(U, axis, angle_deg):
    """Compose U with a rotation: returns U @ R_axis(angle_deg).

    Use to rotate the viewing frame after U is found, e.g.
    rotate_U(U, 'z', 90). Replaces rotate_Uz / rotate_Uy (which had
    hardcoded angles and a missing Rotation import).
    """
    R = Rotation.from_euler(axis, np.radians(angle_deg)).as_matrix()
    return U @ R


def rotate_Uz(U0, angle_deg=90):
    """Legacy wrapper: U0 rotated about z (default 90 deg)."""
    return rotate_U(U0, 'z', angle_deg)


def rotate_Uy(U0, angle_deg=45):
    """Legacy wrapper: U0 rotated about y (default 45 deg)."""
    return rotate_U(U0, 'y', angle_deg)


def view_axes_transform(Vin1, Vin2, Vout, U0):
    """U for viewing the data along custom axes.

    The new x, y, z axes are the (normalized) directions Vin1, Vin2, Vout
    expressed in the frame of U0. Returns inv([Vin1; Vin2; Vout]) @ U0.

    Replaces the notebook cell where Vout was accidentally overwritten
    with a rotated copy of Vin2, making the matrix singular.
    """
    rows = [np.asarray(v, dtype=float) for v in (Vin1, Vin2, Vout)]
    rows = [v / np.linalg.norm(v) for v in rows]
    T = np.array(rows)
    if abs(np.linalg.det(T)) < 1e-6:
        raise ValueError("Vin1, Vin2, Vout are (nearly) coplanar -- "
                         "they must span 3D space")
    return np.linalg.inv(T) @ U0


# ======================================================================
# Resampling (interpolation onto transformed grids)
# ======================================================================

def transform_slab(data, H, K, L, U, Hi, Ki, Li, order=1):
    """Resample onto the grid (Hi, Ki, Li) in the U-aligned frame.

    The value at new point x is data interpolated at U @ x -- the same
    convention as main_transformation, implemented with
    scipy.ndimage.map_coordinates on index coordinates (faster than
    RegularGridInterpolator and with no separate interpolator object to
    rebuild in every cell). order=1 trilinear, order=0 nearest. Outside
    points are NaN.

    Works for full volumes and for thin slabs (e.g. Li of length 3 around
    one L value) -- the slab pattern keeps slice plots cheap.
    """
    U = np.asarray(U, dtype=float)
    Hg, Kg, Lg = np.meshgrid(Hi, Ki, Li, indexing='ij')
    pts = U @ np.stack((Hg.ravel(), Kg.ravel(), Lg.ravel()))
    ih = (pts[0] - H[0]) / (H[1] - H[0])
    ik = (pts[1] - K[0]) / (K[1] - K[0])
    il = (pts[2] - L[0]) / (L[1] - L[0])
    out = map_coordinates(np.asarray(data, dtype=np.float32),
                          [ih, ik, il], order=order,
                          mode='constant', cval=np.nan)
    return out.reshape(len(Hi), len(Ki), len(Li))


def main_transformation(interpolator, Hi, Ki, Li, U):
    """Legacy resampling via a prebuilt RegularGridInterpolator.

    Kept for compatibility; transform_slab gives identical values without
    the interpolator object.
    """
    Hg, Kg, Lg = np.meshgrid(Hi, Ki, Li, indexing='ij')
    coords = np.stack((Hg.ravel(), Kg.ravel(), Lg.ravel()), axis=-1)
    transformed = coords @ np.asarray(U, dtype=float).T
    out = interpolator(transformed)
    return out.reshape(Hi.size, Ki.size, Li.size)


# ======================================================================
# Plotting
# ======================================================================

def plot_slice(intensity, H, K, L, slice_type, slice_value, ax,
               dH=5, dK=5, dL=5, vmin=1.4, vmax=2.5, cmap='inferno',
               units='$\\AA^{-1}$'):
    """Slab of an (untransformed) volume summed over one axis.

    slice_type 'H'/'K'/'L'; the slab half-width is dH/dK/dL bins.
    """
    axes = {'H': (0, H, dH, (K, L), ('K', 'L')),
            'K': (1, K, dK, (H, L), ('H', 'L')),
            'L': (2, L, dL, (H, K), ('H', 'K'))}
    if slice_type not in axes:
        raise ValueError("slice_type must be 'H', 'K' or 'L'")
    axis, grid, dwin, (xg, yg), (xn, yn) = axes[slice_type]

    i = fi(grid, slice_value)
    sl = [slice(None)] * 3
    sl[axis] = slice(max(i - dwin, 0), i + dwin + 1)
    img = intensity[tuple(sl)].sum(axis=axis)

    im = ax.imshow(np.log10(img.T + 1), origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f'{xn}, ({units})')
    ax.set_ylabel(f'{yn}, ({units})')
    ax.set_title(f"{slice_type} = {grid[i]:.2f}")
    return im


def plot_transformed_slice(data, H, K, L, U, slice_type, value, dQ=0.05,
                           n=1000, nslab=3, vmin=0.5, vmax=1.5,
                           cmap='inferno', order=1, units='$\\AA^{-1}$',
                           ax=None, xlim=None, ylim=None):
    """Slice of the U-aligned volume at slice_type = value (one call).

    Interpolates only a thin slab (n x n x nslab) around the requested
    plane -- the efficient pattern from the notebook -- and averages over
    the slab thickness 2*dQ. Replaces the six duplicated
    interpolator+main_transformation+imshow cells.
    """
    full = {'H': (H, K, L), 'K': (K, H, L), 'L': (L, H, K)}
    if slice_type not in full:
        raise ValueError("slice_type must be 'H', 'K' or 'L'")
    slab_axis_grid, xg_full, yg_full = full[slice_type]

    slab = np.linspace(value - dQ, value + dQ, nslab, dtype=np.float32)
    xg = np.linspace(xg_full[0], xg_full[-1], n, dtype=np.float32)
    yg = np.linspace(yg_full[0], yg_full[-1], n, dtype=np.float32)

    if slice_type == 'H':
        vol = transform_slab(data, H, K, L, U, slab, xg, yg, order=order)
        img = np.nanmean(vol, axis=0)         # (K, L)
        xn, yn = 'K', 'L'
    elif slice_type == 'K':
        vol = transform_slab(data, H, K, L, U, xg, slab, yg, order=order)
        img = np.nanmean(vol, axis=1)         # (H, L)
        xn, yn = 'H', 'L'
    else:
        vol = transform_slab(data, H, K, L, U, xg, yg, slab, order=order)
        img = np.nanmean(vol, axis=2)         # (H, K)
        xn, yn = 'H', 'K'

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.log10(img.T + 1), origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f'{xn}, ({units})')
    ax.set_ylabel(f'{yn}, ({units})')
    ax.set_title(f"Slice at {slice_type} = {value:.2f} "
                 f"with a width of 2x{dQ:.2f}")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return img, (xg, yg), im


def plot_projections(data, H, K, L, height=4e5):
    """1D projections onto H, K, L with detected peaks marked.

    Returns (H_peaks, K_peaks, L_peaks). NaNs in the volume (outside the
    transformed region) are ignored.
    """
    from matplotlib.ticker import ScalarFormatter
    from scipy.signal import find_peaks

    I_H = np.nansum(data, axis=(1, 2))
    I_K = np.nansum(data, axis=(0, 2))
    I_L = np.nansum(data, axis=(0, 1))

    fig, ax = plt.subplots(1, 3, figsize=[10, 3])
    out = []
    for a, grid, I in zip(ax, (H, K, L), (I_H, I_K, I_L)):
        pk, _ = find_peaks(I, height=height)
        a.plot(grid, I)
        a.plot(grid[pk], I[pk], 'o')
        a.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        a.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        print(grid[pk])
        out.append(grid[pk])
    fig.tight_layout()
    return tuple(out)


def line_cut(image, point1, point2, dw=2, xaxis=None, yaxis=None,
             scale=1.0):
    """Width-averaged intensity profile along a line in a 2D slice.

    image      : 2D array indexed [row, col]
    point1/2   : (row, col) endpoints in pixel indices
    dw         : perpendicular averaging width in pixels
    xaxis/yaxis: optional col/row coordinate grids; distance is returned in
                 those units if given, else in pixels
    scale      : extra factor on the distance (use 1/sqrt(2) to reproduce
                 the old r.s.u. convention for [101]-type diagonals)

    Returns (distance, values).
    """
    p1 = np.asarray(point1, dtype=float)
    p2 = np.asarray(point2, dtype=float)
    length = int(np.hypot(*(p2 - p1)))
    rows = np.linspace(p1[0], p2[0], length)
    cols = np.linspace(p1[1], p2[1], length)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    offs = np.linspace(-dw / 2, dw / 2, max(dw, 2))
    values = np.empty(length)
    for i in range(length):
        seg = map_coordinates(
            image,
            [rows[i] - offs * np.sin(angle), cols[i] + offs * np.cos(angle)],
            order=1, mode='nearest')
        values[i] = seg.mean()

    if xaxis is not None and yaxis is not None:
        total = np.hypot(yaxis[int(p2[0])] - yaxis[int(p1[0])],
                         xaxis[int(p2[1])] - xaxis[int(p1[1])])
    else:
        total = np.hypot(*(p2 - p1))
    return total * np.linspace(0, 1, length) * scale, values


def export_vtk(volume, filepath, name="values"):
    """Export a 3D volume to a legacy VTK file (for ParaView etc.).

    Pass the volume you actually want (e.g. the transform_slab output of a
    zoom region) -- the old notebook cell exported the full untransformed
    array under a 'zoom' filename.
    """
    from pyvtk import VtkData, StructuredPoints, PointData, Scalars
    vtk_data = VtkData(StructuredPoints(volume.shape),
                       PointData(Scalars(np.nan_to_num(volume).flatten(),
                                         name=name)))
    vtk_data.tofile(filepath)
    print(f"wrote {filepath}")
