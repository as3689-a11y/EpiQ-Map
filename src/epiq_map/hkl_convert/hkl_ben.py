"""
hkl_ben.py -- ctypes wrapper for the platform-native hklBen library.

The library is loaded relative to THIS file, not the working directory,
so scripts using this module can run from anywhere (this removes the old
requirement to os.chdir into the package directory before importing).

Fast path used by autoRSM.py:
    rotation_matrix(eta, chi, phi, UB)  -- 3x3 lab->HKL rotation per frame
    HKLHIST(q, M, counts, H, K, L, vol, norm)  -- fused rotate + histogram

Legacy functions (HIST, HIST2, HISTARB, correct_HKL, correct_HKL2,
Calc_HKL, ben_HKL) are kept so that existing notebooks continue to work.
"""

import os
import numpy as np
import ctypes as ct
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

_module_dir = Path(__file__).resolve().parent


def _load_kernel():
    """Load the platform-specific native kernel included in the wheel."""
    candidates = []
    for suffix in EXTENSION_SUFFIXES:
        candidates.extend(_module_dir.glob(f"libhklBen*{suffix}"))
    candidates.extend(_module_dir.glob("libhklBen*.so"))
    candidates.extend(_module_dir.glob("libhklBen*.dylib"))
    candidates.extend(_module_dir.glob("libhklBen*.dll"))
    if not candidates:
        raise ImportError(
            "The hklBen native kernel is missing. Reinstall EpiQ-Map from a "
            "platform wheel or build it locally with `python -m build`."
        )
    return ct.CDLL(str(candidates[0]))


_libhkl = _load_kernel()

_d1 = np.ctypeslib.ndpointer(dtype=np.double,  ndim=1, flags='CONTIGUOUS')
_f1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

# ----------------------------------------------------------------------
# C function signatures
# ----------------------------------------------------------------------

# hklhist(q, M, counts, Hbin, Kbin, Lbin, vol, norm, N, hn, kn, ln, mon)
_libhkl.hklhist.argtypes = [_d1, _d1, _f1, _d1, _d1, _d1, _f1, _f1,
                            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float]
_libhkl.hklhist.restype = ct.c_void_p

_libhkl.calchkl.argtypes = [_d1, _d1,
                            ct.c_double, ct.c_double, ct.c_double, ct.c_double,
                            ct.c_double,
                            _d1, _d1, _d1, _d1, ct.c_int]
_libhkl.calchkl.restype = ct.c_void_p

_libhkl.benhkl.argtypes = [_d1, _d1, _d1, _d1, _d1, _d1,
                           ct.c_double,
                           _d1, _d1, _d1, _d1, ct.c_int]
_libhkl.benhkl.restype = ct.c_void_p

# hist/hist2: (Hbin, Kbin, Lbin, HR, KR, LR, counts, vol, norm, errors,
#              N, hn, kn, ln, mon)
_hist_args = [_d1, _d1, _d1, _d1, _d1, _d1, _f1, _f1, _f1, _f1,
              ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float]
_libhkl.hist.argtypes = _hist_args
_libhkl.hist.restype = ct.c_void_p
_libhkl.hist2.argtypes = _hist_args
_libhkl.hist2.restype = ct.c_void_p

_libhkl.histarb.argtypes = [_d1, _d1, _f1, _d1, _d1, _d1, _f1, _f1,
                            ct.c_float, ct.c_float,
                            _f1, _f1, _f1,
                            ct.c_int, ct.c_int, ct.c_int, ct.c_float]
_libhkl.histarb.restype = ct.c_void_p


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def detector_q(poni):
    """Detector q-vectors in the lab frame, in units of 1/Angstrom/(2 pi).

    Depends only on the pyFAI geometry, not on the goniometer angles, so it
    is computed once per dataset and reused for every frame.

    Returns a contiguous (3, N) float64 array with rows (qx, qy, qz).
    """
    Q = poni.qArray() / 10.0 / (2 * np.pi)   # nm^-1 -> A^-1/(2 pi)
    tth = poni.twoThetaArray()
    gam = poni.chiArray()
    N = Q.size
    Q = Q.reshape(N)
    tth = tth.reshape(N)
    gam = gam.reshape(N)

    q = np.empty((3, N))
    q[0] =  Q * np.sin(np.pi / 2 - tth / 2) * np.sin(-gam)
    q[1] = -Q * np.cos(np.pi / 2 - tth / 2)
    q[2] = -Q * np.sin(np.pi / 2 - tth / 2) * np.cos(-gam)
    return np.ascontiguousarray(q)


def rotation_matrix(eta_deg, chi_deg, phi_deg, UB):
    """3x3 matrix M such that (h, k, l) = M . q_lab.

    M = Uinv.T @ PHI @ CHI @ ETA with the transposed Busing-Levy (1967)
    rotation conventions used at QM2 (same arithmetic as correct_HKL2).
    """
    eta = np.deg2rad(eta_deg)
    chi = np.deg2rad(chi_deg)
    phi = np.deg2rad(phi_deg)

    Umatrix = UB.T / (2 * np.pi)
    Uinv = np.linalg.inv(Umatrix)

    PHI = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                    [np.sin(phi),  np.cos(phi), 0.0],
                    [0.0,          0.0,         1.0]])

    CHI = np.array([[np.cos(chi), 0.0, -np.sin(chi)],
                    [0.0,         1.0,  0.0],
                    [np.sin(chi), 0.0,  np.cos(chi)]])

    ETA = np.array([[np.cos(eta), -np.sin(eta), 0.0],
                    [np.sin(eta),  np.cos(eta), 0.0],
                    [0.0,          0.0,         1.0]])

    return np.ascontiguousarray(Uinv.T @ PHI @ CHI @ ETA)


# ----------------------------------------------------------------------
# Fast path
# ----------------------------------------------------------------------

def HKLHIST(q, M, counts, Hbin, Kbin, Lbin, vol, norm, mon=1.0):
    """Fused HKL transform + 3D histogram of one detector frame.

    q       : (3, N) contiguous float64 from detector_q() (frame-independent)
    M       : (3, 3) rotation from rotation_matrix() (one per frame)
    counts  : (N,) contiguous float32 intensities; negative pixels skipped
    H/K/Lbin: sorted 1D float64 bin grids
    vol     : (hn*kn*ln,) float32 accumulator -- intensity
    norm    : (hn*kn*ln,) float32 accumulator -- hit count
    mon     : monitor value; counts are divided by this in C
    """
    N = q.shape[1]
    _libhkl.hklhist(q.ravel(), np.ascontiguousarray(M, dtype=np.double).ravel(),
                    counts, Hbin, Kbin, Lbin, vol, norm,
                    N, len(Hbin), len(Kbin), len(Lbin), mon)


# ----------------------------------------------------------------------
# Legacy API (kept for existing notebooks)
# ----------------------------------------------------------------------

def HIST(IN, counts, mon, Hbin, Kbin, Lbin, vol, norm, errors):
    HR = IN.transpose()[0]
    KR = IN.transpose()[1]
    LR = IN.transpose()[2]
    I = np.asarray(counts, dtype=np.float32)
    _libhkl.hist(Hbin, Kbin, Lbin, HR, KR, LR, I, vol, norm, errors,
                 len(HR), len(Hbin), len(Kbin), len(Lbin), mon)


def HIST2(IN, counts, mon, Hbin, Kbin, Lbin, vol, norm, errors):
    HR = IN.transpose()[0]
    KR = IN.transpose()[1]
    LR = IN.transpose()[2]
    I = np.asarray(counts, dtype=np.float32)
    _libhkl.hist2(Hbin, Kbin, Lbin, HR, KR, LR, I, vol, norm, errors,
                  len(HR), len(Hbin), len(Kbin), len(Lbin), mon)


def HISTARB(IN, counts, mon, Q1bin, Q2bin, phat, vec1, vec2,
            prangemin, prangemax, vol, norm):
    HR = IN.transpose()[0]
    KR = IN.transpose()[1]
    LR = IN.transpose()[2]
    I = np.asarray(counts, dtype=np.float32)
    _libhkl.histarb(Q1bin, Q2bin, phat, HR, KR, LR, vec1, vec2,
                    prangemin, prangemax, I, vol, norm,
                    len(HR), len(Q1bin), len(Q2bin), mon)


def correct_HKL2(q, theta, chi, phi, UB):
    """Rotate precomputed lab-frame q (3, N) into HKL. Returns (N, 3)."""
    M = rotation_matrix(theta, chi, phi, UB)
    hklvector = M @ q
    return np.ascontiguousarray(hklvector.T)


def correct_HKL(Q, tth, gam, theta, chi, phi, WL, UB):
    """Compute lab-frame q from detector angles, then rotate into HKL."""
    N = Q.size
    Q = Q.reshape(N)
    tth = tth.reshape(N)
    gam = gam.reshape(N)

    q = np.empty((3, N))
    q[0] =  Q * np.sin(np.pi / 2 - tth / 2) * np.sin(-gam)
    q[1] = -Q * np.cos(np.pi / 2 - tth / 2)
    q[2] = -Q * np.sin(np.pi / 2 - tth / 2) * np.cos(-gam)
    return correct_HKL2(q, theta, chi, phi, UB)


def Calc_HKL(pol, az, eta, mu, chi, phi, WL, U):
    P = pol.reshape(pol.size)
    A = az.reshape(az.size)
    eta = ct.c_double(eta * np.pi / 180.0)
    mu = ct.c_double(mu * np.pi / 180.0)
    chi = ct.c_double(chi * np.pi / 180.0)
    phi = ct.c_double(phi * np.pi / 180.0)
    WL = ct.c_double(WL)

    U = np.asarray(U / (2 * np.pi)).reshape(9)

    HR = np.empty(len(P), dtype=np.float64)
    KR = np.empty(len(P), dtype=np.float64)
    LR = np.empty(len(P), dtype=np.float64)

    _libhkl.calchkl(P, A, eta, mu, chi, phi, WL, U, HR, KR, LR, len(P))
    return np.vstack((HR, KR, LR)).T


def ben_HKL(Q, tth, gam, theta, chi, phi, WL, UB):
    N = Q.size
    Q = Q.reshape(N)
    tth = tth.reshape(N)
    gam = gam.reshape(N)

    theta_r = np.deg2rad(theta)
    chi_r = np.deg2rad(chi)
    phi_r = np.deg2rad(phi)

    Umatrix = UB.T / (2 * np.pi)
    UBinvT = np.ascontiguousarray(np.linalg.inv(Umatrix).T.reshape(9))

    PHI = np.array([[np.cos(phi_r), -np.sin(phi_r), 0.0],
                    [np.sin(phi_r),  np.cos(phi_r), 0.0],
                    [0.0,            0.0,           1.0]]).reshape(9)
    CHI = np.array([[np.cos(chi_r), 0.0, -np.sin(chi_r)],
                    [0.0,           1.0,  0.0],
                    [np.sin(chi_r), 0.0,  np.cos(chi_r)]]).reshape(9)
    THETA = np.array([[np.cos(theta_r), -np.sin(theta_r), 0.0],
                      [np.sin(theta_r),  np.cos(theta_r), 0.0],
                      [0.0,              0.0,             1.0]]).reshape(9)

    HR = np.empty(N, dtype=np.float64)
    KR = np.empty(N, dtype=np.float64)
    LR = np.empty(N, dtype=np.float64)

    _libhkl.benhkl(Q, tth, gam, THETA, CHI, PHI, WL, UBinvT, HR, KR, LR, N)
    return np.vstack((HR, KR, LR)).T
