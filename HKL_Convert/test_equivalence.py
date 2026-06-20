"""Verify the fused hklhist kernel against the legacy path and benchmark it.

Legacy path = the exact arithmetic of the old autoRSM_masked.py:
    IN = correct_HKL2_old(q, eta, chi, phi, UB)   # pure numpy, old code
    HIST2(IN, counts, 1.0, H, K, L, data, norm, errors)

New path:
    M = rotation_matrix(eta, chi, phi, UB)
    HKLHIST(q, M, counts, H, K, L, data, norm)
"""
import time
import numpy as np
import hklBen


def correct_HKL2_old(q, theta, chi, phi, UB):
    """Verbatim arithmetic of the original correct_HKL2."""
    eta = np.deg2rad(theta)
    chi = np.deg2rad(chi)
    phi = np.deg2rad(phi)
    UB = UB.T / (2 * np.pi)
    UB = UB.reshape(9)
    Umatrix = np.zeros((3, 3))
    Umatrix[0][0] = UB[0]; Umatrix[0][1] = UB[1]; Umatrix[0][2] = UB[2]
    Umatrix[1][0] = UB[3]; Umatrix[1][1] = UB[4]; Umatrix[1][2] = UB[5]
    Umatrix[2][0] = UB[6]; Umatrix[2][1] = UB[7]; Umatrix[2][2] = UB[8]
    Uinv = np.linalg.inv(Umatrix)
    PHI = np.zeros((3, 3)); CHI = np.zeros((3, 3)); ETA = np.zeros((3, 3))
    PHI[0][0] = np.cos(phi); PHI[0][1] = -np.sin(phi)
    PHI[1][0] = np.sin(phi); PHI[1][1] = np.cos(phi); PHI[2][2] = 1.0
    CHI[0][0] = np.cos(chi); CHI[0][2] = -np.sin(chi); CHI[1][1] = 1.0
    CHI[2][0] = np.sin(chi); CHI[2][2] = np.cos(chi)
    ETA[0][0] = np.cos(eta); ETA[0][1] = -np.sin(eta)
    ETA[1][0] = np.sin(eta); ETA[1][1] = np.cos(eta); ETA[2][2] = 1.0
    hklvector = Uinv.T @ PHI @ CHI @ ETA @ q
    IN = np.vstack((hklvector[0], hklvector[1], hklvector[2]))
    return IN.transpose()


rng = np.random.default_rng(42)
N = 2_000_000  # ~ Pilatus 2M-class detector

# synthetic detector q and a plausible UB
q = np.ascontiguousarray(rng.uniform(-0.3, 0.3, (3, N)))
q[2] = np.abs(q[2])
UB = np.array([[7.198694e-01, -6.940937e-01, 1.856282e-04],
               [6.940591e-01,  7.198305e-01, -1.213340e-02],
               [8.361443e-03,  8.831679e-03,  9.999264e-01]]) * 2 * np.pi

counts = rng.uniform(0, 100, N).astype(np.float32)
counts[rng.random(N) < 0.05] = -2.0  # masked pixels

H = np.linspace(-1.5, 1.5, 400)
K = np.linspace(-1.5, 1.5, 400)
L = np.linspace(0.0, 1.5, 400)
nvox = len(H) * len(K) * len(L)

eta, chi, phi = 12.3, -0.4, 37.7

# ---- legacy path ----
data1 = np.zeros(nvox, dtype=np.float32)
norm1 = np.zeros(nvox, dtype=np.float32)
err1 = np.zeros(nvox, dtype=np.float32)
t0 = time.perf_counter()
IN = correct_HKL2_old(q, eta, chi, phi, UB)
hklBen.HIST2(IN, counts, 1.0, H, K, L, data1, norm1, err1)
t_legacy = time.perf_counter() - t0

# ---- new fused path ----
data2 = np.zeros(nvox, dtype=np.float32)
norm2 = np.zeros(nvox, dtype=np.float32)
t0 = time.perf_counter()
M = hklBen.rotation_matrix(eta, chi, phi, UB)
hklBen.HKLHIST(q, M, counts, H, K, L, data2, norm2)
t_fused = time.perf_counter() - t0

print(f"norm identical:      {np.array_equal(norm1, norm2)}")
print(f"hits binned:         {int(norm1.sum())} vs {int(norm2.sum())}")
print(f"max |data diff|:     {np.max(np.abs(data1 - data2)):.3e}")
print(f"sum data:            {data1.sum():.6e} vs {data2.sum():.6e}")
print(f"legacy per frame:    {t_legacy*1e3:7.1f} ms")
print(f"fused  per frame:    {t_fused*1e3:7.1f} ms")
print(f"speedup:             {t_legacy/t_fused:.1f}x")

# repeat fused timing warm (allocator warm, M trivial)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    M = hklBen.rotation_matrix(eta, chi, phi, UB)
    hklBen.HKLHIST(q, M, counts, H, K, L, data2, norm2)
    times.append(time.perf_counter() - t0)
print(f"fused warm (best):   {min(times)*1e3:7.1f} ms")

# legacy correct_HKL2 wrapper in new hklBen must match the old arithmetic too
IN_new = hklBen.correct_HKL2(q, eta, chi, phi, UB)
print(f"correct_HKL2 compat: max diff {np.max(np.abs(IN - IN_new)):.3e}")
