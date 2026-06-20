# hklconv — 3D Reciprocal Space Map Reconstruction (QM2/CHESS)

Converts stacks of area-detector images (CBF) from phi and theta scans into a
3D reciprocal-space map, either in raw lab-frame Q or indexed (H, K, L) using
a UB matrix. Output is a NeXus (`.nxs`) file on a regular grid.

## Quick start

```bash
# 1. Compile the C library -- ON THE MACHINE THAT WILL RUN THE JOB.
#    The .so is CPU-specific; a binary built elsewhere can crash with
#    "Illegal instruction (core dumped)". No root needed; the library is
#    written into this folder, next to hklBen.py, which is where it is
#    loaded from.
cd HKL_Convert
gcc -O3 -march=native -funroll-loops -fopenmp -fPIC -shared \
    -o libhklBen.so hklBen.c
#    On heterogeneous clusters, replace -march=native with the oldest
#    CPU generation you will run on, e.g. -march=sandybridge.

# 2. Sanity-check the build (~30 s; needs a few GB of RAM):
python test_equivalence.py
#    Expect "norm identical: True" and per-frame timings.

# 3. Write a config file (copy config_example.txt and edit the paths;
#    every key in the example must be present -- see "Config file format").

# 4. Run. Threads are set via OMP_NUM_THREADS (use PHYSICAL cores,
#    not hyperthreads; the kernel is memory-bound):
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
python autoRSM.py config.txt

# For long jobs over ssh:
screen -dmS rsm bash -c 'OMP_NUM_THREADS=12 python autoRSM.py config.txt \
    > rsm.log 2>&1'
```

Output lands in `{Output Directory}/indexed_objects/` (with UB) or
`{Output Directory}/transformed_objects/` (without UB); the exact filename
is printed at startup. Progress is shown per scan with a tqdm bar — the
it/s rate is the per-frame throughput.

## Contents

| File | Purpose |
|---|---|
| `autoRSM.py` | Main script. Reads a config file, loops over scans/frames, accumulates the 3D histogram, writes the NeXus output. |
| `hklBen.py` | ctypes wrapper around `libhklBen.so`, plus the goniometer rotation-matrix construction (transposed Busing–Levy convention). |
| `hklBen.c` | C/OpenMP kernels. `hklhist` is the fused fast path; `hist`, `hist2`, `histarb`, `benhkl`, `calchkl` are kept for older notebooks. |
| `Makefile` | Builds `libhklBen.so`. |
| `config_example.txt` | Annotated config template. |
| `test_equivalence.py` | Checks the fused kernel against the legacy path on synthetic data. |

## Requirements

Python ≥ 3.8 with `numpy`, `pyFAI`, `fabio`, `tqdm`, `nexusformat`,
`spec2nexus`. A C compiler with OpenMP (gcc).

## Build details

`make` is equivalent to the gcc line in the quick start (edit `CFLAGS` in
the Makefile to change the architecture flag). The library is loaded
relative to `hklBen.py` itself, not the working directory, so the script
can be run from anywhere — but the freshly built `libhklBen.so` must sit
in the same folder as `hklBen.py`. If Python reports
`AttributeError: ... hklhist`, an old library without the fused kernel is
being picked up — rebuild in this folder.

## Config file format

Plain text, one `Key: value` per line. Blank lines and lines starting with
`#` are ignored. Unknown keys raise an error (typos won't silently vanish).

```
PONI File: /path/to/geometry.poni
Material: V2O3
Sample Name: NW220910B
Scan Number: 12
Scan List: [12, 13, 14]
Theta Scan Number: None
Theta Scan List: []
Temperature: 150
Mask File: /path/to/mask.edf
Specfile: /path/to/specfile
Temperature Directory: /path/to/T150/
Image Directory: /path/to/images/
Output Directory: /path/to/output/
# Optional -- omit both lines for an unindexed (lab-frame Q) map:
UB: [[0.7199, -0.6941, 0.0002], [0.6941, 0.7198, -0.0121], [0.0084, 0.0088, 0.9999]]
Substrate Lattice Params: (4.95, 4.95, 14.0)
```

Notes:

- **`Scan List`** — phi scans to merge into one volume. Images for scan *n*
  are expected in `{Temperature Directory}/{Material}_{n:03d}/`.
- **`Theta Scan Number` / `Theta Scan List`** — set the number to anything
  other than `None` to also merge the listed theta scans.
- **`UB` and `Substrate Lattice Params`** must be given together. With UB
  the grid spans ±Qmax·a/2π etc. in (H, K, L) and output goes to
  `{Output Directory}/indexed_objects/`. Without UB the volume is binned in
  lab-frame Q (Å⁻¹/2π) and goes to `{Output Directory}/transformed_objects/`.
- **`Scan Number`** and **`Temperature`** are informational (printed only).

## Output

`{Material}_{Sample Name}_scans_{...}_out.nxs`, an `NXdata` group with:

- `counts` — `data.clip(0) / norm.clip(0.9)` on the (H, K, L) grid
- `norm` — number of detector pixels contributing to each voxel.
  Voxels with `norm == 0` were never measured; use this to distinguish
  empty regions from genuinely zero intensity.

If the file exists, `_more` is appended rather than overwriting.

## Processing model

Per frame:

1. CBF image is read (prefetched on a background thread while the previous
   frame is histogrammed, hiding read/decompression latency).
2. The static mask is applied (masked pixels are set negative and skipped in
   C). For theta scans, rows below the sample horizon (2θ < η) are
   additionally masked, recomputed for each frame (QM2-specific cut).
3. Intensities are normalized by ion chamber (`ic2`, scan-averaged to 1) and
   per-pixel solid angle. Frames with `ic2 ≤ 0` are skipped.
4. The fused C kernel `hklhist` rotates the precomputed detector q-vectors
   into (H, K, L) with the per-frame goniometer matrix
   `M = U⁻¹ᵀ·Φ·Χ·Η` and bins them by binary search, accumulating intensity
   and hit count with OpenMP atomics.

The detector q-array, solid-angle correction, and 2θ map depend only on the
pyFAI geometry and are computed once per run.

## Differences from autoRSM_masked.py (behavioral)

Intentional fixes — results can differ from the old script where these
applied:

1. **Theta-scan horizon mask actually works now.** The old code computed the
   cutoff row index on the *flattened* 2θ array, producing a flat pixel
   index far beyond the number of detector rows; the slice
   `currentmask[th_index:, :]` was empty and the progressive mask did
   nothing. It is now computed on the 2D array, and the mask is reset per
   frame instead of accumulating across frames.
2. **No silent UB fallback.** Previously, a config with `UB` but without
   `Substrate Lattice Params` silently discarded the UB and produced an
   unindexed map. This is now a hard error.
3. **Beam-center pixel indices** use matched axes (`poni1/pixel1`,
   `poni2/pixel2`); the old code crossed them (no effect for square pixels).
4. Output directories are created if missing; malformed or unknown config
   lines raise errors instead of crashing or being ignored.

Unchanged: the q-vector convention, the rotation arithmetic (verified
bit-identical to `correct_HKL2`), bin-edge conventions (strict
inequalities at the outer edges), the `data.clip(0)/norm.clip(0.9)`
normalization, and the output naming scheme.

## Performance notes

- Measured end-to-end: ~3x faster than autoRSM_masked.py on a 12-core
  dual-socket Sandy Bridge node (Pilatus 6M data, 1000^3 grid).
- The fused kernel eliminates roughly six full passes over N-sized arrays
  per frame (HR/KR/LR allocation, `vstack`, two transposes, and float64
  intermediate for counts) and the useless `errors += 0` atomic per pixel.
- For long scans the wall-clock bottleneck is often CBF reading and
  decompression; the background prefetch overlaps it with histogramming.
- Compile with `-O3 -march=native -fopenmp` (the Makefile default). On a
  NUMA server, pin threads (`OMP_PROC_BIND=close OMP_PLACES=cores`).
- Going further than this would require per-thread partial histograms
  (memory-prohibitive at 1000³ ≈ 4 GB per array per thread) or processing
  multiple frames per C call; neither is worth it until frame I/O is no
  longer the limiting factor.

## Validation

`python test_equivalence.py` compares the fused path against the legacy
`correct_HKL2` + `HIST2` path on 2 M synthetic pixels: identical bin
assignments (norm arrays equal), intensity sums equal to float32 rounding
(atomics change the summation order).
