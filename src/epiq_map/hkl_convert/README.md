# hklconv — 3D reciprocal-space reconstruction

This package converts area-detector image stacks into a regular reciprocal-
space map, either in lab-frame Q or indexed (H, K, L) coordinates.

## Build and install

Install a published platform wheel:

```bash
python -m pip install EpiQ-Map
```

Or build a checkout with the standard Python frontend:

```bash
python -m pip install --upgrade build
python -m build
python -m pip install dist/*.whl
```

The PEP 517 build compiles `hkl_ben.c` into the wheel. Linux and Windows
enable OpenMP; macOS uses the same source in serial mode because Apple Clang
does not bundle an OpenMP runtime. Release wheels are built with
`cibuildwheel` for Linux, Windows, Intel macOS, and Apple Silicon.

## Run

Copy `config_example.txt`, edit the data paths, then run:

```bash
export OMP_NUM_THREADS=12
python -m epiq_map.hkl_convert.auto_rsm config.txt
```

The monitor invokes this packaged module automatically. Use
`python -m epiq_map.hkl_convert.auto_rsm_rods config.txt` for a direct CTR
multi-rod reconstruction.

## Contents

| File | Purpose |
|---|---|
| `auto_rsm.py` | Main reconstruction module. |
| `auto_rsm_rods.py` | CTR multi-rod reconstruction module. |
| `hkl_ben.py` | ctypes wrapper around the packaged native library. |
| `hkl_ben.c` | Fused HKL transform and histogram kernels. |
| `config_example.txt` | Annotated reconstruction configuration. |

The native equivalence check is in
`tests/integration/test_hkl_equivalence.py`.
