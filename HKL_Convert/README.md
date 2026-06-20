# HKL_Convert

Data-acquisition / reciprocal-space conversion stack (the fused C kernel and
its Python driver). This is the layer that turns raw detector frames into the
binned H/K/L volumes that `rsm_viewer.py` and `rsm_monitor.py` consume.

These files live on a separate (working) server and are copied in here
unchanged. Expected contents:

- `autoRSM.py`            -- driver: config parsing, grid building, spec reading
- `hklBen.py`            -- Python wrapper around the C kernel
- `hklBen.c`             -- fused detector-q -> HKL histogram kernel
- `libhklBen.so`         -- compiled kernel (built via the Makefile)
- `Makefile`             -- builds `libhklBen.so` from `hklBen.c`

`rsm_monitor.py` invokes the production `autoRSM.py` via a configured path
(see the `autoRSM` option in its defaults), not by importing this copy, so the
monitor runs without this folder. The acquisition unit tests
(`test_wrapper3.py`) do import `autoRSM`; run them from here only after these
files are in place, and ensure this folder is on `PYTHONPATH`.
