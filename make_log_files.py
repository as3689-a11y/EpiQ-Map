#!/usr/bin/env python3
"""
make_log_files.py

Walks a beamtime's raw image directory tree, finds scan directories with
image data, classifies each scan (theta vs. other) by reading the SPEC
file's scan header, writes a per-scan log file describing the scan, and
builds a command list to run autoRSM_labframe_Lorentz.py on each.

Directory depth assumption: scan directories are expected at a fixed
depth under BASE_DIR, with material, sample_name, temperature, and
scan_xxx as the last four path components (indices 7-10 when the
absolute path is split on "/"). If you point this at a beamtime with a
different directory layout, the path indices below will silently
extract the wrong fields, so check BASE_DIR's structure first (or pass
--scan-depth / --material-idx / etc. to override).

Usage:
    python3 make_log_files.py \\
        --base-dir /nfs/chess/id4b/2024-3/singer-4058-b/raw6M \\
        --spec-dir /nfs/chess/id4b/2024-3/singer-4058-b/ \\
        --output-dir /nfs/chess/id4baux/2024-3/singer-4058-b/output/ \\
        --poni-file /nfs/chess/id4baux/2024-3/singer-4058-b/calibrations/CeO2_15keV.poni \\
        --mask-file /nfs/chess/id4baux/2024-3/singer-4058-b/calibrations/mask.edf

Run `python3 make_log_files.py --help` for all options.
"""

import argparse
import os
import re
import sys


# --- Scan-type classification (from the SPEC file) -------------------
#
# Each scan's type (theta scan, phi scan, etc.) is read directly from
# the SPEC file's scan header line, rather than guessed from a
# file-count threshold. A SPEC scan header looks like:
#
#     #S 12  ascan  phi 0 360  3600 1
#
# 12 is the scan number, ascan is the scan macro, and phi is the motor
# being scanned. We match the scan number from the directory name
# against the "#S <num>" in the spec file to find which motor was used.
#
# Convention: "th" scans go into theta_list; every other motor ("tth",
# "phi", etc.) goes into scan_list.

_SCAN_HEADER_RE = re.compile(r"^#S\s+(\d+)\s+\S+\s+(\S+)", re.MULTILINE)
_spec_cache = {}  # avoid re-reading the same spec file for every scan


def get_scan_motor(spec_path, scan_number):
    """Read a SPEC file's '#S <num> <macro> <motor> ...' header lines and
    return the motor name used for the given scan number. Returns None if
    the spec file is missing or the scan number isn't found in it."""
    if spec_path not in _spec_cache:
        try:
            with open(spec_path, "r", errors="replace") as f:
                _spec_cache[spec_path] = f.read()
        except FileNotFoundError:
            print(f"  Spec file not found: {spec_path}")
            _spec_cache[spec_path] = ""

    spec_text = _spec_cache[spec_path]
    for match in _SCAN_HEADER_RE.finditer(spec_text):
        found_scan_num, motor = match.groups()
        if int(found_scan_num) == scan_number:
            return motor
    return None


# --- Log file content and writing --------------------------------------

def generate_log_content(image_dir, temperature, material, sample_name,
                          scan_number, scan_list, theta_list, cfg):
    return f"""PONI File: {cfg.poni_file}
Material: {material}
Sample Name: {sample_name}
Scan Number: {scan_number}
Scan List: {scan_list}
Theta Scan List: {theta_list}
Temperature: {temperature}
Mask File: {cfg.mask_file}
Specfile: {cfg.spec_dir}/{material}
Temperature Directory: {os.path.dirname(image_dir)}/
Image Directory: {image_dir}/
Output Directory: {cfg.output_dir}
"""


def write_log_file(log_content, root, log_dir):
    """Write log_content to a file named after root's path, unless that
    file already exists (in which case it's left untouched and its path
    is returned as-is)."""
    log_file_name = os.path.join(log_dir, root.replace("/", "_") + ".txt")
    if os.path.exists(log_file_name):
        print(f"  Log file already exists, skipping: {log_file_name}")
    else:
        with open(log_file_name, "w") as f:
            f.write(log_content)
        print(f"  Log file written: {log_file_name}")
    return log_file_name


# --- Main traversal ------------------------------------------------------

def traverse_and_write_logs(cfg):
    """Walk cfg.base_dir, and at every directory matching cfg.scan_depth,
    extract material / sample / temperature / scan number from the path,
    count image files, and -- if there are enough to count as a real
    scan -- look up the scan's motor in the SPEC file and write a per-scan
    log file (a self-contained autoRSM config you can also rerun by hand).
    Returns the list of log files written."""
    log_count = 0
    written = []

    for root, dirs, files in os.walk(cfg.base_dir):
        path_parts = root.split(os.sep)
        if len(path_parts) != cfg.scan_depth:
            continue

        material = path_parts[cfg.material_idx]
        sample_name = path_parts[cfg.sample_idx]
        temperature = path_parts[cfg.temperature_idx]
        try:
            scan_number = int(path_parts[cfg.scan_dir_idx].split("_")[-1])
        except ValueError:
            print(f"Skipping {root}: can't parse scan number from directory name")
            continue

        num_files = sum(1 for entry in os.scandir(root) if entry.is_file())
        if num_files <= cfg.min_files_for_scan:
            continue

        print(f"Processing {root} ({num_files} files)")

        spec_path = os.path.join(cfg.spec_dir, material)
        motor = get_scan_motor(spec_path, scan_number)

        if motor is None:
            print(f"  Scan {scan_number} not found in {spec_path}; skipping")
            continue

        if motor == "th":
            scan_list, theta_list = [], [scan_number]
        else:
            scan_list, theta_list = [scan_number], []
        print(f"  Motor: {motor!r} -> scan_list={scan_list}, theta_list={theta_list}")

        log_content = generate_log_content(
            root, temperature, material, sample_name,
            scan_number, scan_list, theta_list, cfg,
        )
        log_file_name = write_log_file(log_content, root, cfg.log_dir)

        written.append(log_file_name)
        log_count += 1
        if log_count >= cfg.max_log_files:
            break

    print(f"\nTotal logs processed: {log_count}")
    return written


# --- CLI -------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate per-scan log files and a command list for autoRSM processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Directory layout
    p.add_argument("--base-dir", required=True,
                    help="Raw image base directory to walk (e.g. .../raw6M)")
    p.add_argument("--spec-dir", required=True,
                    help="Directory containing SPEC files, one per material")
    p.add_argument("--output-dir", required=True,
                    help="Output directory; logs are written to <output-dir>/logs")
    p.add_argument("--poni-file", required=True, help="Path to the .poni calibration file")
    p.add_argument("--mask-file", required=True, help="Path to the mask .edf file")

    # Path-depth indices. Defaults match the standard CHESS layout:
    #   .../raw6M/<material>/<sample_name>/<temperature>/<scan_xxx>
    p.add_argument("--scan-depth", type=int, default=11,
                    help="len(path.split('/')) expected at a valid scan directory")
    p.add_argument("--material-idx", type=int, default=7, help="Path index of the material name")
    p.add_argument("--sample-idx", type=int, default=8, help="Path index of the sample name")
    p.add_argument("--temperature-idx", type=int, default=9, help="Path index of the temperature")
    p.add_argument("--scan-dir-idx", type=int, default=10, help="Path index of the scan_xxx directory")

    # Scan inclusion / processing
    p.add_argument("--min-files-for-scan", type=int, default=1000,
                    help="A directory only counts as a real scan if it has more than this many image files")
    p.add_argument("--max-log-files", type=int, default=1000,
                    help="Safety limit on number of logs written in one run")

    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)
    cfg.log_dir = os.path.join(cfg.output_dir, "logs")

    os.makedirs(cfg.log_dir, exist_ok=True)

    written = traverse_and_write_logs(cfg)

    print(f"Per-scan logs written to {cfg.log_dir} ({len(written)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
