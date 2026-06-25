#!/usr/bin/env python3
"""
rsm_output_status.py

Given one autoRSM config (log) file, print the .nxs output path autoRSM
would produce and whether it already exists. Used by watch_and_process.sh
to skip datasets that are already done, independently of the
processed-commands ledger.

The output-naming logic here MUST match autoRSM.output_filename:
    {Material}_{Sample Name}_scans_{phi+theta scans}_{Output Tag}.nxs
when an Output Tag is set, else _full.nxs (unindexed lab-frame Q) or
_out.nxs (bare indexed), in indexed_objects/ (if a UB is present) or
transformed_objects/ (if not).

Exit status:
    0  output exists       (prints: DONE <path>)
    1  output missing       (prints: TODO <path>)
    2  could not parse file (prints: ERROR <reason>)

Usage:
    python3 rsm_output_status.py /path/to/log.txt
"""

import ast
import os
import sys


def parse_config(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or ': ' not in line:
                continue
            key, value = line.split(': ', 1)
            cfg[key] = value
    return cfg


def output_path(cfg):
    material = cfg['Material']
    sample = cfg['Sample Name']
    scan_list = ast.literal_eval(cfg.get('Scan List', '[]'))
    theta_list = ast.literal_eval(cfg.get('Theta Scan List', '[]'))
    scans = list(scan_list) + list(theta_list)
    scan_str = '_'.join(str(s) for s in scans)
    base = f"{material}_{sample}_scans_{scan_str}"
    tag = cfg.get('Output Tag', '').strip()
    if tag:
        fname = f"{base}_{tag}.nxs"
    else:
        fname = f"{base}_{'out' if 'UB' in cfg else 'full'}.nxs"

    out_dir = cfg['Output Directory']
    subdir = 'indexed_objects' if 'UB' in cfg else 'transformed_objects'
    return os.path.join(out_dir, subdir, fname)


def main(argv):
    if len(argv) != 2:
        print("ERROR usage: rsm_output_status.py <log.txt>")
        return 2
    try:
        cfg = parse_config(argv[1])
        path = output_path(cfg)
    except (KeyError, ValueError, SyntaxError, FileNotFoundError) as e:
        print(f"ERROR {type(e).__name__}: {e}")
        return 2

    # autoRSM appends _more.nxs if the base name exists, so the dataset is
    # "done" if the base output OR any _more variant is present.
    candidates = [path]
    stem = path[:-4]
    while True:
        stem = stem + '_more'
        candidates.append(stem + '.nxs')
        if len(candidates) > 50:
            break

    for c in candidates:
        if os.path.exists(c):
            print(f"DONE {c}")
            return 0
    print(f"TODO {path}")
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
