#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#
# parse_latency.py — Parse qaic-runner latency profiling output and report statistics
#
# Usage:
#   python parse_latency.py <prof_out_dir>
#   python parse_latency.py <aic-profiling-program-0-latency.txt>
#
# The script reads totalRoundtripTimeUs (end-to-end host-measured latency per inference).
# Despite the "Us" column name, values in the calibrated qaic-runner output are
# in 100ns units — empirically verified: CSV values are 10× the true µs mean
# derived from TotalDuration/InferenceCnt in runner.log.
# UNIT_SCALE=10.0 corrects for this; set to 1.0 if a future SDK writes true µs.
# Output is in milliseconds.
#
# Sanity check: if runner.log exists in the same directory, InferenceCnt and TotalDuration
# are parsed and compared against the CSV mean. If they differ by more than SANITY_RATIO_MAX,
# the fallback mean (TotalDuration / InferenceCnt) is used and a warning is printed.
# TotalDuration in runner.log is always in real µs (explicit "us" suffix in the log line).
#
# Output:
#   Samples: N
#     mean  X.XXX ms   (XXX inf/s)
#     p50   X.XXX ms
#     p95   X.XXX ms
#     p99   X.XXX ms
#     min   X.XXX ms
#     max   X.XXX ms
#   Sanity check (TotalDuration/InferenceCnt): X.XXX ms  →  XXX inf/s  [OK | WARNING]

import sys
import re
import pathlib
import numpy as np


COLUMN = "totalRoundtripTimeUs"
UNITS_TO_MS = 1000.0       # µs → ms
# Empirical unit scale: calibrated qaic-runner output writes 100ns values in
# totalRoundtripTimeUs despite the "Us" name, so raw values are 10× true µs.
# Set to 1.0 if a future SDK corrects the column to true µs.
UNIT_SCALE  = 10.0
SANITY_RATIO_MAX = 2.0     # warn and fall back if CSV mean and fallback differ by more than this


def parse_runner_log(log_path):
    """Return (inference_cnt, total_duration_us) from a qaic-runner stdout log, or (None, None)."""
    text = pathlib.Path(log_path).read_text()
    # Matches: "InferenceCnt 14913 TotalDuration 30001561us" (unit suffix optional)
    m = re.search(r'InferenceCnt\s+(\d+)\s+TotalDuration\s+(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <prof_out_dir | latency.txt> [runner_log.txt]", file=sys.stderr)
        sys.exit(1)

    arg = pathlib.Path(sys.argv[1])
    lat_file = arg / "aic-profiling-program-0-latency.txt" if arg.is_dir() else arg

    if not lat_file.exists():
        print(f"File not found: {lat_file}", file=sys.stderr)
        sys.exit(1)

    lines = lat_file.read_text().splitlines()

    hdr_idx = next((i for i, l in enumerate(lines) if COLUMN in l), None)
    if hdr_idx is None:
        print(f"Column '{COLUMN}' not found in {lat_file}", file=sys.stderr)
        sys.exit(1)

    cols = [c.strip() for c in lines[hdr_idx].split(",")]
    col_idx = cols.index(COLUMN)

    vals = []
    for line in lines[hdr_idx + 1:]:
        parts = line.split(",")
        if len(parts) > col_idx:
            v = parts[col_idx].strip()
            if v not in ("N/A", ""):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass

    if not vals:
        print(f"No samples found in {lat_file}", file=sys.stderr)
        sys.exit(1)

    arr = np.array(vals) / UNIT_SCALE / UNITS_TO_MS
    csv_mean_ms = arr.mean()

    # Locate runner log: explicit arg > auto-detect in prof_out dir
    log_path = None
    if len(sys.argv) == 3:
        log_path = pathlib.Path(sys.argv[2])
    else:
        candidate = (arg if arg.is_dir() else arg.parent) / "runner.log"
        if candidate.exists():
            log_path = candidate

    fallback_ms = None
    if log_path and log_path.exists():
        inf_cnt, total_dur_us = parse_runner_log(log_path)
        if inf_cnt and inf_cnt > 0:
            fallback_ms = total_dur_us / inf_cnt / 1000.0  # µs → ms

    # Decide whether CSV values are trustworthy
    use_fallback = False
    if fallback_ms is not None:
        ratio = max(csv_mean_ms, fallback_ms) / min(csv_mean_ms, fallback_ms)
        use_fallback = ratio > SANITY_RATIO_MAX

    n = len(arr)
    if use_fallback:
        mean_ms = fallback_ms
        print(f"Samples: {n}  [CSV mean unreliable — using TotalDuration/InferenceCnt fallback]")
    else:
        mean_ms = csv_mean_ms
        print(f"Samples: {n}")

    throughput = 1000.0 / mean_ms if mean_ms > 0 else float("nan")
    print(f"  mean  {mean_ms:.3f} ms   ({throughput:.0f} inf/s)")

    if not use_fallback:
        print(f"  p50   {np.percentile(arr, 50):.3f} ms")
        print(f"  p95   {np.percentile(arr, 95):.3f} ms")
        print(f"  p99   {np.percentile(arr, 99):.3f} ms")
        print(f"  min   {arr.min():.3f} ms")
        print(f"  max   {arr.max():.3f} ms")
    else:
        print(f"  p50–p99  (not available from fallback)")

    if fallback_ms is not None:
        fb_throughput = 1000.0 / fallback_ms if fallback_ms > 0 else float("nan")
        ratio = max(csv_mean_ms, fallback_ms) / min(csv_mean_ms, fallback_ms)
        status = f"WARNING ratio={ratio:.1f}x > {SANITY_RATIO_MAX}x — fell back" if use_fallback else f"OK ratio={ratio:.2f}x"
        print(f"  Sanity (TotalDuration/InferenceCnt): {fallback_ms:.3f} ms  →  {fb_throughput:.0f} inf/s  [{status}]")


if __name__ == "__main__":
    main()
