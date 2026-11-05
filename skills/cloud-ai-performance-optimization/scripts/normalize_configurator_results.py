#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#
# normalize_configurator_results.py - Normalize qaic-model-configurator results.csv
#
# Usage:
#   python normalize_configurator_results.py cfg_out/results.csv
#   python normalize_configurator_results.py cfg_out/results.csv --in-place
#
# qaic-model-configurator reports aggregate throughput plus a derived Avg Latency column.
# Normalize Avg Latency from the trusted relationship:
#
#   Avg Latency (us) = instances * batchSize * 1_000_000 / Throughput (inf/sec)
#
# By default this writes results.normalized.csv next to the input. With --in-place it
# preserves the original as results.raw.csv, then replaces results.csv.
#

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import shutil
import sys


BATCH = "batchSize"
INSTANCES = "instances"
THROUGHPUT = "Throughput (inf/sec)"
AVG_LATENCY = "Avg Latency (us)"
REQUIRED_COLUMNS = (BATCH, INSTANCES, THROUGHPUT, AVG_LATENCY)


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "n/a", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: str) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def normalized_latency_us(row: dict[str, str]) -> float | None:
    batch = parse_int(row.get(BATCH, ""))
    instances = parse_int(row.get(INSTANCES, ""))
    throughput = parse_float(row.get(THROUGHPUT, ""))

    if not batch or not instances or throughput is None or throughput <= 0:
        return None

    return instances * batch * 1_000_000.0 / throughput


def normalize_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, float | int]]:
    normalized_rows: list[dict[str, str]] = []
    changed = 0
    skipped = 0
    max_ratio = 1.0

    for row in rows:
        out = dict(row)
        normalized = normalized_latency_us(out)
        raw = parse_float(out.get(AVG_LATENCY, ""))

        if normalized is None:
            skipped += 1
            normalized_rows.append(out)
            continue

        out[AVG_LATENCY] = f"{normalized:.2f}"

        if raw is None or raw <= 0:
            changed += 1
        else:
            ratio = max(raw, normalized) / min(raw, normalized)
            max_ratio = max(max_ratio, ratio)
            if abs(raw - normalized) > 0.5:
                changed += 1

        normalized_rows.append(out)

    return normalized_rows, {
        "rows": len(rows),
        "changed": changed,
        "skipped": skipped,
        "max_ratio": max_ratio,
    }


def default_output_path(path: pathlib.Path) -> pathlib.Path:
    if path.name == "results.csv":
        return path.with_name("results.normalized.csv")
    return path.with_suffix(path.suffix + ".normalized.csv")


def raw_backup_path(path: pathlib.Path) -> pathlib.Path:
    if path.name == "results.csv":
        return path.with_name("results.raw.csv")
    return path.with_suffix(path.suffix + ".raw")


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
        return fieldnames, list(reader)


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize qaic-model-configurator results.csv latency column")
    parser.add_argument("results_csv", type=pathlib.Path, help="qaic-model-configurator results.csv")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Output CSV path")
    parser.add_argument("--in-place", action="store_true", help="Preserve original as results.raw.csv, then replace input CSV")
    args = parser.parse_args()

    path = args.results_csv
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    try:
        fieldnames, rows = read_csv(path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    normalized_rows, stats = normalize_rows(rows)

    raw_path = None
    if args.in_place:
        raw_path = raw_backup_path(path)
        if not raw_path.exists():
            shutil.copy2(path, raw_path)
        output_path = path
    else:
        output_path = args.output or default_output_path(path)

    write_csv(output_path, fieldnames, normalized_rows)

    print(f"Input:  {path}")
    if raw_path:
        print(f"Raw:    {raw_path}")
    print(f"Output: {output_path}")
    print(
        "Rows: {rows}  changed: {changed}  skipped: {skipped}  "
        "max_raw_vs_normalized_ratio: {max_ratio:.2f}x".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
