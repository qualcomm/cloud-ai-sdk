#!/usr/bin/env bash
####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#
# qaic_profile.sh — Collect and analyze per-op runtime stats for any Cloud AI 100 QPC
#
# Usage:
#   bash qaic_profile.sh --qpc <dir> (--batch-json <inputs.json> | --input <file.raw>) [OPTIONS]
#
# Required:
#   --qpc   <dir>     QPC directory (must contain programqpc.bin)
#   --batch-json <path>  qaic-runner IO-files JSON (canonical collection input)
#                         or --input <file> when collection is enabled
#   --input <file>    Deprecated raw-input fallback; use only when Batch JSON is unavailable
#
# Optional:
#   --device  <id>    Device ID (default: 0)
#   --samples <n>     Number of profiling samples to collect (default: 3). Keep small —
#                     1–3 is sufficient; the op breakdown is stable across samples.
#                     Each sample writes a large .bin file and requires qaic-opstats
#                     post-processing. Large values (e.g. 100) waste time and disk.
#   --iter    <n>     Total inferences to run (default: 100)
#   --start   <n>     Iteration to start collecting samples (default: 10)
#   --stats-dir <dir> Directory for raw .bin stats buffers (default: ./qaic_stats)
#   --out-dir   <dir> Directory for qaic-opstats output (default: ./qaic_stats_out)
#   --no-collect      Skip collection, re-analyze existing stats in --stats-dir
#
# Prerequisites:
#   - QPC compiled with -stats-level=70 (operator-level cycle counts)
#   - /opt/qti-aic/exec/qaic-runner and /opt/qti-aic/exec/qaic-opstats available
#
# Output:
#   Printed analysis:
#     - Thread utilization (HVX / HMX / DMA) averaged across all cores
#     - Top 20 ops by ucycles (core 0 as representative)
#     - Bottleneck signals with automatic diagnosis
#     - Per-core DDR from the decoded opstats summary when the QPC has -ddr-stats
#   Files:
#     <stats-dir>/   raw .bin stats buffers from device
#     <out-dir>/     .summary.txt and .trace.json per sample
#                    (view .trace.json at chrome://tracing or https://ui.perfetto.dev)
#     <out-dir>/qaic_runner_stats.log
#                    qaic-runner stdout with graph-suffixed runtime metrics
#

set -euo pipefail

# ── Argument parsing ───────────────────────────────────────────────────────────

QPC_DIR=""
INPUT_RAW=""
BATCH_JSON=""
DEVICE_ID=0
NUM_SAMPLES=3
NUM_ITER=100
START_ITER=10
STATS_DIR="./qaic_stats"
OUT_DIR="./qaic_stats_out"
NO_COLLECT=0

usage() {
    sed -n '/^# Usage/,/^#$/p' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qpc)        QPC_DIR="$2";    shift 2 ;;
        --input)      INPUT_RAW="$2";  shift 2 ;;
        --batch-json) BATCH_JSON="$2"; shift 2 ;;
        --device)     DEVICE_ID="$2";  shift 2 ;;
        --samples)  NUM_SAMPLES="$2"; shift 2 ;;
        --iter)     NUM_ITER="$2";   shift 2 ;;
        --start)    START_ITER="$2"; shift 2 ;;
        --stats-dir) STATS_DIR="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2";    shift 2 ;;
        --no-collect) NO_COLLECT=1;  shift ;;
        -h|--help)  usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "${QPC_DIR}" ]]; then
    echo "ERROR: --qpc is required"
    usage
fi

if [[ -n "${BATCH_JSON}" && -n "${INPUT_RAW}" ]]; then
    echo "ERROR: --batch-json and --input cannot be combined"
    usage
fi

if [[ "${NO_COLLECT}" -eq 0 && -z "${BATCH_JSON}" && -z "${INPUT_RAW}" ]]; then
    echo "ERROR: --batch-json or --input is required when collection is enabled"
    usage
fi

QPC_DIR="$(realpath "${QPC_DIR}")"
QPC_BIN="${QPC_DIR}/programqpc.bin"

if [[ ! -f "${QPC_BIN}" ]]; then
    echo "ERROR: programqpc.bin not found in ${QPC_DIR}"
    exit 1
fi

if [[ -n "${BATCH_JSON}" ]]; then
    if [[ ! -f "${BATCH_JSON}" ]]; then
        echo "ERROR: Batch JSON file not found: ${BATCH_JSON}"
        exit 1
    fi
    BATCH_JSON="$(realpath "${BATCH_JSON}")"
fi

if [[ -n "${INPUT_RAW}" ]]; then
    if [[ ! -f "${INPUT_RAW}" ]]; then
        echo "ERROR: Input file not found: ${INPUT_RAW}"
        exit 1
    fi
    INPUT_RAW="$(realpath "${INPUT_RAW}")"
fi

STATS_DIR="$(realpath -m "${STATS_DIR}")"
OUT_DIR="$(realpath -m "${OUT_DIR}")"
if [[ "${STATS_DIR}" == "${OUT_DIR}" || "${STATS_DIR}/" == "${OUT_DIR}/"* || "${OUT_DIR}/" == "${STATS_DIR}/"* ]]; then
    echo "ERROR: --stats-dir and --out-dir must be separate, non-nested directories"
    exit 1
fi

ensure_empty_output_dir() {
    local dir="$1"
    local label="$2"
    if [[ -e "${dir}" && ! -d "${dir}" ]]; then
        echo "ERROR: ${label} is not a directory: ${dir}"
        exit 1
    fi
    if [[ -d "${dir}" ]] && [[ -n "$(find "${dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: ${label} already contains artifacts: ${dir}"
        echo "       Choose a new directory or inspect and remove its contents explicitly."
        exit 1
    fi
    mkdir -p "${dir}"
}

RUNNER_STDOUT=$(mktemp)
trap 'rm -f "${RUNNER_STDOUT}"' EXIT
RUNNER_METRICS_SOURCE="${RUNNER_STDOUT}"

# ── Step 1: Collect raw device stats ──────────────────────────────────────────

if [[ "${NO_COLLECT}" -eq 0 ]]; then
    echo "=== Collecting raw device stats ==="
    ensure_empty_output_dir "${STATS_DIR}" "--stats-dir"
    ensure_empty_output_dir "${OUT_DIR}" "--out-dir"
else
    echo "=== Skipping collection, using existing stats in ${STATS_DIR} ==="
    mkdir -p "${OUT_DIR}"
fi

RUNNER_STATS_LOG="${OUT_DIR}/qaic_runner_stats.log"

if [[ "${NO_COLLECT}" -eq 0 ]]; then
    if [[ -n "${BATCH_JSON}" ]]; then
        /opt/qti-aic/exec/qaic-runner \
            -t "${QPC_DIR}" \
            --aic-batch-json-input "${BATCH_JSON}" \
            --aic-profiling-type raw_device_stats \
            --aic-profiling-out-dir "${STATS_DIR}" \
            --aic-profiling-num-samples "${NUM_SAMPLES}" \
            --aic-profiling-start-iter "${START_ITER}" \
            -n "${NUM_ITER}" -a 1 -S 1 -d "${DEVICE_ID}" 2>&1 | tee "${RUNNER_STDOUT}" | grep -v "^$"
    else
        echo "WARNING: --input uses deprecated qaic-runner input mode; raw buffers are contained in ${STATS_DIR}."
        (
            cd "${STATS_DIR}"
            /opt/qti-aic/exec/qaic-runner \
                -t "${QPC_DIR}" \
                -i "${INPUT_RAW}" \
                --aic-profiling-type raw_device_stats \
                --aic-profiling-out-dir "${STATS_DIR}" \
                --aic-profiling-num-samples "${NUM_SAMPLES}" \
                --aic-profiling-start-iter "${START_ITER}" \
                -n "${NUM_ITER}" -a 1 -S 1 -d "${DEVICE_ID}"
        ) 2>&1 | tee "${RUNNER_STDOUT}" | grep -v "^$"
    fi
    cp "${RUNNER_STDOUT}" "${RUNNER_STATS_LOG}"
elif [[ -f "${RUNNER_STATS_LOG}" ]]; then
    RUNNER_METRICS_SOURCE="${RUNNER_STATS_LOG}"
else
    echo "WARNING: ${RUNNER_STATS_LOG} is missing; runner metrics are unavailable."
fi

# ── Step 2: Post-process with qaic-opstats ─────────────────────────────────────

echo ""
echo "=== Post-processing with qaic-opstats ==="

FIRST_BIN=$(find "${STATS_DIR}" -maxdepth 1 -type f -name '*.bin' -print -quit 2>/dev/null || true)
if [[ -z "${FIRST_BIN}" ]]; then
    echo "ERROR: No .bin stats files found in ${STATS_DIR}"
    echo "       Check that the QPC was compiled with -stats-level=70"
    exit 1
fi

OPSTATS_MARKER=$(mktemp "${OUT_DIR}/.qaic_profile.XXXXXX")
trap 'rm -f "${RUNNER_STDOUT}" "${OPSTATS_MARKER:-}"' EXIT

/opt/qti-aic/exec/qaic-opstats \
    --qpc "${QPC_BIN}" \
    --input-dir "${STATS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --summary \
    --trace 2>&1

SUMMARY=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.summary.txt' \
    -newer "${OPSTATS_MARKER}" -print -quit 2>/dev/null || true)
if [[ -z "${SUMMARY}" ]]; then
    echo "ERROR: No fresh summary file generated in ${OUT_DIR}"
    exit 1
fi

TRACE=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.trace.json' \
    -newer "${OPSTATS_MARKER}" -print -quit 2>/dev/null || true)
if [[ -z "${TRACE}" ]]; then
    echo "ERROR: No fresh trace file generated in ${OUT_DIR}"
    exit 1
fi

echo ""
echo "=== Analysis: ${SUMMARY} ==="
echo ""

# ── Step 3: Thread utilization across all cores ────────────────────────────────

echo "--- Thread Utilization ---"
echo "  (% of inference time each thread unit is active)"
echo ""

awk '
/t[0-9]+ HVX/ && /utilization/ {
    for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+%$/) { sub(/%/,"",$i); hvx_sum+=$i; hvx_n++ }
}
/t[0-9]+ HMX/ && /utilization/ {
    for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+%$/) { sub(/%/,"",$i); hmx_sum+=$i; hmx_n++ }
}
/t[0-9]+ DMAIssue/ && /utilization/ {
    for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+%$/) { sub(/%/,"",$i); dma_sum+=$i; dma_n++ }
}
END {
    if (hvx_n > 0) printf "  HVX (vector, element-wise ops):  avg %5.1f%%  across %d threads\n", hvx_sum/hvx_n, hvx_n
    if (hmx_n > 0) printf "  HMX (matrix, Conv/MatMul):        avg %5.1f%%  across %d threads\n", hmx_sum/hmx_n, hmx_n
    if (dma_n > 0) printf "  DMAIssue (weight loads):          avg %5.1f%%  across %d threads\n", dma_sum/dma_n, dma_n
}
' "${SUMMARY}"

echo ""

# ── Step 4: Top ops by ucycles (core 0 as representative) ─────────────────────

echo "--- Top Ops by ucycles (core 0) ---"
echo "  ucycles   %total  op"
echo ""

awk '
/^c0[[:space:]]/ && /[0-9]+\.[0-9]+%/ && !/total:|t[0-9] |activate|batch/ {
    ucyc = $2
    pct  = $3
    op = ""
    for (i=11; i<=NF; i++) op = op " " $i
    sub(/^ /, "", op)
    printf "  %7d  %6s  %s\n", ucyc, pct, op
}
' "${SUMMARY}" | sort -rn | head -20

echo ""

# ── Step 5: Bottleneck signals ────────────────────────────────────────────────

echo "--- Bottleneck Signals ---"
echo ""

awk '
/^c0[[:space:]]/ && !/total:|t[0-9] |activate|batch/ {
    ucyc = $2
    line = $0

    # Sync / stall
    if (line ~ /sync HMX/)                sync_hmx  += ucyc
    if (line ~ /sync HVX/)                sync_hvx  += ucyc
    if (line ~ /barrier HVX/)             bar_hvx   += ucyc
    if (line ~ /barrier HMX/)             bar_hmx   += ucyc

    # Compute
    if (line ~ /aicconvolutiond32/ && line ~ / HMX /)
                                            conv      += ucyc
    if ((line ~ /aicmatmul/ || line ~ /aicgemm/) && line ~ / HMX /)
                                            matmul    += ucyc

    # Format conversion (FP16 <-> D32, required around every HMX op)
    if (line ~ /aicconverttod32/)          cvt_to    += ucyc
    if (line ~ /aicconvertfromd32/)        cvt_from  += ucyc

    # Activations (HVX)
    if (line ~ /sigmoid/)                  sigmoid   += ucyc
    if (line ~ /elementmul/ ||
        line ~ /aicmulsplat/ ||
        line ~ /aicbroadcastmul/)          mul       += ucyc
    if (line ~ /elementadd/ ||
        line ~ /elementsub/)               add       += ucyc
    if (line ~ /aicsoftmax/)               softmax   += ucyc
    if (line ~ /aicmaxpoold32/ ||
        line ~ /aicavgpoold32/)            pool      += ucyc
    if (line ~ /aicupsamplenearestd32/ ||
        line ~ /aicupsamplelineard32/)     upsample  += ucyc
    if (line ~ /aicdecimated32/)           decimate  += ucyc

    # Data movement
    if (line ~ /aiccopytovtcm.*DDR/)       ddr_load  += ucyc
    if (line ~ /aicmulticastvtcm/)         mcast     += ucyc
    if (line ~ /aiccopysamevtcm/)          vtcm_copy += ucyc
}
/^c0 total:/ { total = $3 }

END {
    fmt_conv   = cvt_to + cvt_from
    hmx_work   = conv + matmul
    hvx_act    = sigmoid + mul + add + softmax + pool + upsample + decimate
    data_move  = ddr_load + mcast + vtcm_copy

    printf "  HMX compute (Conv/MatMul):   %5.1f%%  — actual matrix compute\n",              hmx_work/total*100
    printf "  Format conversion:           %5.1f%%  — FP16<->D32 around every HMX op\n",    fmt_conv/total*100
    printf "  HVX activations:             %5.1f%%  — sigmoid/mul/add/softmax/pool/resize\n", hvx_act/total*100
    printf "  Data movement:               %5.1f%%  — DDR loads + multicast + VTCM copies\n", data_move/total*100
    printf "  Sync HMX (HVX waits):        %5.1f%%  — HVX threads stalled waiting for HMX\n", sync_hmx/total*100
    printf "  Sync HVX (HMX waits):        %5.1f%%  — HMX thread stalled waiting for HVX\n",  sync_hvx/total*100
    printf "  Barriers:                    %5.1f%%  — inter-thread sync points\n",            (bar_hvx+bar_hmx)/total*100
    printf "\n"

    # Diagnosis
    if (sync_hmx/total > 0.5)
        print "  DIAGNOSIS: HVX threads spend >50% of time waiting for HMX. Pipeline-latency bound."
    else if (sync_hvx/total > 0.3)
        print "  DIAGNOSIS: HMX thread spends >30% waiting for HVX. HVX (activation) bound."

    if (fmt_conv/total > 0.15)
        print "  NOTE: Format conversion >15% of cycles. INT8 would reduce this significantly."

    if (ddr_load/total > 0.10)
        print "  NOTE: DDR load cost >10%. Consider higher -mos or -multicast-weights."

    if (hmx_work/total < 0.25)
        print "  NOTE: HMX active <25% of wall time. Not compute-bound on matrix unit."

    if (hvx_act/total > 0.20)
        print "  NOTE: HVX activations >20%. Check for fusable ops or hardsigmoid substitution."
}
' "${SUMMARY}"

# ── Step 6: DDR Traffic (requires -ddr-stats compile) ─────────────────────────

echo "--- DDR Traffic ---"
echo ""

PER_CORE_DDR=$(awk '
/^c[0-9]+: DDR traffic count [0-9.]+ MB, bandwidth is [0-9.]+ GB\/s/ {
    core = $1
    sub(/:$/, "", core)
    printf "%s %.6f %.6f\n", core, $5, $9
}
' "${SUMMARY}")

if [[ -n "${PER_CORE_DDR}" ]]; then
    DDR_RANGE=$(awk '
NR == 1 { min = $2; max = $2 }
{ if ($2 < min) min = $2; if ($2 > max) max = $2 }
END {
    ratio = min > 0 ? max / min : 0
    printf "%.3f-%.3f MB/inf across %d core(s), %.2fx range", min, max, NR, ratio
}
' <<<"${PER_CORE_DDR}")
    echo "  Per-core DDR from the decoded opstats summary: ${DDR_RANGE}"
    printf "  %-8s %12s %12s\n" "Core" "MB/inf" "GB/s"
    while read -r core mb bandwidth; do
        printf "  %-8s %12.3f %12.3f\n" "${core}" "${mb}" "${bandwidth}"
    done < <(sort -k2,2nr <<<"${PER_CORE_DDR}")
else
    echo "  No per-core DDR metrics were decoded from this QPC."
fi

echo ""
DDR_LINES=$(grep -E '^(DDRTrafficMB|DDRBandwidthGBPerSec)_[^,]+,' "${RUNNER_METRICS_SOURCE}" || true)

if [[ -z "${DDR_LINES}" ]]; then
    echo "  No graph-level runner DDR metric was observed."
else
    echo "  Graph-level metrics from qaic-runner stdout (avg, min, max, stddev):"
    sed 's/^/  /' <<<"${DDR_LINES}"
fi

echo ""
echo "Trace files (open at chrome://tracing or https://ui.perfetto.dev):"
ls "${OUT_DIR}"/*.trace.json 2>/dev/null | head -3
