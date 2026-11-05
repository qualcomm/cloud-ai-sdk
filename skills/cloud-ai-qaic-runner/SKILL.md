---
name: cloud-ai-qaic-runner
description: Detailed reference for qaic-runner on Cloud AI 100. Use when user needs to run a compiled QPC binary, measure throughput or latency, capture output tensors, collect latency, trace, or raw-device profiling data, or post-process opstats with qaic-opstats.
---

# Skill: Cloud AI 100 – `qaic-runner` Detailed Tool Guide

## Overview

This skill is a detailed reference for `qaic-runner`, the runtime tool for executing QPC binaries on Cloud AI 100 devices.

It covers:

- Command-line arguments and usage patterns
- Device selection and multi-device mapping
- Input JSON files and output capture
- Live reporting and profiling
- Throughput and latency measurement patterns

For compiling models to QPC, see the `cloud-ai-qaic-compile` skill and `cloud-ai-model-conversion` skill.

---

## 1. Basic Usage

### 1.1 Quick-Start (Benchmarking)

Use `--aic-profiling-type latency` with `--aic-profiling-start-delay` for benchmarking. The start-delay provides time-based warmup before any timing is recorded — this works correctly regardless of model throughput, unlike an iteration-count warmup which is insufficient for high-throughput models:

```bash
mkdir -p ./prof_out
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  --aic-profiling-type latency \
  --aic-profiling-start-delay 2000 \
  --aic-profiling-num-samples 500 \
  --aic-profiling-out-dir ./prof_out \
  --time 30 -a 1 -S 1 -d <QID>
```

- `-t ./model_qpc/`: directory containing the QPC binary.
- `--aic-batch-json-input ./inputs.json`: maps preprocessed input files to QPC tensors.
- `--aic-profiling-type latency`: collect per-inference host-side timing into a file.
- `--aic-profiling-start-delay 2000`: wait 2 seconds before recording samples (warmup period).
- `--aic-profiling-num-samples 500`: collect 500 timed samples after warmup.
- `--time 30`: run for 30 seconds total (warmup + samples + margin).
- `-a 1`: single activation — correct baseline for single-inference latency.
- `-S 1`: one ExecObj per activation.
- `-d <QID>`: device ID. Select a QID where `Nsp Free ≥` the model's core count — see §2.1.

The run prints overall `Inf/Sec` to stdout. For per-inference latency percentiles, use `parse_latency.py` from the `cloud-ai-performance-optimization` skill's `scripts/` directory:

```bash
python parse_latency.py ./prof_out
```

The script reads `totalRoundtripTimeUs` and outputs mean, p50, p95, p99, min, max in milliseconds. Values are in microseconds (µs) as the column name indicates — divide by 1000 for ms. **Do not use `execToCompleteUs`** — that column measures device execution time only and excludes host-side enqueue and post-processing overhead.

**Note:** `-i` is deprecated in favour of `--aic-batch-json-input`. Batch JSON works with `latency`, `raw_device_stats`, and `--time`. It cannot be combined with live reporting (`-l`). `-s` enables detailed live reporting, so it is also unavailable in batch mode. Use `-i` for live reporting.

---

## 2. Device Selection

### 2.1 Single Device

| Flag | Description |
|---|---|
| `-d`, `--dev <qid>` | Run on a single device (QID). If omitted, a device is selected automatically. |

**Do not hardcode `-d <QID>`.** A model compiled with `N` cores requires `N` free NSPs on the target device. If another process is already using the device, `Nsp Free` may be less than `N` and the run will fail.

Before running, scan all devices for available NSPs:

```bash
/opt/qti-aic/tools/qaic-util -q | grep -e QID -e "Nsp Free"
```

Example output:

```
QID 0
        Nsp Free:4      # partially in use — only 4 NSPs free
QID 1
        Nsp Free:16     # idle — safe to use for a 16-core model
QID 2
        Nsp Free:0      # fully occupied — skip
```

Select the first QID where `Nsp Free ≥ N` (the core count the QPC was compiled with). Pass that QID to `-d`:

```bash
/opt/qti-aic/exec/qaic-runner -t ./model_qpc/ -i ./input.raw -d <QID>
```

### 2.2 Multi-Device (MDP) Execution

When the QPC was compiled with MDP (multi-device partitioning):

| Flag | Description |
|---|---|
| `-D`, `--dev-list <qid-list>` | Device list for multi-device networks. |

Formats:

- `3:1:2` – ordered assignment (partition 0 → device 3, partition 1 → device 1, partition 2 → device 2).
- `3,4,5` – choose devices from the set `{3,4,5}`.
- `3..5,7..9` – choose from ranges 3–5 and 7–9.

Examples:

```bash
# Ordered assignment
/opt/qti-aic/exec/qaic-runner -t ./mdp_qpc/ --aic-batch-json-input ./inputs.json -D 3:1:2

# Auto-pick from pool
/opt/qti-aic/exec/qaic-runner -t ./mdp_qpc/ --aic-batch-json-input ./inputs.json -D 3..5,7..9
```

---

## 3. Input Data

### 3.1 Input Modes

**Live reporting (`-i`)** — use only when host reporting is required:

```bash
-i ./input.raw
```

Pass a single preprocessed `.raw` file (float32, row-major, matching the compiled input shape). Multiple inputs can be passed with repeated `-i` flags. `-i` is deprecated in the SDK but remains the only mode compatible with `-l` and `-s`.

**Batch mode (`--aic-batch-json-input`)** — use for batch inference, output capture, latency collection, or raw-device profiling. It cannot be combined with `-l` or `-s`.

> **CRITICAL: `qaic-runner`'s JSON format is completely different from `qaic-compile`'s format.** Do NOT use `{"inputs": [...]}` (the qaic-compile calibration format) with `qaic-runner` — it crashes with `nlohmann::json type_error.305`. `qaic-runner` uses the `IO-files` schema shown below.

The JSON must be a file path (not inline). Format — one IO set per inference, each IO set is an array of entry objects:

```json
{
  "IO-files": [
    [
      { "path": "input.raw", "io-direction": "in",  "data-type": "float", "map-to": "<input_tensor_name>"  },
      { "path": "out.bin",   "io-direction": "out", "data-type": "float", "map-to": "<output_tensor_name>" }
    ]
  ]
}
```

Required fields per entry: `path` (absolute or relative to JSON file), `io-direction` (`"in"` or `"out"`), `data-type` (e.g. `"float"`, `"float16"`, `"int8_t"`), `map-to` (tensor name from the ONNX graph).  
Optional: `dims` (shape as int array, strongly recommended for partial/output buffers), `skip-validation` (bool, output only).

Output entries can be omitted — the runner writes all output activations to `--write-output-dir` automatically when `--write-output-dir` is used; in that case only the input entries are needed in the JSON.

Working minimal example (input-only, output captured via `--write-output-dir`):

```bash
cat > inputs.json << 'EOF'
{
  "IO-files": [[
    {"path":"/workspace/input.raw","io-direction":"in","data-type":"float","map-to":"input"}
  ]]
}
EOF

/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  --write-output-dir ./qaic_output \
  --write-output-start-iter 0 --write-output-num-samples 1 \
  -n 5 -a 1 -d <QID>
```

| Flag | Description |
|---|---|
| `-i <path>` | Single raw input file for benchmarking (supports live reporting) |
| `--aic-batch-json-input <path>` | JSON file for batch inference, output capture, or profiling (incompatible with `-l`) |
| `--aic-batch-max-memory <MB>` | Limit memory used for loading input files in batch mode |

Notes:

- `.raw` files must be written in row-major order consistent with your layout (e.g., NCHW float32).
- Generate `.raw` files with `numpy.ndarray.tofile()` after preprocessing.

### 3.2 Random Input Data (Testing Only)

`qaic-runner` can also rely on defaults or random data for quick tests (consult SDK docs); however, for accuracy and performance evaluation you should prefer explicit JSON inputs.

---

## 4. Output Capture

| Flag | Description |
|---|---|
| `--write-output-dir <path>` | Directory to write output tensors |
| `--write-output-start-iter <N>` | First iteration from which outputs are captured |
| `--write-output-num-samples <N>` | Number of samples for which outputs are captured |

Example:

```bash
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  --write-output-dir ./outputs \
  --write-output-start-iter 0 \
  --write-output-num-samples 5 \
  -n 100 -d <QID>
```

Use this to compare device outputs with CPU baseline for a handful of inputs. Batch mode (`--aic-batch-json-input`) is appropriate here since live reporting is not needed.

---

## 5. Throughput & Latency

### 5.1 Core Flags

| Flag | Description |
|---|---|
| `-n`, `--num-iter <N>` | Number of inferences to run |
| `--time <seconds>` | Run inferences for a fixed duration instead of fixed count |
| `-a`, `--aic-num-activations <N>` | Number of independent inference pipelines to run concurrently |
| `-S`, `--set-size <N>` | ExecObjs per activation |

> **For standard model-onboarding benchmarks, pass `-S 1` and report `S=1`.**
>
> `-a` selects independent inference pipelines. It does not spread one request across more resources or reduce its unloaded latency.
>
> Use `-a 1` for unloaded per-request latency. For a selected full-SoC throughput row, set `-a` to that row's `instances` value. `cloud-ai-performance-optimization` owns the sweep and full-SoC selection; do not invent a concurrency value in this tool reference.

Example for steady-state throughput:

```bash
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  -a <instances> -S 1 --time 30 \
  -d <QID>
```

Guidelines:

- Use `--time` for steady-state throughput; `-n` for fixed-count micro-benchmarks.
- For unloaded latency, use `-a 1 -S 1`. For the selected full-SoC throughput row, use `-a <instances> -S 1`.
- `Inf/Sec` is aggregate QPC executions/sec across `-a` activations. For a
  static-batch QPC, completed inference units/sec is `Inf/Sec * compiled batch`.
  Query the QPC with `qaic-runner -t <qpc> -q` to obtain its input dimensions;
  the summary `BatchSize` field is not the compiled batch.

### 5.2 Live Reporting & Stats

| Flag | Description |
|---|---|
| `-l`, `--live-reporting` | Enable live throughput reporting (only with `-i`, not `--aic-batch-json-input`) |
| `-r`, `--live-reporting-period <ms>` | Reporting period in milliseconds |
| `-s`, `--stats` | Detailed host-side stats (enqueue/submit, pre/postproc times) |
| `-v` | Verbose logging |

Live host telemetry requires `-i ./input.raw -l -s`; Batch JSON cannot supply this path.

Example:

```bash
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  -i ./input.raw -l -s \
  -a 1 --time 30 -r 1000 -v
```

Interpretation:

- Live reporting shows inferences-per-second (IPS) over time.
- `--stats` (`-s`) reports host enqueue, submit, pre-processing, execution, post-processing, and queue-depth statistics. Compare `enqueueLatencyStats`, `submitLatencyStats`, `execTotalTimeStats`, and `postProcessingLatencyStats` to locate host submission or processing overhead.
- `queueDepthStats max: 0` at `-a 1` means the queue never backs up — this is expected and correct for benchmarking.
- Live host telemetry is diagnostic, not a benchmark result.

---

## 6. Profiling

`qaic-runner` can collect various profiling data from device execution. The
device-stat and trace modes require an appropriate `-stats-level`; host latency
collection does not. See §6.1.

| Flag | Description |
|---|---|
| `--aic-profiling-type <type>` | Profiling mode: `latency`, `trace`, `raw_device_stats` |
| `--aic-profiling-out-dir <path>` | Directory for profiling output files (must exist) |
| `--aic-profiling-start-iter <N>` | Iteration at which profiling starts (default: 0) |
| `--aic-profiling-num-samples <N>` | Number of samples to save to file (default: 1). Keep this small — 1–3 is sufficient; the op breakdown is stable across samples. Each sample writes a large `.bin` file and requires `qaic-opstats` post-processing. |
| `--aic-profiling-start-delay <ms>` | Delay in ms before profiling begins |

### 6.1 Compile-Time Prerequisite: `-stats-level`

The QPC must be compiled with `-stats-level` to embed a stats buffer. Stats levels are additive:

| Level | What it enables |
|---|---|
| 40 (minimum; omitting the flag embeds no stats) | Per-core/thread inference duration (UCycles, PCycles), activation time |
| 50 | Adds total sync, op and barrier times |
| 70 | **Operator-level cycle counts (opstats)** — required for `qaic-opstats` post-processing |
| 80 | More detailed per-operation stats |
| 100 | PMU stats |

**Level 70 or higher is required for per-operator breakdown via `qaic-opstats`.** Host latency collection does not require `-stats-level`. Omit `-stats-level` from production builds when diagnostic data is not needed.

> **Critical:** If the QPC was compiled without `-stats-level >= 70`, `--aic-profiling-type raw_device_stats` may be rejected or may complete without producing a `.bin` file, depending on the runtime. Recompile with `-stats-level=80` (or 70) before collecting opstats. Preserve the raw buffer with its diagnostic QPC and decode it with `qaic-opstats` from the matching SDK release. Collection succeeds only when that decode writes the expected summary and trace files; a `.bin` file alone is not evidence of a usable profile.

### 6.2 Profiling Types

**`latency`** — per-inference host-side timing written to a CSV-style `.txt` file. Use this for benchmarking with warmup and percentile calculation. Does not require a specific `-stats-level`. Use `--aic-profiling-start-delay` (ms) for time-based warmup — this is robust across all throughput levels. Do not use `--aic-profiling-start-iter` for warmup: an iteration count that is sufficient for a 200 inf/sec model may be inadequate for a 4000 inf/sec model.

```bash
mkdir -p ./prof_out
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  --aic-profiling-type latency \
  --aic-profiling-out-dir ./prof_out \
  --aic-profiling-start-delay 2000 \
  --aic-profiling-num-samples 500 \
  --time 30 -a 1 -S 1 -d <QID>
```

Output file: `prof_out/aic-profiling-program-0-latency.txt`. Key column: `totalRoundtripTimeUs` — end-to-end host-measured latency per inference (µs). Use `parse_latency.py` from the `cloud-ai-performance-optimization` skill to compute percentiles.

**`raw_device_stats`** — raw `.bin` stats buffers from the device. Required for per-operator profiling with `qaic-opstats`. Requires `-stats-level >= 70`. Collect one diagnostic QPC instance with `-a 1`.

```bash
mkdir -p ./stats
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  --aic-batch-json-input ./inputs.json \
  --aic-profiling-type raw_device_stats \
  --aic-profiling-out-dir ./stats \
  --aic-profiling-num-samples 5 \
  --aic-profiling-start-iter 10 \
  -n 100 -a 1 -S 1 -d <QID>
```

Output files are named: `aic-profiling-program-<P>-activation-<A>-inf-<I>-<NetworkName>-aiccyclecounts.bin`

When the diagnostic QPC was compiled with `-ddr-stats`, runner stdout can include graph-level metrics such as:

```text
DDRTrafficMB_<GraphName>,         573.01, 526.47, 585.55, 17.58
DDRBandwidthGBPerSec_<GraphName>, 27.80,  23.62,  30.08,  1.78
```

The graph suffix is emitted by the QPC; do not hard-code it. If multiple graphs emit a requested metric, select one graph before comparing frequency, cycles, or DDR values. These are graph aggregates; obtain per-core DDR from the decoded opstats summary. Use a diagnostic QPC, not a production benchmark QPC, for `-ddr-stats` or `-stats-level` collection.

**`trace`** — runtime-processed Chrome Tracing JSON files. Requires `-stats-level >= 40`.

The `stats` profiling value is rejected by SDK 1.23. Use `latency` for host
latency files, `raw_device_stats` for `qaic-opstats`, or `trace` for a runtime
trace.

### 6.3 Post-Processing with `qaic-opstats`

`qaic-opstats` decodes `raw_device_stats` `.bin` files into human-readable summaries and Chrome trace files.

#### End-to-end opstats workflow

**Step 1 — Compile with stats level ≥ 70** (add to your `qaic-compile` invocation):

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=model.onnx \
  -aic-hw -aic-hw-version=ai100 \
  -aic-binary-dir=./model_qpc \
  -stats-level=80          # ← required; 70 minimum, 80 recommended
```

**Step 2 — Collect raw stats buffer** (output dir must exist):

```bash
mkdir -p ./stats_buf
/opt/qti-aic/exec/qaic-runner \
  -t ./model_qpc/ \
  -num-iter 10 \
  --aic-batch-json-input ./inputs.json \
  --aic-profiling-type raw_device_stats \
  --aic-profiling-num-samples 1 \
  --aic-profiling-out-dir ./stats_buf \
  -a 1 -S 1
```

This writes one binary file per sample:
```
stats_buf/aic-profiling-program-0-inf-1-<NetworkName>-aiccyclecounts_<NetworkName>_.bin
```

**Step 3 — Decode with `qaic-opstats`** (output dir must exist):

```bash
mkdir -p ./stats_out
/opt/qti-aic/exec/qaic-opstats \
  -q ./model_qpc/programqpc.bin \
  -i ./stats_buf/ \
  --summary \
  --trace \
  -o ./stats_out

# Verify output files were written
ls ./stats_out/*.qaic-opstats.summary.txt
```

Outputs:
- `*.qaic-opstats.summary.txt` — per-op cycle breakdown (text)
- `*.qaic-opstats.trace.json` — Chrome Tracing file (open at [ui.perfetto.dev](https://ui.perfetto.dev))

#### Reading the summary

```
core  ucycles  ucycles(%)     pcycles  pcycles(%)  count  out KB  opdetails
c0     630000      99.35%    36100000      99.35%      1   3060   <op-type> HVX CacheableDDR
c0      12000       1.90%         N/A         N/A      2     N/A  barrier HVX
c0       4000       0.63%         N/A         N/A      4   3363   aiccopytovtcm DMAIssue DDR
...
c0    t0 HVX    630867 ucycles    36143407 pcycles  97.80% utilization
c0    t1 HVX    630867 ucycles    36143403 pcycles  99.21% utilization
```

Key columns:
- **ucycles** — wall-clock cycles (all threads in parallel); use for total inference time
- **pcycles** — pipeline cycles (single-thread compute); use for per-op compute cost
- **ucycles%** — fraction of total inference time this op occupies
- **count** — number of times this op executed
- **opdetails** — op type, execution unit (HVX/HMX/DMAIssue), memory type

Thread utilization rows show per-thread HVX utilization — low utilization on some threads indicates load imbalance.

#### Common patterns and what they mean

| Pattern | Interpretation |
|---|---|
| `aiccopytovtcm` + `aiccopyfromvtcm` < 1% | DMA is not the bottleneck; focus on compute |
| `barrier HVX` > 5% | Thread sync overhead is significant — consider reducing sync points or rebalancing work |
| Thread utilization varies across threads | Load imbalance — some threads handle more boundary cases |
| HMX utilization 0% | No matrix multiply in the kernel — expected for filter/custom ops |
| `raw_device_stats` fails with "Program does not support" | QPC compiled without `-stats-level >= 70` — recompile |
| Only a latency `.txt` file is available for opstats | Latency profiling does not produce the raw `.bin` buffers; collect `raw_device_stats` from a stats-enabled diagnostic QPC |

Key flags:

| Flag | Description |
|---|---|
| `-q`/`--qpc <path>` | Path to `programqpc.bin` |
| `-i`/`--input-dir <path>` | Directory containing `.bin` stats buffer files |
| `-o`/`--output-dir <path>` | Output directory (must exist; default: `.`) |
| `--summary` | Generate `.summary.txt` per-op text summaries |
| `--trace` | Generate `.trace.json` Chrome Tracing files |
| `--select-cores <list>` | Comma-separated core numbers (default: all) |
| `--flow-events full` | Show operator dependency flow events in trace |

`.trace.json` files are viewable in Chrome at `chrome://tracing` or at [ui.perfetto.dev](https://ui.perfetto.dev).

Notes:

- Profiling introduces overhead; don't use it for final throughput numbers.
- `trace` is processed by the runtime; `raw_device_stats` requires `qaic-opstats` post-processing.
- For per-operator breakdown, always use `raw_device_stats` + `qaic-opstats`;
  latency and trace outputs are different artifacts.

---

## 7. Timeouts & Robustness

| Flag | Description |
|---|---|
| `--datapath-timeout <ms>` | Timeout waiting for inference completion (0 = kernel default) |

Default is typically large (e.g., 2 hours); you can lower it for testing.

**`Status:Error` on all QIDs / `Permission denied … Error no 13`:** The container user must be root or a member of the `qaic` group to open `/dev/accel/` devices (`crw-rw---- root:qaic`). Symptom from `qaic-util -q`:
```
[error] Device Path /dev/accel/accel0 failed to open. Error no 13, Permission denied
...
QID 0
        Status:Error
```
Fix — choose one:
- **Run as root (simplest):** omit the `-u` flag from `docker run` (the inference container defaults to root).
- **Keep non-root user:** add `--group-add $(getent group qaic | cut -d: -f3)` to `docker run` to give the container user access to `/dev/accel/`. See the `cloud-ai-env-setup` skill §4 for the full command.

---

## 8. Workflow Summary

1. Compile model to QPC with `qaic-compile`.
2. Preprocess a representative input to a `.raw` file (`numpy.ndarray.tofile()`).
3. Run `qaic-runner` with:
   - `-t` to point to the QPC directory.
   - `--aic-batch-json-input` for benchmarking, batch inference, output capture, latency collection, and raw-device profiling; `-i ./input.raw` only for live reporting.
   - `-a 1` for benchmarking; `-n` / `--time` to control run length.
   - `-l` / `-s` for live stats (only with `-i`).
4. Use `--write-output-*` with `--aic-batch-json-input` for correctness checks.
5. Use `--aic-profiling-*` to gather detailed per-op performance data.

Combine this tool guide with:

- **Accuracy Tuning** for quantization effects (`cloud-ai-accuracy-tuning` skill).
- **Performance Optimization** for the optimization order, graph/compile prerequisites, and profiling workflow (`cloud-ai-performance-optimization` skill).
- **Architecture Context** to interpret NoC/VTCM constraints behind observed performance (`cloud-ai-architecture` skill).
- **`qaic-compile` Detailed Tool Guide** for compile-time options (`cloud-ai-qaic-compile` skill).
- **`qaic.Session` Python API Guide** for programmatic benchmarking (`cloud-ai-qaic-python` skill).

---

## Related Skills

- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Go here first to produce the QPC binary and `inputs.json` that `qaic-runner` consumes.
- **Cloud AI 100 – Accuracy Tuning (FP16 & INT8)** (`cloud-ai-accuracy-tuning` skill) — Go here when `--write-output-*` results show accuracy degradation after quantization.
- **Cloud AI 100 – Performance Optimization & Runtime** (`cloud-ai-performance-optimization` skill) — Go here for compiler flag tuning (`-mos`, `-ols`) and how to interpret throughput results against target benchmarks.
- **Cloud AI 100 – SoC Architecture** (`cloud-ai-architecture` skill) — Consult to interpret *why* throughput is limited (DDR-bound, repeated VTCM spill/fill traffic, NoC saturation).
- **Cloud AI 100 – `qaic-compile` Detailed Tool Guide** (`cloud-ai-qaic-compile` skill) — Go here to recompile with different flags when `qaic-runner` results indicate a compile-time bottleneck.
- **Cloud AI 100 – `qaic.Session` Python API Guide** (`cloud-ai-qaic-python` skill) — Use as a Python alternative for integration testing and output comparison; prefer `qaic-runner` for throughput benchmarking.
