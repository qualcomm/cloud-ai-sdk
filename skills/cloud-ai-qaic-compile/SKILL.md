---
name: cloud-ai-qaic-compile
description: Detailed flag reference for qaic-compile on Cloud AI 100. Use when user needs specific compiler flags for quantization, performance tuning, SDP/MDP partitioning, custom I/O, diagnostics, or stats-level profiling. Not for the end-to-end model conversion workflow — use cloud-ai-model-conversion for that.
---

# Skill: Cloud AI 100 – `qaic-compile` Detailed Tool Guide

## Overview

This skill is a **task-focused reference** for `qaic-compile`, the compiler and graph loader for Cloud AI 100. It assumes you already understand:

- Basic model conversion (ONNX/TF → QPC)
- High-level performance/accuracy goals

This guide focuses on:

- Invocation patterns and core flags
- Input/Output & quantization-related options
- Performance and partitioning flags (SDP/MDP)
- Diagnostics and debugging options

For architectural context and high-level workflows, see:

- **Model Conversion & Compilation** (`cloud-ai-model-conversion` skill)
- **Accuracy Tuning (FP16/INT8)** (`cloud-ai-accuracy-tuning` skill)
- **Performance Optimization & Runtime** (`cloud-ai-performance-optimization` skill)

---

## 1. Basic Invocation Patterns

### 1.1 Minimal FP16 Compile

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=16 \
  -aic-binary-dir=./model_fp16_qpc \
```

> **`-aic-binary-dir` must always be a dedicated subdirectory** — never `.` or the current working directory. `qaic-compile` writes `programqpc.bin` and supporting files into that directory and fails if it already exists. Use a versioned name per compile (e.g. `model_fp16_qpc_16c`, `model_fp16_qpc_8c`) so prior QPCs are preserved. If recompiling to the same dedicated subdirectory is intentional, `rm -rf` it first.

Use this to create a baseline FP16 QPC for correctness and performance testing. Set `-aic-num-cores` to the device NSP count (`/opt/qti-aic/tools/qaic-util -q | grep "Nsp Total"`). Model onboarding requires a sweep across 1, 2, 4, 8, and 16 cores — compile a separate QPC per core count. **Do not set `-mos` on the initial compile** — omit the flag for the compiler heuristic. Omit for compiler heuristic - the heuristic selection scales with core count and model structure; for residual-heavy architectures it often outperforms `-mos=1`. MOS tuning belongs in the performance optimization sweep, not the initial baseline.

### 1.1.1 Parallel independent QPC compiles

Launch `qaic-compile` processes concurrently where multiple QPC configs are required. Assess the container or system CPU and memory resources to determine concurrency.  Use the same immutable ONNX, give each process its own
`-aic-binary-dir` and log path, then run the normal per-compile validation on every QPC; retry only failed jobs serially.

> **Never omit `-convert-to-fp16` for a whole-network FP32 compile.** HMX — the high-throughput matrix unit — handles only FP16 and INT8; without this flag Conv and MatMul ops do not map to HMX, and inference latency grows to seconds per frame. If specific ops need FP32 for accuracy, keep them selectively and sparingly via the mixed-precision flags in §3.2.

### 1.2 ONNX Symbolic Dimensions and Specialization

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -onnx-define-symbol=batch_size,1 \
  -onnx-define-symbol=seq_len,128 \
  ...
```

| Flag | Description |
|---|---|
| `-onnx-define-symbol=<sym,value>` | Map a symbolic ONNX dimension to a concrete integer. Repeat per symbol. |
| `-network-specialization-config=<config.json>` | JSON config defining one symbol with multiple values. The compiler emits a separate specialization (compiled QPC variant) per value. Common pattern: LLM prefill/decode specialization on `seq_len`. |
| `-dynamic-shape-input=<input,...>` | Inform the compiler that the listed inputs have dynamic shape (compile a graph that admits multiple shapes at runtime). |
| `-batchsize=<N>` | Override the model's batch dimension and recompile the graph at batch N. Input data must match the resulting batch size. **Only supported for models with a single input and single output** — this is a documented limitation; the tool does not reject it for multi-IO models but behavior is undefined. Prefer `-onnx-define-symbol=batch_size,N` for multi-IO models or when the ONNX has a symbolic batch dim. For tuning guidance see `cloud-ai-performance-optimization` → `references/batch_size_tuning.md`. |
| `-stats-batchsize=<N>` | Normalizes reported performance statistics to per-inference only. **Does not change the compiled graph.** Auto-detected from the graph when unset. Never a substitute for `-batchsize`. |

### 1.3 Profile-Guided Quantization (PGQ)

PGQ is a two-step process:

1. **Profile generation** (with calibration data):

   ```bash
   /opt/qti-aic/exec/qaic-compile \
     -m=./model.onnx \
     -aic-batch-json-input=calibration_inputs.json \
     -dump-profile=pgq_profile.yaml \
     -quantization-schema-activations=symmetric_with_uint8 \
     -quantization-calibration=KLMinimization
   ```

2. **Compile with profile**:

   ```bash
   /opt/qti-aic/exec/qaic-compile \
     -m=./model.onnx \
     -aic-batch-json-input=./inputs.json \
     -load-profile=pgq_profile.yaml \
     -quantization-schema-activations=symmetric_with_uint8 \
     -quantization-precision=Int8 \
     -quantization-precision-bias=Int32 \
     -aic-hw -aic-hw-version=ai100 -aic-num-cores=4 \
     -aic-binary-dir=./model_int8_qpc
   ```

---

## 2. Input/Output & Data Handling Flags

### 2.1 Input Specification

| Flag | Description |
|---|---|
| `-aic-batch-json-input=<path>` | JSON file describing input buffers (paths, shapes, datatypes) |
| `-input-path=<path>` | Root path for inputs, used with `-aic-batch-json-input` |
| `-use-random-input-data=<type>` | Generate random inputs (`gaussian`, `uniform`) when real data is not provided |
| `-num-iter=<N>` | Number of iterations during compile-time profiling (PGQ) |
| `-write-output-dir=<path>` | Directory for dumping outputs during profiling or debug |

Notes:

- Use **real** calibration data with `-aic-batch-json-input` for PGQ whenever possible.
- `-use-random-input-data=gaussian` can be used for quick tests but generally does not mimic real data distributions and may reduce INT8 accuracy.

### 2.2 Output & Debugging

| Flag | Description |
|---|---|
| `-output-node-name=<name,name,...>` | Save intermediate layer outputs (comma-separated list) |
| `-write-output-dir=<path>` | Directory to write outputs when using `-output-node-name` |
| `-aic-perf-warnings` | Print performance warnings during compilation |
| `-aic-perf-metrics` | Print compiler performance metrics |

Using `-output-node-name` prevents some fusions and may require regenerating the PGQ profile; use it only for debugging.

---

## 3. Quantization & Mixed Precision Flags

### 3.1 Quantization Schema & Calibration

| Flag | Description |
|---|---|
| `-convert-to-quantize` | Enable quantization transformation (deprecated in some flows; use PGQ) |
| `-quantization-schema-activations=<schema>` | Activation quantization schema. Valid: `asymmetric`, `symmetric`, `symmetric_with_uint8` (default), `symmetric_with_power2_scale`. |
| `-quantization-schema-constants=<schema>` | Weight/constant quantization schema. Same valid values as activations. |
| `-quantization-precision=<type>` | Main quantization type. `Int8` is currently the only supported value. |
| `-quantization-precision-bias=<type>` | Bias precision: `Int8` or `Int32` (default). |
| `-quantization-calibration=<method>` | Calibration. Valid: `None` (default — MinMax), `KLMinimization`, `KLMinimizationV2`, `Percentile`, `MSE`, `SQNR`. |
| `-percentile-calibration-value=<float>` | Percentile value used with `-quantization-calibration=Percentile`. Range 90–100; default 100. |
| `-num-histogram-bins=<N>` | Number of histogram bins per node during profiling (default 512). |
| `-enable-rowwise` | Enable row-wise quantization for FullyConnected and SparseLengthsSum ops. |
| `-enable-channelwise` | Enable per-channel quantization for Convolution ops. |
| `-dump-profile=<file>` | Dump PGQ profile |
| `-load-profile=<file>` | Load PGQ profile |
| `-external-quantization=<file>` | Use external quantization profile (e.g., from AIMET/QAT) |

### 3.2 Mixed Precision

| Flag | Description |
|---|---|
| `-node-precision-info=<yaml>` | Per-node instance precision overrides (e.g., FP16 in an INT8 model) |
| `-keep-original-precision-for-nodes=<op_kind>` | Keep all nodes of given kind in original precision (often FP32) |
| `-execute-nodes-in-fp16=<op_kind>` | Execute all nodes of given kind in FP16 in a quantized model |

Constraints:

- When using `-node-precision-info`, always generate the PGQ profile **with** this flag and load with the same flag.
- Do not specify the same op in both `-keep-original-precision-for-nodes` and `-execute-nodes-in-fp16`.

### 3.3 Stateful & Embedding Flags

For models with persistent state (e.g. KV cache) or external embedding tables. For LLM-specific compilation workflows, see the LLM onboarding skill.

| Flag | Description |
|---|---|
| `-retained-state[=<bool>]` | Allocate persistent buffers across inference calls for tensors with matching `<name>` / `<name>_RetainedState` input/output pairs (e.g. KV cache: `cache0` / `cache0_RetainedState`). |
| `-load-embedding-tables=<zipfile>` | Load embedding tables from a zip file at runtime instead of baking into the QPC. Removes those constants from the model. PyTorch only. Cannot combine with `-convert-to-quantize` or `-dump-profile`. |
| `-dump-embedding-tables=<zipfile>` | Extract embedding tables from a PyTorch model into a zip file for runtime loading. |

---

## 4. Performance & Partitioning Flags

### 4.1 Core Utilization & Tiling

Flag reference. For tuning workflow (which to set, in what order, when to escalate), see the performance-optimization skill.

| Flag | Description |
|---|---|
| `-aic-num-cores=<N>` | Number of AI cores to use (default: 1). |
| `-mos=<N>` | Max out-channel split (weight partition factor across cores). Value must be ≤ `-aic-num-cores`. Omit for compiler heuristic - the heuristic selection scales with core count and model structure. Accuracy is not guaranteed across values; see `cloud-ai-accuracy-tuning` [Static-shape MOS control](../cloud-ai-accuracy-tuning/references/output-comparison.md#static-shape-mos-control). |
| `-ols=<N>` | Overlap split factor (range 1–32). Raises the minimum tile count in the intra-core size-split pass — most impactful on D32 conv/pool ops. Raises peak VTCM use. Omit `-ols` for the compiler heuristic; `-ols=1` is an explicit candidate. `-aic-enable-depth-first` forces OLS to 1 internally. Tuning workflow in the performance-optimization skill. |
| `-mdts-mos=<N>` | Weight splitting across multi-device tensor slices (MDP/TS). Complements `-mos`: `-mos` partitions weights across cores within a slice, `-mdts-mos` partitions across slices. Value must be ≤ number of tensor slices. Omit for compiler heuristic. |
| `-vtcm-working-set-limit-ratio=<r>` | Maximum fraction of VTCM any single instruction may use (range 0–1, default 1.0). Reduce (e.g. 0.25) to prevent over-commitment when multiple ops compete for VTCM. |
| `-size-split-granularity=<KiB>` | Maximum tile size target for ops too large for VTCM, in **KiB** (range 512–2048; e.g. `1024` = 1 MiB). |
| `-allocator-dealloc-delay=<N>` | Increase buffer lifetimes (range 0–10, e.g. 1) to reduce false dependencies. |
| `-multicast-weights` | Reduce DDR bandwidth by loading weights used on multiple cores once and multicasting via NoC. Default-on. |
| `-mxfp6-matmul` | Compress constant MatMul weights to MXFP6 E2M3 to reduce DDR traffic at the expense of slightly more compute. Effective for DDR-bound MatMul-heavy models (transformer architectures, large FC layers). |
| `-aic-enable-depth-first[=<bool>]` | Enable depth-first scheduling — reorders ops to reuse VTCM tiles before eviction. **Forces OLS to 1.** |
| `-aic-depth-first-mem=<N>` | Depth-first memory budget. Must be **8 or 32** (MB); used with `-aic-enable-depth-first`. |

Example:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 -aic-num-cores=14 \
  -mos=2 -ols=2 -multicast-weights \
  -aic-binary-dir=./model_fp16_tuned \
```

### 4.2 Single Device Partitioning (SDP)

SDP partitions the network into subgraphs mapped to core clusters that execute in a pipeline.

| Flag | Description |
|---|---|
| `-sdp-cluster-sizes=x,y[,z,w]` | Cluster sizes per partition (sum must equal `-aic-num-cores`) |

Examples:

- 14-core dual-cluster: `-sdp-cluster-sizes=7,7`
- 16-core quad: `-sdp-cluster-sizes=4,4,4,4`

Per-cluster MOS example:

```bash
-mos=1,4 -sdp-cluster-sizes=7,7
```

### 4.3 Multi-Device Partitioning (MDP)

MDP distributes a graph across multiple Cloud AI SoCs (devices). For the conceptual overview of MDP and Tensor Slicing, see `cloud-ai-architecture` `references/cloud_ai_soc_architecture.md` §7. For LLM-specific TS sizing examples (dense, MoE), see `cloud-ai-llm-onboarding`.

| Flag | Description |
|---|---|
| `-mdp-dump-partition-config=<file>` | Emit a partition template JSON describing all nodes on one device |
| `-mdp-load-partition-config=<file>` | Load a user-edited partition JSON and compile for MDP |
| `-allow-mxint8-mdp-io` | Allow MXINT8 compression on inter-device links (may impact accuracy) |

#### Partition Config Format

The MDP config is a JSON file passed via `-mdp-load-partition-config`. It declares which SoCs participate and how many cores each uses.

**Example — TS4 (4 SoCs = 1 Ultra card):**

```json
{
    "connections": [
        {
            "devices": [0, 1, 2, 3],
            "type": "p2p"
        }
    ],
    "partitions": [
        {
            "name": "Partition0",
            "devices": [
                {"deviceId": 0, "numCores": 16},
                {"deviceId": 1, "numCores": 16},
                {"deviceId": 2, "numCores": 16},
                {"deviceId": 3, "numCores": 16}
            ]
        }
    ]
}
```

**Reading hardware requirements from a config:** Count entries in `partitions[0].devices` for the SoC count; divide by 4 for the Ultra card count.

| `devices` count | TS level | Ultra cards |
|---|---|---|
| 1 | TS1 (no MDP) | ¼ card |
| 2 | TS2 | ½ card |
| 4 | TS4 | 1 card |
| 8 | TS8 | 2 cards |
| 16 | TS16 | 4 cards |

> **TS16 (16 SoCs = 4 Ultra cards) is the maximum supported configuration.**

#### Workflow

1. Dump a template:

   ```bash
   /opt/qti-aic/exec/qaic-compile \
     -m=./model.onnx \
     -mdp-dump-partition-config=partition.json
   ```

2. Edit `partition.json` to split `nodeList` across partitions and set `connections`.
3. Compile with `-mdp-load-partition-config=partition.json`:

   ```bash
   /opt/qti-aic/exec/qaic-compile \
     -m model.onnx \
     -aic-num-cores=16 \
     -mdp-load-partition-config=partition.json \
     -aic-binary-dir=output_dir/
   ```

For QEfficient-driven LLM workflows, the MDP config is auto-generated from the `mdp_ts_num_devices` parameter — see `cloud-ai-llm-onboarding`.

---

## 5. Custom I/O and Custom Ops

### 5.1 Custom I/O (`-custom-IO-list-file`)

Custom I/O lets you override input/output layouts and datatypes.

| Flag | Description |
|---|---|
| `-custom-IO-list-file=<yaml>` | Custom I/O configuration (layout, precision, scale/offset per I/O) |
| `-dump-custom-IO-config-template=<file>[,<profile>]` | Dump a template YAML, optionally with scale/offset from PGQ profile |

The YAML file specifies, per I/O:

- `IOName`: model input/output name
- `Layout`: e.g., `NCHW` or `NHWC`
- `Precision`: `float`, `float16`, `int8`, `mxint8` (for some LLM states)
- `Scale` and `Offset` for `int8` where needed

### 5.2 Custom Ops (`-register-custom-op`)

| Flag | Description |
|---|---|
| `-register-custom-op=<config-file>` | Register a custom op package (domain/opType → kernel mapping) |

Use this to supply kernels for non-standard ONNX/TF operators.

---

## 6. Diagnostics & Debugging

| Flag | Description |
|---|---|
| `-v`, `-vv`, `-vvv` | Verbose logging from `qaic-compile` (increasing levels). |
| `-aic-perf-warnings` | Print performance warnings during compilation. |
| `-aic-perf-metrics` | Print compile-time performance metrics (VTCM peak, DDR traffic estimate, multicast counts, MAC counts) into compile.log. |
| `-aic-precision-warnings[=<bool>]` | Warnings on precision decisions that may impact accuracy. |
| `-time-passes` | Print compile-time statistics (per-pass timing). |
| `-enable-debug[=<bool>]` | Enable compiler debug mode. |
| `-compile-threads=<N>` | Parallel compilation threads (default: host logical cores). |
| `-profiling-threads=<N>` | Threads for PGQ profile generation (default 1). |
| `-stats-level=<N>` | Embed a runtime stats buffer in the QPC. Off by default. See level table below. |
| `-ddr-stats` | Embed runtime DDR instrumentation in a diagnostic QPC. Use it with `-stats-level=80`, then collect `raw_device_stats` through Batch JSON and inspect the graph-level runner metrics if they are emitted. Distinct from `-aic-perf-metrics`, which produces compile-time DDR estimates in compile.log. Do not use the diagnostic QPC for throughput benchmarks. |
| `-aic-pmu-recipe=<recipe>` | Built-in PMU recipe: `AxiRd`, `AxiWr`, `AxiRdWr`, `KernelUtil`, `HmxMacs`. Embeds PMU counters in the runtime stats stream. |
| `-aic-pmu-events=<event,event,...>` | Track up to 8 NSP PMU events. Event IDs are hex (e.g. `3F,70,200`). |

#### `-stats-level` reference

Levels are cumulative; higher levels imply all lower-level data plus more detail and more runtime overhead.

| Level | Detail |
|---|---|
| 0 | Disabled (default — no stats buffer embedded) |
| 10 | Minimal per-activation stats |
| 20 | Minimal per-inference stats |
| 30 | (reserved — above this requires an internal stats buffer with greater overhead) |
| 40 | Total time per inference |
| 50 | Additional per-inference stats |
| 60 | Additional per-inference stats |
| 70 | Total time per operation (required for `qaic-opstats`) |
| 80–100 | Additional per-operation stats |

For per-op profiling with `qaic-opstats`, use level 70 or higher. Host latency collection uses the runner's `latency` profiling type and does not require a stats-level buffer.

#### Compile-time signals to inspect

Two distinct sources of diagnostic information:

- **Compile-time** (compile.log): produced by `-aic-perf-metrics`. Static estimates of VTCM use, DDR traffic, multicast, MAC counts.
- **Runtime** (stats stream extracted by `qaic-runner` and `qaic-opstats`): produced by `-stats-level` (and optionally `-ddr-stats`, `-aic-pmu-*`). Actual measured per-operation cycles and optional graph-level DDR or PMU metrics. Confirm the available metric source in the SDK used for the diagnostic QPC before making a per-core DDR claim.

Inspect the compile.log signals from `-aic-perf-metrics`:

```bash
/opt/qti-aic/exec/qaic-compile ... -aic-perf-metrics -aic-perf-warnings 2>&1 | tee compile.log
grep -E "(VTCMMaxSize|DDR traffic|constant only|MC  traffic|barriers|Multicasts|WARNING|Not VTCM|operation size histogram)" compile.log
```

Key things to check:

- `AIC HMX MAC estimate` / `AIC non-HMX MAC estimate` — non-HMX should be 0; non-zero means Conv/MatMul ops missed HMX. The MAC estimate covers matrix ops only; for HVX op timing use runtime profiling.
- `barriers` per core — should be 0; non-zero means inter-core sync stalls.
- Per-core `DDR traffic write` and `Spilling buffer ... to DDR` / `BAR fatness` warnings — actionable spill signals.
- `Not VTCM` in multicast counts — non-zero means weight multicasts are spilling out of VTCM.
- `constant only` fraction of DDR reads — if ~95%+ of DDR reads are constants (weights), the model is weight-read-bound.
- `MC traffic write` per core — actual multicast bandwidth used; compare before/after `-multicast-weights`.
- `VTCMMaxSize` — scratchpad high-water mark for the compiled target. It is capped by target VTCM capacity; excess working set appears as DDR write/spill warnings or runtime `DMAIssue`, not as `VTCMMaxSize` exceeding the architecture maximum. Do not report it as a pass/fail "within limit" result.

For runtime profiling, see the `qaic-runner` skill.

---

## 7. Utility Flags

**Pre-/post-processing (PPP) default:** element-wise ops (quantize/dequantize, type casts such as FP32↔FP16) run on the host CPU; NHWC↔NCHW transpose and D32 format conversion run on the device. This split is the recommended configuration for most workloads — leave it at default unless there is a specific reason to change it.

> PPP ops are tensor-level boundary operations, not application-level preprocessing (image resize, crop, normalization — those are handled in user code before passing inputs to the model).

| Flag | Description |
|---|---|
| `-host-preproc` | Additionally moves NHWC↔NCHW transpose and D32 format conversion to the host CPU (these run on device by default). Rarely beneficial — increases host CPU load and changes the compiled binary; validate for accuracy impact if used. |
| `-aic-preproc` | Moves element-wise ops (quantize/dequantize, type casts) from host to device (these run on host by default), so all PPP runs on device. **Incompatible with `-network-specialization-config` and `-dynamic-shape-input`** — combining produces a compilation error. |
| `-operators-supported=<type>` | Write the supported-operator list for a model type (`onnx`, `tensorflow`, `pytorch`, `caffe`, or `caffe2`) to `<type>SupportedOperators.txt`. This is not a model compilation or compatibility test. |
| `-version`, `--version` | Print the QAic Graph API version and exit. |
| `-version-extended` | Print the version with compiler SHA. |
| `-compiler-help`, `--compiler-help` | List compiler-specific help options. |

---

## Related Skills

- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Go here for the end-to-end workflow: ONNX export, graph cleanup, and the quickstart compile command before diving into this flag reference.
- **Cloud AI 100 – Accuracy Tuning (FP16 & INT8)** (`cloud-ai-accuracy-tuning` skill) — Go here for the workflow around `-load-profile`, `-node-precision-info`, and `-quantization-*` flags covered in this guide.
- **Cloud AI 100 – Performance Optimization & Runtime** (`cloud-ai-performance-optimization` skill) — Go here for guidance on *which* performance flags to use and in what order; this guide provides the flag details.
- **Cloud AI 100 – SoC Architecture** (`cloud-ai-architecture` skill) — Consult to understand the hardware rationale behind flags like `-mos`, `-ols`, `-multicast-weights`, and `-aic-num-cores`.
- **Cloud AI 100 – `qaic-runner` Detailed Tool Guide** (`cloud-ai-qaic-runner` skill) — Use to run and benchmark the QPC binaries produced by `qaic-compile`.

## Reference Files

- **[ONNX Operator Support Table](references/cloud_ai_onnx_operator_support.md)** — Full supported operator table (SDK 1.21, opset 7+), constant tensor requirements per operator, unsupported dynamic patterns and their fixes, and ONNX symbol mapping for dynamic shapes.
