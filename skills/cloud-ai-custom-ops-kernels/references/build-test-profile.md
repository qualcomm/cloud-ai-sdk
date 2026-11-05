# Building, Testing, and Profiling Custom Ops

## Contents

- Build and compile
- Interpreter, hardware, and runner testing
- Host and device profiling
- Opstats interpretation
- Testing best practices

## Build and compile

### 1. Build the host-side library

```bash
g++ -shared -std=c++11 -fPIC \
    -o CustomOp_Package/MyOp/src/myop_lib.so \
    CustomOp_Package/MyOp/src/myop_functions.cpp \
    CustomOp_Package/MyOp/src/myop_interpreter.cpp \
    -I/opt/qti-aic/dev/inc
```

### 2. Generate the ONNX model

```bash
python3 onnx_torch/gen_myop.py
# Produces: custom.onnx, input_data.raw, reference_output.raw, aic_batch_io.json
```

### 3. Compile to QPC

```bash
/opt/qti-aic/exec/qaic-compile \
    -m=custom.onnx \
    -register-custom-op=CustomOp_Package/custom_op_config.yaml \
    -aic-num-cores=1 \
    -aic-hw -aic-hw-version=ai100 \
    -input-path=./ \
    -aic-batch-json-input=./aic_batch_io.json \
    -aic-binary-dir=hw_bin_dir
```

Output: `hw_bin_dir/` — pass this directory to `Qpc::Factory()`.

### Deprecated flags (do not use)

| Old | New |
|-----|-----|
| `-aic-hw-version=2.0` | `-aic-hw-version=ai100` |
| `-json-input-file=` | `-aic-batch-json-input=` |
| `-compile-only` | Unsupported by `qaic-compile`; omit it. |

---

## Testing

### Interpreter (CPU reference)

```bash
/opt/qti-aic/exec/qaic-compile \
    -m=custom.onnx \
    -register-custom-op=CustomOp_Package/custom_op_config.yaml \
    -run-on-interpreter \
    -input-path=./ \
    -aic-batch-json-input=./aic_batch_io.json \
    -write-output-dir=out/
# Output filename: out/<output_name>_out_0_0.raw  (interpreter naming convention)
python3 check_all_close.py reference_output.raw out/<output_name>_out_0_0.raw
```

### Hardware

```bash
/opt/qti-aic/exec/qaic-runner \
    -t hw_bin_dir \
    -a 1 -S 1 \
    -n 1 \
    --aic-batch-json-input aic_batch_io.json \
    --write-output-dir=out/
# Output filename: out/<output_name>-activation-0-inf-0.bin  (qaic-runner naming convention)
python3 check_all_close.py reference_output.raw out/<output_name>-activation-0-inf-0.bin
```

Always test on the interpreter before hardware. The interpreter runs on the
host CPU and supports `printf`, file I/O, and standard library calls, making it
the fastest place to catch shape, dtype, and reference mismatches. A bug that
passes the interpreter but fails on hardware is much harder to diagnose.

Generate reference outputs in the `gen_*.py` ONNX model script from a pure
Python implementation and keep that reference, the C interpreter, and the AIC
kernel in sync. A Python/C discrepancy indicates a porting bug; a C/AIC
discrepancy indicates an HVX bug.

Use numerical comparison with tolerance rather than exact byte comparison.
Integer rounding differences between HVX vectorized and scalar paths are
expected; MAE < 0.11 is acceptable for uint8 data. Use
`check_all_close.py` or an equivalent comparator.

### Standalone runner

For integration testing outside the full pipeline, build a standalone C++ runner
that loads the QPC directly via `QAicApi`:

```cpp
auto ctx    = qaic::rt::Context::Factory(nullptr, {0});
auto qpc    = qaic::rt::Qpc::Factory("hw_bin_dir");
auto prog   = qaic::rt::Program::Factory(ctx, 0, "test", qpc);
prog->load(); prog->activate();

auto infvec = qaic::rt::InferenceVector::Factory(
    qpc->getBufferMappings(),
    qaic::rt::InferenceVector::DataSourceType::USER_BUFFERS);
auto exec = qaic::rt::ExecObj::Factory(ctx, prog);
exec->setData(infvec);

// Write inputs
float *in = (float *)infvec->getVector()[0].buf;
// ... fill in ...

// Register the current buffers, then run synchronously.
exec->setData(infvec);
exec->run(infvec);

// Read outputs
float *out = (float *)infvec->getVector()[1].buf;
```

See `<opname>_runner/` in your op package for a complete example.

---

## Profiling

### Host-side throughput stats (`qaic-runner` default output)

Every `qaic-runner` invocation prints a summary line:

```
 ---- Stats ----
InferenceCnt 20 TotalDuration 13544us BatchSize 1 Inf/Sec 1476.669
```

**What to look for:**
- `TotalDuration` / `InferenceCnt` = average end-to-end latency per inference
  (host enqueue → host completion, includes PCIe DMA both ways).
- `Inf/Sec` = aggregate QPC executions/sec. For a static-batch QPC, multiply
  by compiled batch before comparing application throughput and end-to-end
  latency budgets.

### Device-side phase breakdown (`--aic-profiling-type latency`)

```bash
/opt/qti-aic/exec/qaic-runner \
    -t hw_bin_dir \
    -a 1 -S 1 \
    -n 20 \
    --aic-batch-json-input aic_batch_io.json \
    --aic-profiling-type latency \
    --aic-profiling-out-dir profiling_out/
```

Output file: `profiling_out/aic-profiling-program-0-latency.txt`

```
Sample ,devExecUs ,devInWaitTimeUs ,devOutWaitTimeUs ,hostRoundTripUs ,preProcTimeUs ,postProcUs
0      ,N/A       ,N/A             ,N/A              ,843.000         ,515.024       ,576.875
```

**What to look for:**

| Field | Meaning | Action if high |
|-------|---------|----------------|
| `preProcTimeUs` | Host time preparing input (conversion loops, memcpy into DMA buffer) | Prefer INT8 plus `memcpy`; otherwise optimize host conversion and measure |
| `postProcUs` | Host time reading output (conversion loops, memcpy from DMA buffer) | Same as above |
| `hostRoundTripUs` | Wall time from enqueue to completion, excluding pre/post | PCIe DMA bottleneck — reduce buffer size or pipeline |
| `devInWaitTimeUs` | Time device waited for input DMA to complete | Large input buffer or PCIe contention |
| `devOutWaitTimeUs` | Time device waited for output DMA to complete | Large output buffer or PCIe contention |

If `preProcTimeUs + postProcUs` dominates: the bottleneck is host-side data
preparation, not the device.  If `hostRoundTripUs` dominates: the bottleneck
is PCIe DMA or NSP compute — use `stats-level` profiling to distinguish them.

### Per-operation cycle counts (`-stats-level=70` + `qaic-opstats`)

Recompile with `-stats-level=70` to embed a stats buffer in the QPC:

```bash
/opt/qti-aic/exec/qaic-compile \
    -m=custom.onnx \
    -register-custom-op=CustomOp_Package/custom_op_config.yaml \
    -aic-num-cores=1 -aic-hw -aic-hw-version=ai100 \
    -input-path=./ -aic-batch-json-input=./aic_batch_io.json \
    -aic-binary-dir=hw_bin_dir_stats \
    -stats-level=70
```

Run with `--aic-profiling-type raw_device_stats` to capture binary stats buffers:

```bash
mkdir -p profiling_out_raw
/opt/qti-aic/exec/qaic-runner \
    -t hw_bin_dir_stats \
    -a 1 -S 1 \
    -n 20 \
    --aic-batch-json-input aic_batch_io.json \
    --aic-profiling-type raw_device_stats \
    --aic-profiling-out-dir profiling_out_raw/
```

This writes one binary `.bin` file per inference to `profiling_out_raw/` and also
prints the aggregated text report to stdout:

```
--- Aggregated Device Metrics Report ---
Metric,                      Avg,           Min,           Max,           Std
ExecTimeUs_<GraphName>,            510.792,       496.927,       569.219,       17.830
InfPCycles_<GraphName>,            324300.250,    308065.000,    371547.000,    12802.972
EffectiveFrequencyMHz_<GraphName>, 635.357,       595.173,       652.830,       26.304
InputWaitTimeUs_<GraphName>,       515.146,       0.052,         9980.833,      2172.691
OutputWaitTimeUs_<GraphName>,      0.349,         0.260,         0.677,         0.111
```

The graph suffix is emitted by the QPC; use the same suffix when comparing
related metrics.

Post-process the binary buffers with `qaic-opstats` for a per-op breakdown:

```bash
mkdir -p opstats_out
/opt/qti-aic/exec/qaic-opstats \
    -q hw_bin_dir_stats/programqpc.bin \
    -i profiling_out_raw/ \
    --summary \
    -o opstats_out/
```

Sample `opstats_out/*.summary.txt`:

```
Specialization: QAicGraph

core  ucycles  ucycles(%)     pcycles  pcycles(%)  count      out KB  ucycles/KB  pcycles/KB  opdetails
c0      10180      98.78%         N/A         N/A      3         N/A         N/A         N/A  sync HMX
c0      10123      98.22%         N/A         N/A      1         N/A         N/A         N/A  sync DMAIssue
c0       5224      50.69%      161917      50.68%      1     2025.00        2.58       79.96  customop HVX CacheableDDR (i8, float)
c0       4857      47.13%         N/A         N/A      2         N/A         N/A         N/A  sync HVX
c0       3070      29.79%         N/A         N/A      2     2025.00        1.52         N/A  aiccopysamevtcm HVX CacheableDDR
c0       1910      18.53%         N/A         N/A      1     2025.00        0.94         N/A  aiccopysamevtcm DMAIssue CacheableDDR
c0        173       1.68%         N/A         N/A      2         N/A         N/A         N/A  barrier HVX
...

c0 total:  10306 ucycles  319485 pcycles (thread max)
c0    t0 HVX    10290 ucycles   318977 pcycles  51.51% utilization
c0    t1 HVX    10289 ucycles   318970 pcycles  52.43% utilization
c0    t2 HVX    10291 ucycles   318975 pcycles  51.12% utilization
c0    t3 HVX    10291 ucycles   318983 pcycles  52.55% utilization
c0    t4 HMX    10291 ucycles   318978 pcycles   1.83% utilization
c0    t5 DMAIssue 10306 ucycles 319485 pcycles   1.74% utilization
```

**What to look for:**

| Field | Meaning | Action if high |
|-------|---------|----------------|
| `ExecTimeUs_<GraphName>` (stdout) | Pure NSP kernel execution time (µs) | Vectorize with HVX; add thread parallelism |
| `InfPCycles_<GraphName>` (stdout) | NSP processor cycles per inference | Divide by `EffectiveFrequencyMHz_<GraphName> × 1e6` to get seconds |
| `EffectiveFrequencyMHz_<GraphName>` (stdout) | Measured NSP clock | Reference only — not actionable |
| `InputWaitTimeUs_<GraphName>` (stdout) | Time NSP waited for input DMA (avg) | High mean = large buffer; high variance (check Min vs Max) = PCIe contention |
| `customop … ucycles(%)` (summary) | Fraction of total inference time in your kernel | Should dominate; if low, overhead is in DMA copies or sync ops |
| `aiccopysamevtcm … ucycles` (summary) | Compiler-inserted DMA copies between memory tiers | See [Configuration and memory](configuration-and-memory.md) for placement interpretation |
| `tN HVX … utilization` (summary) | Per-thread HVX utilization | Low % = threads are stalled on memory; high % = compute-bound |
| Thread balance (summary) | `t0`–`t3` ucycles should be closely balanced | Large imbalance = uneven row distribution; check stride-partition logic |

For the meaning and placement action for `aiccopysamevtcm` entries, see
[Configuration and memory](configuration-and-memory.md). The `aiccopysamevtcm`
rows remain useful here as evidence when deciding whether a placement change
removed compiler-inserted staging.

Add `--trace` to also generate a Chrome-compatible JSON trace viewable in
`chrome://tracing` or Perfetto:

```bash
/opt/qti-aic/exec/qaic-opstats \
    -q hw_bin_dir_stats/programqpc.bin \
    -i profiling_out_raw/ \
    --summary --trace \
    -o opstats_out/
```

**Distinguishing PCIe vs compute bottleneck:**

- If `InputWaitTimeUs_<GraphName>` avg >> `ExecTimeUs_<GraphName>`: the NSP is idle
  waiting for data — PCIe input DMA is the bottleneck. Reduce buffer size
  (e.g. switch to INT8) or pipeline with the async queue API.
- If `ExecTimeUs_<GraphName>` >> `InputWaitTimeUs_<GraphName>`: the kernel itself is
  slow — apply HVX vectorization and thread parallelism.
- High `InputWaitTimeUs_<GraphName>` variance with a low minimum and much higher
  maximum suggests PCIe bus contention from other processes; compare against
  buffer size before changing the kernel.

Useful levels for custom op tuning:

| Level | What it adds |
|-------|-------------|
| 0 | Disabled (default) |
| 10 | Minimal per-activation stats |
| 20 | Minimal per-inference stats |
| 40 | Total time per inference |
| **70** | **Total time per operation — use this for kernel tuning** |
| 100 | Full per-operation stats (highest overhead) |

Levels are cumulative. Levels above 30 allocate an internal stats buffer at
compile time, which adds a small runtime overhead — use a separate
`hw_bin_dir_stats/` directory and keep the production QPC without stats.

Use `--aic-profiling-type raw_device_stats` when you want to post-process with
`qaic-opstats`. It writes the binary `.bin` buffers that `qaic-opstats`
requires. The SDK 1.23 `stats` profiling value is rejected; it is not an
alternative collection mode.
