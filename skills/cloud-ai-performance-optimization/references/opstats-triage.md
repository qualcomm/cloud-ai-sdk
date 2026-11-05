# Performance Opstats Triage

Use this reference after a clean FP16 baseline and configurator sweep show a
performance miss or plateau. It collects static compiler signals and runtime
opstats, then maps the signals to bounded diagnoses.

## Diagnostic QPC and collection

**REQUIRED SUB-SKILL:** Use `cloud-ai-qaic-runner` for the runner `IO-files`
JSON schema. Do not pass the compiler calibration `{"inputs": [...]}` format
to `qaic-runner` or `qaic_profile.sh`.

Never benchmark a QPC compiled with `-stats-level`. If the production QPC lacks
diagnostic stats, compile a separate `_diag` QPC with `-stats-level=80 -ddr-stats`.
Keep the clean production QPC unchanged. The metrics and stats QPCs must use
the same winner model revision, input/output shapes, batch, core count, and
MOS/OLS or other compiler tuning flags. Collect one diagnostic QPC instance
with `-a 1`. Use one metrics QPC for `-aic-perf-metrics` and a separate stats
QPC such as `qpc_winner_diag` for runtime collection. Never pass a metrics-only
QPC to `qaic_profile.sh`.

The metrics compile may include `-aic-perf-metrics -aic-perf-warnings` to collect
static compiler signals. Do not combine `-stats-level` with those metrics, and
do not use either diagnostic binary for the configurator or benchmark.
Capture the large compiler output in `compile.log`:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx -convert-to-fp16 -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=<winner_cores> -aic-perf-metrics -aic-perf-warnings \
  -aic-binary-dir=./qpc_winner_metrics 2>&1 | tee compile_metrics.log
```

Copy the recorded winner compile command. Replace its metrics-only flags with
the diagnostic instrumentation and a new output directory:

```bash
/opt/qti-aic/exec/qaic-compile \
  <recorded winner arguments> \
  -stats-level=80 -ddr-stats -aic-binary-dir=./qpc_winner_diag \
  2>&1 | tee compile_diag.log
```

Use the selected model, shapes, batch, cores, MOS/OLS, and other production
compile arguments in place of `<recorded winner arguments>`.

From the model workspace, invoke the helper at the loaded skill path. Use new
empty output directories; the helper rejects non-empty directories rather than
deleting their contents.

```bash
bash <path-to-cloud-ai-performance-optimization>/scripts/qaic_profile.sh \
  --qpc ./qpc_winner_diag \
  --batch-json ./inputs.json --device 0 \
  --stats-dir ./qpc_winner_diag_stats --out-dir ./qpc_winner_diag_opstats
```

The helper collects one diagnostic QPC instance with `-a 1`; do not substitute
the throughput row's instance count.

Use `--no-collect` only to re-analyse existing stats. Record both sources before
recommending a flag:

```text
compile.log: HMX MAC, non-HMX MAC, DDR read/write, barriers, multicast, VTCMMaxSize
qaic_profile.sh: HVX%, HMX%, DMAIssue%, bottleneck, top HVX ops, multicast presence
```

Use each diagnostic source for its own question:

- Queue, host submission, and host processing: legacy `-i` with `-l -s`.
- Host p50/p99: Batch JSON with `--aic-profiling-type latency --time`.
- Per-operation cycles and thread utilization: Batch JSON with `raw_device_stats`, then `qaic-opstats`.
- Static MAC, VTCM, multicast, and estimated DDR: `-aic-perf-metrics` compile log.
- Runtime DDR: per-core values from the decoded opstats summary and graph-level
  runner metrics when the same diagnostic QPC emits them. Decode with
  `qaic-opstats` from the matching SDK release that compiled the diagnostic QPC.

Compile-log DDR totals are aggregate estimates. Cross-check them with runtime
`DMAIssue`, decoded per-core DDR, and, when available, same-QPC graph-level DDR
metrics before calling a model DDR-bound. `VTCMMaxSize` is context, not a pass/fail limit. Frequency
variation between runs is normal measurement jitter unless runtime evidence shows
a real device problem. Diagnostic artifacts do not replace a clean production
latency or throughput run.

## Signal-to-diagnosis table

| Signal | Diagnosis | Next bounded check |
|---|---|---|
| `AIC non-HMX MAC estimate` > 0 | Conv/MatMul missed HMX | Identify the operator and check HMX geometry in the compiler/operator-support references. |
| `DDR traffic write`, spill warnings, or DDR-suffix HVX ops (for example, `aicnormalization HVX DDR` or `aicupsamplebilineard32 HVX DDR`) | Activations spill or round-trip through DDR | Inspect adjacent graph nodes for a fusion opportunity; compare runtime DMA. |
| High `DMAIssue` plus constant-dominant DDR reads | Weight streaming is limiting throughput | Verify multicast, then compare default MOS and higher MOS only when the DDR conditions hold. |
| `DMAIssue` > 10% plus activation spill, with `constant only` DDR traffic below roughly 50% | Activation movement is material and not primarily weight-read-bound | Test depth-first 8 MB/32 MB or a measured fusion; do not choose depth-first when constant-only DDR reads dominate. |
| `DMAIssue` < 5% with no DDR-suffix ops | DDR is unlikely to be the limiting resource | Do not raise MOS or add depth-first speculatively; inspect compute/synchronization signals. |
| High `constant only` DDR fraction | Weight reads dominate | Start MOS at the compiler default and compare lower splitting before increasing it. |
| Non-zero `Not VTCM` multicast count | Multicast destinations spill | Check duplicated branch inputs and per-core weight tiling. |
| `barriers` or uneven core utilization | Synchronization/load imbalance | Inspect partitioning and MOS; do not infer cause from aggregate frequency. |
| `sync HVX` in top-20 | HMX-to-HVX pipeline is serialized | Trace the HMX output to its immediate HVX consumer for Cast/Quantize/Reshape or coarse tiling. |
| `sync HMX` in top-20 | HVX-to-HMX wait or pipeline latency | Inspect HVX-heavy ops, DDR-suffix ops, and `sync HVX` independently. |
| `aicconverttod32`/`fromd32` dominates HVX | Format conversion cost | Compare INT8 only when explicitly authorized and accuracy-gated. |
| `sigmoid`, `elementmul`, or `softmax` dominates HVX | HVX-only activation cost | Check whether INT8 LUT fusion applies; otherwise treat it as architecture-bound. |
| `aicconvhvx` plus dilation warnings | Conv is HMX-ineligible | Fix geometry at model/export level or evaluate a measured custom HVX kernel. |
| `aicgatherelements HVX TCM (ui8)` dominates | INT8 gather is slower on this backend | Compare INT8 and FP16 end-to-end; revert INT8 if it regresses. |

`aiccopytovtcm` indicates weight movement; DDR-suffix HVX indicates activation
movement. Do not use one as a substitute for the other. If HMX utilization is
low while the HMX MAC estimate is large and no listed signal explains the cycle
distribution, inspect meaningful non-sync top-20 ops, graph connectivity, and
`pcycles` (pipeline cycles) before choosing a transform; use `ucycles` for
wall-clock inference time and `pcycles` for per-op structural compute cost.

`sync HVX` and `sync HMX` are independent signals. Investigate both when both
appear; never collapse them into a single percentage or one universal flag.

If an architecture-inherent HVX op remains the bottleneck after supported graph
alternatives are checked, a custom op can still be appropriate for a measured
fusion or specialized attention/composite block. Custom kernels run on HVX and
cannot remove the hardware-bound HMX/HVX ceiling; validate interpreter output,
hardware accuracy, and end-to-end performance before retaining one.

## One-pass anomaly scan

After the structured diagnosis, make one pass over unexplained top-20 ops and
compile-log fields. Record op name, cycle share, and why it is unexplained. Also
note non-zero `Not VTCM`, MCID DDR buy-down, barrier asymmetry, unexpected
warnings, non-HMX MACs, and disproportionate conversion/gather costs. Do not
start recursive investigations; place candidates into the ranked optimization
plan and stop the scan.

## Artifact

Record the compile-log fields, profiler output, diagnosis, selected flag or graph
candidate, before/after throughput or latency, and the production versus
diagnostic QPC paths. Remove diagnostic flags from production builds.
