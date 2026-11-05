---
name: cloud-ai-performance-optimization
description: Use when optimizing throughput or latency for a CNN, vision, text, or audio / ASR model on Cloud AI 100, tuning compiler flags, benchmarking a QPC, collecting opstats, or diagnosing runtime bottlenecks. Not for LLMs, VLMs, diffusion, embeddings, or rerankers.
---

# Cloud AI 100 Performance Optimization

## Purpose

Use this skill after a matched FP16 ONNX/QPC baseline exists and the user needs
measured throughput, latency, stability, or memory improvements. It owns the
optimization order and acceptance gates. It does not replace the compiler flag,
hardware, runner, accuracy, conversion, or custom-op references.

## When to use

- Throughput or latency misses a use-case target.
- A configurator sweep needs to select latency and throughput winners.
- Compiler metrics, `qaic-opstats`, DDR traffic, barriers, HMX/HVX utilization,
  or `DMAIssue` need diagnosis.
- A measured bottleneck justifies a flag, graph transform, precision change, or
  custom kernel.

Do not use this skill for LLM/VLM/diffusion/embedding/reranker workflows. Route
those tasks to the corresponding generative-model skill.

## Non-negotiable rules

1. Start from a clean FP16 baseline. INT8 calibration is user-requested only;
   performance dissatisfaction alone is not authorization.
2. Never benchmark or report production performance from a QPC compiled with
   `-stats-level`. Diagnostic QPCs use a separate `_diag` directory and never
   replace the clean production QPC.
3. Use the canonical `qaic_profile.sh` for opstats. Do not write replacement
   profiling scripts or substitute `grep`/`awk` parsing for its analysis.
4. Recommend only flags documented by this skill or `cloud-ai-qaic-compile`.
   The compiler skill owns flag semantics; this skill owns when to try them.
5. Diagnose from matched inputs, compile logs, runtime profiling, and repeated
   measurements. Do not tune from one noisy counter or aggregate DDR value.
6. Validate accuracy after every graph, precision, or custom-op change and run
   stability before accepting a production result.

## Decision map

| Need | Read |
|---|---|
| Configurator search, batch/core/instance selection | `references/qaic_model_configurator.md` |
| Batch-size export, stop signals, batch-specific throughput | `references/batch_size_tuning.md` |
| Opstats collection and signal diagnosis | `references/opstats-triage.md` |
| Graph checks, flag conditions, transforms, rollback | `references/graph-optimization.md` |
| Exact `qaic-compile` flags or operator support | `cloud-ai-qaic-compile` |
| Device topology and VTCM/HMX/HVX facts | `cloud-ai-architecture` |
| Runner command syntax and latency parsing | `cloud-ai-qaic-runner` |
| Accuracy, PGQ, mixed precision, or output mismatches | `cloud-ai-accuracy-tuning` |
| ONNX/export transforms | `cloud-ai-model-conversion` |
| Measured fused/specialized NSP kernel | `cloud-ai-custom-ops-kernels` |

## Bounded workflow

### Step 0: Establish the baseline

Confirm the final ONNX revision, conversion manifest, representative input, clean
FP16 QPC, compile command, device/QID, core count, batch, instance count, and
current accuracy result all match. Record the use-case target and whether the
objective is unloaded latency, loaded throughput, or both.

**First pass:** before a sweep, or for a no-sweep run, compile batch-1 FP16 at
the available core count with `-mos` and `-ols` omitted. This is the
compiler-heuristic baseline: `-mos=1` is a tuning candidate, and `-ols=1` is not
the baseline. Benchmark for 60 seconds; report a baseline, not a winner. Add
MOS or OLS only after their measured conditions are met.

### No-sweep onboarding baseline

Use this branch only when the coordinator sets `RUN_SWEEP=0`. When `RUN_SWEEP`
is unset, as in direct harness use, continue with the normal configurator workflow.
This branch replaces the configurator sweep and measured-candidate steps.

1. Compile one clean FP16 batch-1 QPC at the available core count, with `-mos`
   and `-ols` omitted for the compiler heuristic.
2. Run one 60-second capture:
   ```bash
   /opt/qti-aic/exec/qaic-runner -t ./<no_sweep_baseline_qpc>/ -i ./input.raw \
     --time 60 -a 1 -S 1 -d <QID>
   ```
3. Retain that QPC and record one AI100 production row. `Inf/Sec` from this
   capture is incidental output, not a throughput experiment.

Do not compile a second QPC, choose a throughput winner, run a multi-activation throughput pass, or add a second AI100 production row. When a sweep is requested, continue with Step 1.

Read the architecture reference before interpreting HMX/HVX, VTCM, barriers, or
NoC signals. Read the runner reference before writing a benchmark command.

### Step 1: Check graph and compile prerequisites

Read `references/graph-optimization.md`. Check channel/tile alignment, fused
BatchNorm, fusion-breaking Cast/Quantize/Transpose nodes, constant shape inputs,
bound symbolic dimensions, dynamic control flow, operator support, HMX geometry,
and the active VTCM working set. Fix export/conversion defects before tuning flags.

For transformer or attention models, read the
[transformer architecture reference](../cloud-ai-architecture/references/transformer_architecture.md)
in full. Complete its Transformer Optimization Report in triage order and include
the filled-in report with graph-optimization evidence. Before graph optimization
is complete, every row must be marked Applied, Not applicable, or Deferred with a
reason.

Independently of opstats, inspect the final fp16ready ONNX and count op types.
Record repeated per-block sequences, large intermediates that may materialize in
DDR, unusually frequent minor ops, and architecture-mismatch patterns. Add each
candidate to the ranking with known runtime evidence or `unknown`; do not apply
it until its condition is proven.

### Step 2: Run the baseline/configurator sweep

Use the exact invocation in `references/qaic_model_configurator.md`. Check NSP
availability first. The sweep must include:

- latency coverage: batch 1, one instance, powers-of-two core counts;
- throughput coverage: standard batch values for every device-filling core x
  instance row, where `cores × instances == NSP_TOTAL`, subject to export and
  compile-time shape constraints.

Normalize `cfg_out/results.csv`, record the lowest-latency and highest-throughput
rows, and explicitly compile production QPCs from those rows as
`qpc_latency_winner` and `qpc_throughput_winner`. Do not benchmark configurator
output directories as production binaries. Re-measure the latency winner with
the runner's latency workflow and record p50/p99.

The normal latency row is `batch=1`, `instances=1`. For throughput selection,
keep only rows where `cores × instances == NSP_TOTAL`. Exclude every partial-SoC
row (`cores × instances < NSP_TOTAL`) from throughput ranking even when its
reported throughput is higher. Select the maximum numeric `Throughput (inf/sec)`
among the full-SoC rows and preserve its `cores`, `batch`, and `instances` tuple.
If the results contain no full-SoC row, repair or rerun the sweep; never promote
a partial-SoC row as the throughput winner. Do not prune batch values merely
because batch 1 is easy to compile; use the batch-size reference when export
constraints require a batch-specific re-export.

If the throughput winner is at the top of the searched batch range, continue with
the batch-size reference's next-batch and plateau/latency-budget checks.

For unloaded/per-request latency benchmarking, use `-a 1`: each activation is an
independent inference pipeline, not a faster way to complete one request. For
the selected full-SoC throughput row, use `-a <instances>` so the runner uses
the same instance count as the selected row. `-S` is ExecObjs per activation;
use `-S 1` for standard onboarding measurements.
Report the core count, batch, instances, activation count, set size, warmup,
duration, input, QID, and selected latency/throughput metric with every result.
For every benchmark row, keep a result producer and unit ledger: configurator
`results.csv` throughput is completed inference units/sec; direct `qaic-runner`
`Inf/Sec` is QPC executions/sec and needs the compiled batch to derive completed
inference units/sec. Do not use runner `BatchSize` to infer compiled batch.

### Step 3: Collect diagnostic evidence

Read `references/opstats-triage.md`. Collect compile-log signals from the exact
selected configuration and runtime opstats from a matched diagnostic QPC. Keep
the clean production QPC for timing. Compile a separate `_diag` QPC from the
same model, shapes, batch, cores, and compiler tuning flags with
`-stats-level=80 -ddr-stats`; collect one diagnostic QPC instance with `-a 1`.
Never use it for production timing.

Do not proceed to flag recommendations until the evidence block contains HMX and
non-HMX MAC estimates, DDR read/write, barriers, multicast/VTCM context, HVX/HMX
utilization, `DMAIssue`, top ops, and the profiler bottleneck classification.

Treat `sync HVX` and `sync HMX` as independent diagnoses. Cross-check compile-log
aggregate DDR with runtime DMA and inspect CPU-partitioned/unsupported ops before
assuming a compiler flag can fix them.

### Step 4: Test measured candidates

Read `references/graph-optimization.md` and rank candidates backed by diagnostic
or Step 1 structural evidence. Implement only candidates whose condition is proven.
Use this order unless the evidence says to skip a candidate:

1. compiler heuristic MOS, then `-mos=1`, then higher MOS only for proven DDR/tiling pressure;
2. OLS only when relevant to size splitting, remembering depth-first forces OLS to 1;
3. depth-first only for activation-spill evidence, with the documented 8/32 MB budgets;
4. multicast for shared weights when it is not already active;
5. `-mxfp6-matmul` only for MatMul-heavy weight-DDR pressure;
6. SDP/MDP partitioning using the compiler reference;
7. ONNX/PyTorch transforms or a custom op only when graph evidence justifies them.

For every candidate record the condition, production compile directory, workload
configuration, before/after metric, accuracy result, and stability result. Run
latency and throughput winners separately when their rows differ.

Run the measured-candidate tuning pass separately for the latency winner and the
throughput winner whenever their selected rows differ. The latency pass keeps the
latency winner's cores, batch, instances, and input and judges candidates by its
latency target. The throughput pass keeps the throughput winner's own values and
judges candidates by its throughput target. If both winners are the same row, one
pass may serve both objectives only when it records both metrics separately.

MOS and OLS candidates must be benchmarked against each pass target metric. The
omitted-flag compiler heuristic is the first data point for each pass; do not copy
a MOS/OLS result from one workload to the other. Do not use a throughput-only
MOS/OLS sweep to select or label the latency winner. When an OLS relevance gate
does not apply (for example, depth-first forces OLS to 1), record that decision for
each pass instead of silently omitting the pass.

For MOS, benchmark in order:

1. Compiler heuristic - reuse the current pass production QPC with `-mos` omitted.
2. `-mos=1` - compile fresh from the current pass row values.
3. `-mos=2` - compile fresh from the current pass row values.
4. Higher MOS values if `-mos=2` improves the current-pass target metric.

For relevant D32 convolution or pooling models when depth-first is disabled,
benchmark OLS using the MOS winner:

1. Compiler heuristic - reuse the current pass production QPC with `-ols` omitted.
2. `-ols=1` - compile fresh from the current pass row values.
3. `-ols=2` - compile fresh from the current pass row values.
4. `-ols=4` - compile fresh from the current pass row values if `-ols=2` improves the current-pass target metric.

### Step 5: Validate, retain, or revert

After each accepted graph or precision change:

1. compare every output tensor with `cloud-ai-accuracy-tuning`;
2. compile the clean production QPC with the same workload settings;
3. rerun the relevant latency/throughput measurement;
4. run the 60-second stability test without diagnostic profiling flags;
5. retain only a measured improvement with passing accuracy and stability.

If the candidate fails, restore the previous production QPC and record the reason.
Do one bounded anomaly scan after the structured diagnosis; do not recurse through
every unexplained signal.

Use `--time 60 -a <throughput_activation_count> -S 1 -d <QID>` for the standard
throughput stability run. A higher set size is a separately requested concurrency
experiment.

For computer-vision serving, 10-30 fps is a starting reference range, not a
universal acceptance target. Use the user's target and leave headroom for any
multi-model pipeline or host-side preprocessing.

### Step 6: INT8 boundary

Enter INT8 only after explicit authorization. If validated FP16 already exceeds
the use-case target with meaningful headroom and has no memory-pressure problem,
record the decision and skip calibration. Otherwise route PGQ and precision policy
to `cloud-ai-accuracy-tuning`, then return here to benchmark the approved QPC.

## Required final artifacts

Record the target and workload conditions; baseline and selected QPC paths; exact
production flags; configurator CSV; p50/p99 and throughput; compile-log signals;
opstats output and diagnosis; candidate ranking; before/after measurements;
accuracy results; stability result; and the rollback point. Remove all diagnostic
flags from the production command. For transformer or attention models, include
the completed Transformer Optimization Report.

## Related references

- `references/qaic_model_configurator.md` - configurator invocation and result normalization.
- `references/batch_size_tuning.md` - batch-specific export, measurement, and stop criteria.
- `references/opstats-triage.md` - compile-log/runtime signal collection and diagnosis.
- `references/graph-optimization.md` - graph checks, candidate ordering, and validation.
- `cloud-ai-qaic-compile` - authoritative compiler flags and operator support.
- `cloud-ai-qaic-runner` - authoritative runner syntax and latency/profiling details.
- `cloud-ai-qaic-python` - Python benchmarking and integration when `qaic.Session` is required.
- `cloud-ai-accuracy-tuning` - numerical validation and INT8/mixed-precision policy.
