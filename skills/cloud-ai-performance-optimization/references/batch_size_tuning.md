# Batch Size Tuning — Latency vs. Throughput

## The tradeoff

Batch size is the primary knob between two performance goals:

| Goal | Batch | Right for |
|---|---|---|
| **Lowest latency** (fastest single response) | **1** | Real-time / single-stream / interactive (camera at 10–30 fps). Already what the core/MOS sweep optimizes. |
| **Highest throughput** (max img/s) | **> 1** | Offline / batch-processing / latency-tolerant pipelines. Amortizes weight-DDR reads and per-launch overhead across more samples — higher img/s at the cost of higher per-inference latency. |

Batch = 1 is always optimal for latency. Raising batch never improves latency; it only raises img/s until a hardware signal caps it.

```
Always begin the configurator sweep with `-search-batch-size 1,2,4,8`
(Step 2: Run the baseline/configurator sweep). Extend it to `16, ...` when the
throughput winner remains at the top of the searched range.
It produces two winners in one pass:
  Latency winner  → batch=1, lowest-latency core/instance config
  Throughput winner → best batch/core/instance config for max img/s

The deployment goal determines which QPC ships — it does not skip the sweep.
```

---

## Which flag to use

| Situation | Flag |
|---|---|
| ONNX has a symbolic batch dim (e.g. `batch_size`) | `-onnx-define-symbol=batch_size,N` |
| Single-input / single-output model, need to override the compiled batch | `-batchsize=N` |
| Multi-input or multi-output model | `-onnx-define-symbol=batch_size,N` — **`-batchsize` is documented as unsupported for multi-IO models; the tool will not reject it but behavior is undefined** |

`-batchsize=N` recompiles the graph at batch N. Input raw files must have the correct size for that batch — the tool validates `expected_size == N × single_sample_size` and errors on mismatch.

`-stats-batchsize=N` is a **separate, unrelated flag** — it only normalizes reported performance statistics to per-inference. It does not change the compiled graph and is never a substitute for `-batchsize`. It is auto-detected from the graph when unset.

---

## Batch vs. concurrent activations (`-a`)

These are frequently confused:

| Knob | Where | What it does |
|---|---|---|
| `-batchsize=N` | `qaic-compile` | Changes the compiled graph — N samples processed per inference call |
| `-a N` | `qaic-runner` / `qaic.Session` | Runtime concurrency — N inference calls in flight simultaneously on the same QPC |

`-a N` does **not** give you a batch-N graph. It runs the compiled QPC N times concurrently, raising sustained throughput under load. It has no effect on single-inference latency.

They compose: a QPC compiled at batch 4 can be run with `-a 2` (8 samples in flight simultaneously).

---

## When to stop the batch sweep

Sweep batch ∈ {1, 2, 4, 8, 16, …}, recompile and reprofile each value with `qaic_profile.sh`.

### 1. Throughput plateau — primary decision signal

**Completed inference units/sec plateaus or drops** vs the previous power of two. This is the decision. Stop here.

> For a direct `qaic-runner` result, `Inf/Sec` is QPC executions/sec. Multiply it
> by the compiled static batch before comparing batch values. Get the compiled
> batch from `qaic-runner -t <qpc> -q` or the QPC input shape, not its stats
> `BatchSize` field. Configurator `results.csv` throughput is already
> batch-normalized and is compared directly.

### 2. Per-inference latency budget — user-side constraint

Stop if per-batch latency exceeds the use-case ceiling, regardless of throughput. This is independent of hardware signals — it reflects a real-time or pipeline constraint set by the deployment.

### 3. Runtime opstats — explains the plateau, informs next steps

Collect `qaic_profile.sh` at each batch. When throughput stops growing, opstats shows why:

- **Op placement flips from `TCM` to `DDR`** — hot-path conv ops begin showing `DDR` instead of `TCM` in the summary. This is measured activation DDR round-trip, not an estimate.
- **`DMAIssue` thread utilization climbs sharply** — the DMA unit is moving activations to/from DDR. A VTCM-resident model at batch=1 typically shows ~2% `DMAIssue`; a large rise confirms real spill traffic.

These signals don't replace the throughput plateau as the stop criterion — they explain it. Use them to decide whether `-aic-enable-depth-first` or a MOS adjustment at the chosen batch is worth trying before finalizing.

### Compile-time hard-stop (rare)

`### Irreducible fatness ... exceeds BAR limit. Trouble ahead` — printed unconditionally by the compiler when the batch is geometrically too large for the tiler. Stop immediately; do not benchmark this config.

All other compile-time warnings (`Spilling buffer`, `BAR fatness detected`, `DDR traffic write` rising) are estimates that can diverge from runtime. A soft batch=1 warning can coexist with higher-batch throughput wins; treat it as context to watch during the sweep, not as a stop signal by itself. Do not use these as stop signals — use throughput and opstats.

---

## Sweep mechanics

The `cores × instances × batch` sweep can be automated with
`qaic-model-configurator` — see `qaic_model_configurator.md`. It compiles and
benchmarks every valid combination (enforcing the `cores × instances ≤ NSP`
budget) and reports throughput per config. It is signal-blind, so still confirm
the winning config with the diagnostic steps below.

### Normal path: compile-time batch binding

Use this path when `-batchsize=N`, `-onnx-define-symbol=batch_size,N`, or
`-onnx-define-symbol-batch-size batch_size` changes every batch-dependent input,
output, and internal shape in the graph.

To run the sweep manually (or to confirm a configurator winner), perform the
batch-specific coverage in Step 2. Use Step 3's diagnostic evidence and Step 4's
measured-candidate workflow when comparing batch-specific flags:

1. For each batch value in {1, 2, 4, 8, 16, …}:
   - Recompile with `-batchsize=N` (or `-onnx-define-symbol=batch_size,N`) and `-aic-perf-metrics -aic-perf-warnings` for the metrics log.
   - Benchmark with `qaic-runner` using the same `parse_latency.py` workflow as the core sweep.
   - When opstats are needed, compile a separate diagnostic QPC with the same
     batch-specific configuration plus `-stats-level=80 -ddr-stats`, then run
     `qaic_profile.sh` against that diagnostic QPC to collect one diagnostic
     QPC instance. Do not profile the metrics-only or production QPC.
   - Record: raw `Inf/Sec` (QPC executions/sec), completed inference units/sec,
     per-execution latency (ms), `DMAIssue`%, and any new `DDR`-suffix ops.
2. Stop at the first signal above.
3. **Re-evaluate `-mos` at the chosen batch.** The optimal out-channel split can shift with batch — the batch=1 MOS winner is not guaranteed to win at batch=4. Run at least the compiler heuristic (omit `-mos`) and explicit `-mos=1` at the chosen batch before finalising.

> **Use `qaic_profile.sh` at every batch value.** Throughput alone does not show when activations start spilling — the op `TCM→DDR` flip only appears in opstats.

### Fallback path: `TARGET_BATCH` re-export

Use this path for any model where the final ONNX cannot be batch-resized by a
compiler flag because export, onnxsim, or model-specific transforms baked batch
into internal shape tensors. Common symptoms:

- Graph inputs still show `batch_size`, but outputs or internal `Reshape` /
  `Expand` / `Tile` shape tensors are fixed to batch 1.
- Batch>1 compile or ORT fails with `Reshape ... different number of elements`,
  broadcast-shape errors, or input-size mismatches.
- `qaic-model-configurator -search-batch-size 1,2,4,8` fails before producing
  valid batch>1 rows, while batch=1 compiles.

Do **not** conclude batch>1 is impossible from a batch-1-specialized artifact.
If the export script does not already support `TARGET_BATCH`, add that support first;
setting the environment variable alone is not enough. Then produce one ONNX per batch:

1. Add `TARGET_BATCH = int(os.environ.get("TARGET_BATCH", "1"))`.
2. Generate representative inputs and goldens with first dimension `TARGET_BATCH`.
3. Keep symbolic batch in the raw export where possible, but pass `TARGET_BATCH`
   into any transform that must bake shape tensors. Never hardcode `batch_size -> 1`
   in a reusable transform.
4. If output metadata must be concrete, specialize it to `TARGET_BATCH`, not 1.
5. Name artifacts by batch (`model_b1_fp16ready.onnx`, `model_b2_fp16ready.onnx`, ...).
6. Compile each batch-specific ONNX directly and benchmark matching batch input data.
   Do not rely on configurator batch search for this fallback; sweep cores × instances
   for each batch-specific ONNX and compare completed inference units/sec.

Only declare batch>1 unsupported after the `TARGET_BATCH=2` re-export itself fails
export, ORT fidelity, or `qaic-compile` for a non-repairable model reason.

---

## Signals that explain batch-sweep results

Run the configured batch sweep first. These compile/opstats signals explain why
throughput improves, plateaus, or drops; they do not replace throughput as the
decision signal. Do not label a model a good or poor batching candidate from
clean batch=1 compile metrics (HMX MACs, barriers, no dynamic shapes, or
`VTCMMaxSize`); those only show the batch=1 compile is clean. `VTCMMaxSize` is
not a throughput or spill verdict.

- **`constant only` fraction of `DDR traffic read` ≈ 95%+** and **`DMAIssue` > 10%** in opstats — the model is weight-DDR-bound; batching may improve img/s by amortizing weight reads.
- **`BAR fatness detected`, `Spilling buffer`, or rising compile-time DDR estimates at batch=1** — pressure signals to watch during the sweep, not stop signals. Confirm with throughput plateau, runtime `DMAIssue`, and op placement (`TCM` to `DDR` suffix).
- **`### Irreducible fatness ... exceeds BAR limit. Trouble ahead`** — hard stop for that configuration; do not benchmark it.

---

## Common mistakes

| Mistake | Correct |
|---|---|
| Using `-a 8` expecting a batch-8 graph | `-a` is runtime concurrency, not compiled batch. Use `-batchsize=8` to compile a batch-8 graph. |
| Using `-batchsize` on a multi-IO model | Use `-onnx-define-symbol=batch_size,N` for multi-IO models |
| Treating `BAR fatness` or high `VTCMMaxSize` at batch=1 as a reason to skip batching | Run the planned batch sweep; stop on throughput plateau, latency budget, runtime spill (`DMAIssue`, DDR-suffix ops), or the irreducible fatness hard-stop |
| Using `VTCMMaxSize` as the spill cutoff | It saturates at 8 MB once spill begins. Use the op `TCM→DDR` flip and rising `DMAIssue` from opstats instead |
| Assuming the batch=1 MOS winner is optimal at higher batch | Re-evaluate `-mos` at the chosen batch — the optimal value can shift |
| Benchmarking without reprofiling | Always run `qaic_profile.sh` at each batch — throughput alone won't show the `TCM→DDR` flip |
