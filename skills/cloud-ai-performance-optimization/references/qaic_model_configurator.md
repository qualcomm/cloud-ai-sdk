# qaic-model-configurator — Automated (cores × instances × batch) Sweep

`qaic-model-configurator` automates the throughput-axis search: it compiles and
benchmarks a model across combinations of cores, instances, batch size, and
compiler flags, then reports the best configuration for a chosen objective. It is
the throughput-sweep engine referenced by `batch_size_tuning.md`.

Location: `/opt/qti-aic/scripts/qaic-model-configurator/model_configurator.py`.

## When to use it

- **Use for the throughput axis** — sweeping `cores × instances × batch` (and
  optionally `mos`/`ols`/`vtcm-ratio`/etc.) to find max images/sec on one device.
- **Do not use as the authoritative latency-validation path** — it has no opstats,
  no compile-log signals, and no warmup. Step 2 of the perf-optimization skill
  uses the configurator for coverage, then re-measures the selected latency winner
  with the runner's latency workflow. Step 3 and `references/opstats-triage.md`
  provide the diagnostic path. The configurator is a black-box
  throughput/latency *measurement* engine, not a diagnostic one.

## The two search modes

| Mode | Triggered by | What it does |
|---|---|---|
| **Brute-force** | any `-search-*` flag, or `-load-configs <csv>` | Compiles + benchmarks **every** valid combination in the search space |
| **Optimized** (default) | no `-search-*` flags | A guided search (in the closed `config_searcher.so`) that starts from device-filling seed points and optimizes toward one objective; bounded by `max-evaluations` |

> **If you pass no `-search-*` arguments, you get the optimized search, not a
> single compile.** It defaults to objective `max-throughput`, cores `1..NSP`,
> batch `1..8`, instances `1..NSP`, seeded with every `cores × instances == NSP`
> partition. To control exactly what runs, use brute-force with explicit
> `-search-*` ranges.

## NSP budget is enforced automatically

Each instance reserves its own core set: a config is valid only if
`cores × instances ≤ device NSPs`. The tool prunes the rest. On a 16-NSP device,
`-search-cores 1,2,4,8,16 -search-instances 1,2,4` yields `1c×{1,2,4}`, `2c×{1,2,4}`,
`4c×{1,2,4}`, `8c×{1,2}`, `16c×{1}` — the `8c×4` and `16c×{2,4}` combinations are dropped.

### Throughput selection is full-SoC only

`NSP_TOTAL` is the total NSP count captured from the selected QID. A throughput
candidate is eligible for selection only when `cores × instances == NSP_TOTAL`.
The sweep may produce valid partial-SoC rows for latency coverage or diagnosis,
but never rank one as the throughput winner. Select the maximum numeric
`Throughput (inf/sec)` among the eligible full-SoC rows; this value is aggregate
completed inference throughput across all selected instances. If no full-SoC row
exists, repair or rerun the sweep instead of selecting the best partial row.

## Minimal throughput sweep

`model_configurator.py` sets `TMPDIR=<output-dir>/tmp`. Compiler temp writes can fail
at `fallocate` for unsupported filesystem even when plenty of space
is available. Choose a configurator output directory on a filesystem that supports
`fallocate`, then copy the small summary artifacts back to the workspace.

```bash
set -euo pipefail

supports_fallocate() {
  local dir="$1"
  mkdir -p "$dir" || return 1
  local probe="$dir/.fallocate_probe.$$"
  rm -f "$probe"
  if fallocate -l 64M "$probe" >/dev/null 2>&1; then
    rm -f "$probe"
    return 0
  fi
  rm -f "$probe"
  return 1
}

CFG_ROOT=""
if supports_fallocate "./.qaic_cfg_scratch"; then
  CFG_ROOT="./.qaic_cfg_scratch"
elif [ -d /scratch ] && supports_fallocate "/scratch/qaic_cfg_scratch_${JOB_ID:-$$}"; then
  CFG_ROOT="/scratch/qaic_cfg_scratch_${JOB_ID:-$$}"
elif supports_fallocate "/tmp/qaic_cfg_scratch_${JOB_ID:-$$}"; then
  CFG_ROOT="/tmp/qaic_cfg_scratch_${JOB_ID:-$$}"
else
  echo "ERROR: no configurator output location supports fallocate" >&2
  exit 1
fi

CFG_OUT_LOCAL="$CFG_ROOT/cfg_out"
rm -rf "$CFG_OUT_LOCAL"
mkdir -p "$CFG_OUT_LOCAL"

python3 /opt/qti-aic/scripts/qaic-model-configurator/model_configurator.py \
  -model ./model_fp16ready.onnx \
  -output-dir "$CFG_OUT_LOCAL" \
  -convert-to-fp16 \
  -onnx-define-symbol-batch-size batch_size \
  -search-cores 1,2,4,8,16 \
  -search-instances 1,2,4 \
  -search-batch-size 1,2,4,8 \
  -device-id 0 \
  -time 30 \
  -max-parallel-compilations 8

mkdir -p ./cfg_out
cp -f "$CFG_OUT_LOCAL/results.csv" ./cfg_out/results.csv
cp -f "$CFG_OUT_LOCAL/model_configurator.log" ./cfg_out/model_configurator.log
```

The primary output is `results.csv`, which summarizes each candidate
configuration and its measured throughput. Normalize it with
`normalize_configurator_results.py` before ranking rows or copying values. The
normalizer preserves the original as `results.raw.csv` and rewrites `results.csv`
with `Avg Latency (us)` recomputed from throughput, batch size, and instances.
`model_configurator.log` records the configurator run itself. By default,
candidate QPC binaries are temporary and are removed after evaluation, so the
sweep output directory should be treated as a results directory, not as a
reusable compiled-model directory.

## Recipe: pruned brute-force search with `-save-configs` / `-load-configs`

Use this when the full `cores × instances × batch` sweep is too slow but you
still need latency and throughput coverage.

1. Generate the full valid brute-force table without compiling:

```bash
python3 /opt/qti-aic/scripts/qaic-model-configurator/model_configurator.py \
  -model ./model_fp16ready.onnx \
  -output-dir "$CFG_OUT_LOCAL" \
  -convert-to-fp16 \
  -onnx-define-symbol-batch-size batch_size \
  -search-cores 1,2,4,8,16 \
  -search-instances 1,2,4 \
  -search-batch-size 1,2,4,8 \
  -device-id 0 \
  -time 10 \
  -max-parallel-compilations 8 \
  -save-configs ./cfg_full.csv \
  -list-configs
```

2. Edit `cfg_full.csv` into `cfg_pruned.csv`:
   - Keep latency rows: `batch=1`, `instances=1`, powers-of-two `cores <= NSP_TOTAL`.
   - Keep throughput rows: standard batch values `1,2,4,8` for powers-of-two `(cores, instances)` pairs where `cores × instances == NSP_TOTAL`.
   - For `NSP_TOTAL=16` and `instances in 1,2,4`, the throughput pairs are `(1,4)`, `(2,4)`, `(4,4)`, `(8,2)`, `(16,1)`.
   - Do not remove batch values only because `VTCMMaxSize` is high or the batch=1 compile looks clean. Remove batch values only for export constraints, hard compile stops, resource limits, or measured bad results.

3. Run the exact pruned table:

```bash
python3 /opt/qti-aic/scripts/qaic-model-configurator/model_configurator.py \
  -model ./model_fp16ready.onnx \
  -output-dir "$CFG_OUT_LOCAL" \
  -convert-to-fp16 \
  -onnx-define-symbol-batch-size batch_size \
  -load-configs ./cfg_pruned.csv \
  -device-id 0 \
  -time 10 \
  -max-parallel-compilations 8
```

Use `-time 10` for scouting; re-run the chosen rows with a longer duration (for
example `-time 30`) before recording final numbers. Treat the pruned CSV as a
starting point, not proof of optimality. If the winners look suspicious, rows
fail, or explicit coverage is required, run the full exhaustive sweep and compare
`results.csv`.

## Gotchas (verified against runtime + configurator source)

### Keep configurator and runner units separate

`results.csv` `Throughput (inf/sec)` is batch-normalized completed inference
units/sec. Use it directly to compare configurator rows; do not multiply it by
batch again.

Direct `qaic-runner` `Inf/Sec` is a raw QPC execution rate, aggregated across
the selected `-a` activations. For a static-batch QPC, multiply that rate by the
compiled batch to report completed inference units/sec. The stats `BatchSize` field
is not the compiled batch: obtain the latter from the QPC with
`qaic-runner -t <qpc> -q` or from the compiled input shape.

### Normalize `Avg Latency (us)` before ranking

Use `Throughput (inf/sec)` to rank configurator throughput configurations.

Before using `Avg Latency (us)`, run the skill's normalizer on `results.csv`:

```bash
python normalize_configurator_results.py cfg_out/results.csv --in-place
```

The normalizer recomputes latency from the reliable aggregate relationship:

```text
Avg Latency (us) = instances * batchSize * 1_000_000 / Throughput (inf/sec)
```

It preserves the raw configurator output as `results.raw.csv` and makes the
normalized `results.csv` safe for latency-row ranking.

For rows with `instances > 1`, keep the metrics separate:

- Aggregate throughput: total completed inferences per second across all instances.
- Output cadence: `1 / aggregate_throughput`; not request latency.
- Request latency: normalized `Avg Latency (us)` for that row.

### No warmup, and it cannot be added

The benchmark runs `qaic-runner --time <N>` with no warmup or start-delay.
`RunnerConfig` exposes only `run_duration`; there is no warmup parameter in the
interface or the binary. The first inferences (load, cache fill) are included in
the measured window. Mitigations: use a longer `-time` (e.g. 30s) so startup is a
small fraction, and **re-measure the winning config** with the skill's
`parse_latency.py` (which uses `--aic-profiling-start-delay 2000`) for the
final number.

### `-search-batch-size` requires a symbolic batch axis in the ONNX

A fully-static ONNX fails to compile at batch>1 (`Reshape ... different number of elements`). The export script must declare a symbolic batch dim (`dynamic_axes={"input": {0: "batch_size"}, ...}`).

The symbol name is chosen at export time. Pass it to `-onnx-define-symbol-batch-size` so the tool substitutes each search value at compile time:

```
-onnx-define-symbol-batch-size batch_size   # must match the key used in dynamic_axes
-search-batch-size 1,2,4,8
```

`-onnx-define-symbol` is for fixing other ONNX symbols (e.g. `seq_len`) to constants — it is not the batch-sweep flag.

### `-input-list-file` is incompatible with a batch-size search

When sweeping batch, the tool must generate inputs per batch size; passing
`-input-list-file` raises an error. Either omit it (random inputs — fine for
throughput, which is data-independent) or supply per-batch inputs via
`-load-pre-gen-files` / `-create-batched-inputs`.

### Binaries are not retained by default

Candidate QPCs are deleted by default after benchmarking. Use `-retain-binaries`
only when you need all generated binaries preserved; but beware becaise it can use
significant disk space. `results.csv` remains the stable comparison artifact, and
per-candidate compile logs are not captured. If you only need to preserve 1 or 2
binaries from a large search space, don't use `-retain-binaries`.
Instead, use the information in `results.csv` to re-build the specific binaries of interest.

### High memory use with parallel compiles

The tool warns at 90% RAM and **kills itself at 98%**. Large models or high
`-max-parallel-compilations` can trigger this. Lower the parallelism if you see
the memory warning.

## Pruning the search with signals (recommended)

The configurator is signal-blind — it measures throughput but does not know
*why* a config is fast or slow. Bracket it with the perf-optimization skill's
diagnostics:

1. **Before** — use compile-log and opstats signals to choose safe resource settings for the sweep (for example, lower `-max-parallel-compilations` for large models, or avoid obviously impossible batch values after an irreducible fatness hard-stop). Do not use `VTCMMaxSize`, BAR fatness, or clean compile metrics as a reason to skip otherwise valid standard batch values; prune only values ruled out by export constraints, hard compile stops, resource limits, or measured throughput/opstats results.
2. **Run** the configurator on the pruned ranges.
3. **After** — recompile the winning config with `-aic-perf-metrics
   -aic-perf-warnings` for the metrics log. If runtime opstats are needed,
   recompile the same winner as a separate diagnostic QPC with
   `-stats-level=80 -ddr-stats`, preserving the winner's model, shapes, batch,
   cores, MOS/OLS, and other compiler flags. Run `qaic_profile.sh` to collect
   one diagnostic QPC instance only on that separate diagnostic QPC; never
   profile the metrics-only or production QPC. Use the opstats to confirm it is not sitting on a spill
   cliff (`TCM→DDR` op flip, rising `DMAIssue`, `Spilling buffer`/`BAR fatness`
   warnings).

For the signal definitions and the latency-vs-throughput framing, see
`batch_size_tuning.md`.

## Config-file (JSON) alternative

Brute-force and optimized searches can also be driven by `-config <file.json>`
instead of CLI flags. Sample configs ship under
`/opt/qti-aic/scripts/qaic-model-configurator/SampleFiles/` (`brute_force_search/`,
`optimized_search/`, `multi_model_optimized_search/`). The optimized-search JSON
takes `objective` (`max-throughput` | `min-latency`), per-parameter `min`/`max`
ranges, and `initial-values` seed points.

## Quick flag reference

| Flag | Purpose |
|---|---|
| `-model <onnx>` | Model to sweep |
| `-output-dir <dir>` | Results + logs (`results.csv`, `model_configurator.log`) |
| `-search-cores`, `-search-instances`, `-search-batch-size` | Brute-force ranges (`1-14` or `1,2,4,8`) |
| `-search-mos`, `-search-ols`, `-search-vtcm-ratio`, `-search-dealloc-delay`, `-search-split-size`, `-search-depth-first-search-mem`, `-search-set-size` | Additional brute-force search dimensions |
| `-static-<param>` | Pin a parameter instead of searching it |
| `-onnx-define-symbol-batch-size <name>` | ONNX symbol that batch values bind to (required for batch search on ONNX) |
| `-time <sec>` | Per-config run duration (use ≥30 to dilute the no-warmup transient) |
| `-max-parallel-compilations <n>` | Cap parallel compiles (lower if RAM-bound) |
| `-retain-binaries` | Keep compiled QPCs instead of deleting them |
| `-list-configs` | Print the brute-force search space and exit (dry run) |
| `-save-configs <csv>` / `-load-configs <csv>` | Dump / reload a brute-force space (edit the CSV to prune) |
| `-device-id <n>` | Target device QID (repeatable) |
| `-config <json>` | Drive the search from a JSON file instead of CLI flags |
