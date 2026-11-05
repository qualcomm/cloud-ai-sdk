# Custom Op Runtime Integration

## Contents

- QAicApi execution contract
- Buffer identification and batch I/O
- File layout
- Program lifetime, zero-copy, slots, and multi-NSP execution
- Runtime best practices

## Runtime integration (`QAicApi`)

### Critical: call `setData` before every non-zero-copy `run()`

```cpp
// WRONG — output is stale after the first invocation
exec->run(infvec);

// CORRECT — re-registers buffer pointers with DMA engine for each invocation
exec->setData(infvec);
exec->run(infvec);
```

Without `setData`, the non-zero-copy DMA engine uses a cached descriptor from
the first call and does not pick up updated input buffer contents. The output
appears correct on the first inference, then stale on all subsequent ones
(output buffer retains the previous result). Zero-copy `ExecObj` does not
require `setData`.

### `run()` is synchronous

`exec->run(infvec)` blocks until the NSP compute and output DMA are both complete.
Do **not** call `waitForCompletion()` after `run()` — it stalls indefinitely.
`waitForCompletion()` is only for the async `queue->enqueue()` path.

### Buffer layout

Buffer indices in `infvec->getVector()` follow the order in `custom_op_config.yaml`:
inputs first (in declaration order), then outputs (in declaration order).

**Important:** the compiled QPC may reorder buffers relative to the ONNX model
or YAML declaration. Always verify indices at runtime by matching on buffer size
or name via `qpc->getBufferMappings()`.

### Buffer identification by name

When a QPC has multiple buffers of the same size, size matching is ambiguous.
Use `getBufferMappings()` name fields to identify buffers robustly:

```cpp
const int nb = qpc->getBufferMappings().size();
for (int i = 0; i < nb; ++i) {
    const auto &m = qpc->getBufferMappings()[i];
    if      (std::strcmp(m.name, "input")   == 0) idx_in     = i;
    else if (std::strcmp(m.name, "output")  == 0) idx_out    = i;
    else if (std::strcmp(m.name, "status")  == 0) idx_status = i;
}
```

This is more robust than index constants and survives compiler reordering across
QPC rebuilds. Validate that all expected names were found before proceeding.

---

## `aic_batch_io.json` format

Maps raw binary files to named ONNX inputs for `qaic-compile` and `qaic-runner`:

```json
{
  "IO-files": [[{
    "path": "input_data.raw",
    "elem-size": 1,
    "dims": [H, W],
    "io-direction": "in",
    "map-to": "input"
  }]]
}
```

- `elem-size`: bytes per element (1 for int8/uint8, 4 for float32)
- `dims`: tensor shape matching the ONNX model input
- `map-to`: must match the input name in the ONNX graph

---

## File reference

| File | Purpose |
|------|---------|
| `onnx_torch/gen_<opname>.py` | Generates `custom.onnx`, test data, `aic_batch_io.json` |
| `CustomOp_Package/custom_op_config.yaml` | Compiler registration config |
| `CustomOp_Package/<OpName>/src/<opname>_functions.cpp` | Shape inference, verification, selection |
| `CustomOp_Package/<OpName>/src/<opname>_interpreter.cpp` | CPU reference implementation |
| `CustomOp_Package/<OpName>/src/<opname>_aic.cpp` | NSP kernel (HVX vectorized, 4-thread) |
| `run.sh` | Build + compile + test driver |
| `hw_bin_dir/` | Compiled QPC output (passed to `Qpc::Factory()`) |
| `<opname>_runner/` | Standalone runtime test harness |

---

### Runtime integration

**Load the QPC once at init, keep it resident.** `program->activate()` incurs
activation overhead. Do it once at application open and keep the program
resident for the lifetime of the pipeline.

**Avoid `program->deactivate()` in latency-sensitive paths.** Deactivation is
a full NSP teardown and can be expensive. For long-running pipelines, keep
programs resident in a persistent daemon to avoid repeated teardown and
activation costs.

**Use zero-copy `ExecObj` for large buffers.**  For buffers ≥ 1 MB,
`QAIC_EXECOBJ_PROPERTIES_ZERO_COPY_BUFFERS` + `getIoBuffers()` eliminates
the heap→DMA copy per invocation. Write directly into the DMA-mapped buffer
returned by `getIoBuffers()`.

**One `Program` per slot for multi-slot deployments.**  Loading one `Program`
per QID and sharing it across multiple `ExecObj`s results in only one NSP
being used regardless of slot count.  Each slot that should run on a
separate NSP must have its own `Program` instance.

**NSP execution time is content-dependent, not constant.** For ops with
data-dependent work (e.g. kernels that skip processing based on per-block
flags), execution time can vary materially across inputs. Profile with
representative content and size latency budgets against worst-case, not
average. Check Min versus Max in the stats output when diagnosing variance.

**Prefer 1-slot synchronous over ping-pong when slot count is constrained.**
Ping-pong (two `(ExecObj, InferenceVector)` pairs alternating per handle) can
hide host-side copy overhead by overlapping it with NSP execution. However, it
doubles slot consumption. Use ping-pong only when you have confirmed headroom
in the slot budget.

**Multi-NSP per slot: implement `customOpSetProperties`.**  To use multiple
NSPs for a single inference, implement `customOpSetProperties` in the
functions library and return `true` with the desired core count.  This is
the documented hook for per-op NSP tiling; consult the SDK RST docs for the
exact interaction with `-aic-num-cores=N`.

---
