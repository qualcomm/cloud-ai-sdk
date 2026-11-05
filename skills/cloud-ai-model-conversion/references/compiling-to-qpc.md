# Compiling Models to QPC

## Contents

- Basic ONNX and direct TensorFlow compilation
- Required compiler inputs and core-count guidance
- Compile-output validation and retry safety

## 4. Basic Compilation to QPC

### 4.1 ONNX Compilation Example

```bash
rm -rf ./model_qpc
/opt/qti-aic/exec/qaic-compile \
  -m=./model_fp16ready.onnx \
  -onnx-define-symbol=batch_size,1 \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=<N> \
  -aic-perf-metrics \
  -aic-perf-warnings \
  -aic-binary-dir=./model_qpc \
  2>&1 | tee compile.log
```

Key flags:

- `-m=./model_fp16ready.onnx` - final ONNX produced by the export pipeline.
- `-onnx-define-symbol=batch_size,1` - maps a symbolic dim to a concrete value (repeat per symbol).
- `-convert-to-fp16` - converts weights/activations to FP16.
- `-aic-hw`, `-aic-hw-version=ai100` — target Cloud AI 100 hardware.
- `-aic-num-cores=<N>` — number of NSP cores in the deployment target. Compilation does not require a QAIC device; use the target's known baseline core count, usually all target NSP cores. If the target is accessible, `/opt/qti-aic/tools/qaic-util -q` reports it. Core-count sweeps belong to `cloud-ai-performance-optimization`.
- `-aic-binary-dir=./model_qpc` - directory where QPC binaries are written.

Quantization (INT8) is optional. Start with FP16; use `cloud-ai-accuracy-tuning`
for the precision policy and validation workflow.

### 4.2 TensorFlow Direct Compilation

The current `qaic-compile` accepts a frozen TensorFlow GraphDef (`.pb` or
`.pbtxt`) or a TensorFlow SavedModel directory. A Python TensorFlow model must
be exported to one of those formats before compilation. For TensorFlow models,
`-output-node-name` is required. Static input shapes are normally inferred from
the graph; use `-model-input` only when the compiler cannot infer a valid input
contract. Fix dynamic or missing shape metadata in the TensorFlow export, or
use the TensorFlow-to-ONNX fallback.

For a frozen GraphDef:

```bash
rm -rf ./tf_model_qpc
/opt/qti-aic/exec/qaic-compile \
  -m=./model.pb \
  -output-node-name=<output_node[,output_node...]> \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=<N> \
  -aic-perf-metrics \
  -aic-perf-warnings \
  -aic-binary-dir=./tf_model_qpc \
  2>&1 | tee tf_compile.log
```

For a SavedModel, replace `-m=./model.pb` with the SavedModel directory and
inspect its signature input and output tensor names before compiling:

```bash
saved_model_cli show --dir ./saved_model_dir --all
```

Pass the graph node portion of each output tensor name to `-output-node-name`;
strip the `:index` suffix and do not pass the signature dictionary key. Validate
the export in the target SDK container. If the
compiler rejects the graph, signature, or export version, convert the
SavedModel with `tf2onnx` and continue with the ONNX path.

Direct TensorFlow compilation does not produce `_fp16ready.onnx` or use the
ONNX transform pipeline. Generate the framework golden from the original
TensorFlow model, record the selected output names, and compare the QPC output
against that golden after execution.

---

## Compile Workflow

### Step 3: Compile to QPC

Use the ONNX command in 4.1 or the TensorFlow command in 4.2 for the selected
source format. In either case, remove any previous `-aic-binary-dir` before
compiling; `qaic-compile` creates that directory and rejects a pre-existing
directory. Capture the complete compiler output in a log.

After compilation, scan compile.log for `Compiling model with FP32 precision` before proceeding. This line means the precision flag was omitted — a FP32 compile produces a valid QPC where all Conv/MatMul fall back to HVX, `AIC HMX MAC estimate` will be 0, and throughput will be dramatically slower, with no error or warning to indicate the problem. Always include this full string in any grep pattern used on compile.log so it is never silently filtered out.

> **A non-zero exit code does not always mean the compile failed.** `qaic-compile` prints `Generated binary is present at <path>` on success. If that line appears in the output, the binary was written — the failure is in the surrounding shell (e.g. `tee` on a missing absolute path, a deprecated-flag warning surfacing as exit 1). Always check for the binary at `-aic-binary-dir` before deciding to recompile. **Do not recompile if the binary already exists.**

Use the target baseline for `-aic-num-cores` (usually all target NSP cores). **Do not set `-mos` on the initial compile**; leave the compiler heuristic active. Always include `-aic-perf-metrics` on the first compile and record actionable signals first: HMX/non-HMX MACs, barriers, per-core `DDR traffic write`, and spill warnings. `VTCMMaxSize` is a scratchpad high-water mark capped by the compiled target's VTCM capacity; do not report it as a pass/fail "within limit" result. Actual spill or hard failure is indicated by per-core `DDR traffic write`, `Spilling buffer ... to DDR`, irreducible fatness / BAR hard-stop warnings, or runtime opstats showing DDR-suffix ops / rising `DMAIssue`. MOS/OLS tuning belongs in `cloud-ai-performance-optimization`.

For symbolic ONNX dims, add `-onnx-define-symbol=batch_size,1` (repeat per
symbol). TensorFlow input shapes are inferred from the TensorFlow graph instead.
For the full flag reference see the `cloud-ai-qaic-compile` skill.
