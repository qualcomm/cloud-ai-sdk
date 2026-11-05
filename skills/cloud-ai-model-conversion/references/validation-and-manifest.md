# Validation and Conversion Manifest

## Contents

- Required export artifacts and fidelity gate
- Pre-tuning sanity checks
- QPC sanity execution
- Conversion manifest schema

## Sanity Checks Before Tuning

Before handing off to `cloud-ai-accuracy-tuning` or
`cloud-ai-performance-optimization`, confirm:

1. **CPU baseline is valid**
   - For ONNX, run the final ONNX with ONNX Runtime and verify outputs.
   - For direct TensorFlow compilation, run the original TensorFlow model and
     save the framework golden for the selected output nodes.

2. **FP16 QPC runs end-to-end**
   - Compile with `-convert-to-fp16` and run a single inference with
     `qaic-runner` or `qaic.Session` using the matching input contract.

3. **Outputs roughly match CPU**
   - Compare outputs for a few test cases using simple metrics (e.g., MSE, cosine similarity).
   - Large discrepancies usually indicate issues in preprocessing, layout, or graph conversion rather than QAic itself.

Once these checks pass, you can move on to:

- Accuracy Tuning (FP16/INT8, PGQ, mixed precision)
- Performance Optimization & Runtime (compiler perf knobs, SDP/MDP, qaic-runner)
- Hexagon Custom Ops (C++/HVX kernels and custom op integration)

---

## Export Artifact Contract

**Export artifact contract — choose the matching branch:**

For the ONNX route, the export script must produce:

| File | Description |
|---|---|
| `<model_name>.onnx` | Intermediate ONNX from `torch.onnx.export` / `tf2onnx`. Inline surgery (output trimming, output renaming) applied here before passing to `run_pipeline()`. |
| `<model_name>_fp16ready.onnx` | **Final ONNX produced by `run_pipeline()` inside the export script** — onnxsim + model-specific transforms + FP16Clip. This is the file passed to `qaic-compile`. |
| `input.raw` | Contiguous float32 representative input, shape = model input shape |
| `framework_output_0.raw` | Framework (PyTorch / TF) golden output for `input.raw` — required by `cloud-ai-accuracy-tuning` [Generate framework golden outputs](../../cloud-ai-accuracy-tuning/references/onnx-baseline-validation.md#generate-framework-golden-outputs) |
| `cpu_output.raw` | ORT CPU output on `<model_name>_fp16ready.onnx` — generated at the bottom of the export script, after `run_pipeline()` completes, and compared against the framework golden in the same script |

For the **TensorFlow direct-compile branch**, preserve these artifacts instead:

| File | Description |
|---|---|
| `<model_name>.pb` or `<saved_model_dir>/` | The exact frozen GraphDef or SavedModel passed to `qaic-compile`. |
| `input.raw` | Contiguous input matching the TensorFlow input signature and dtype. |
| `framework_output_0.raw`, `framework_output_1.raw`, ... | TensorFlow golden outputs for `input.raw`, one per selected output in output order. |
| `tf_compile.log` | The complete `qaic-compile` output, including the selected output node names. |
| `<qpc_dir>/programqpc.bin` | QPC generated from the TensorFlow source. |

Do not require `_fp16ready.onnx`, `cpu_output.raw`, or `run_pipeline()` for
this branch. If the direct TensorFlow compile fails and the model is converted
to ONNX, switch to the ONNX artifact contract above.

**For the ONNX route, all graph surgery — output trimming, op replacement,
model-specific transforms — must be in the export script.** There is no
separate fix script or manual step between export and compilation. The export
script is the sole reproducible record of every ONNX graph modification.

For the ONNX route, save framework goldens before conversion. For PyTorch,
apply pre-export transforms first and run the transformed model before
`torch.onnx.export`; never regenerate goldens from the unpatched model. For the
TensorFlow-to-ONNX fallback, run the original TensorFlow source before
`tf2onnx`. Save one framework golden per selected output in output order.

For the ONNX route, `cpu_output.raw` must reflect the **fp16ready ONNX** — the exact file passed to `qaic-compile`. Generate it at the bottom of the export script from `<model_name>_fp16ready.onnx`, after `run_pipeline()` completes — never from the raw `<model_name>.onnx`.

For ONNX, `input.raw`, `framework_output_0.raw`, and `cpu_output.raw` together satisfy the artifact requirements of `cloud-ai-accuracy-tuning` [ONNX baseline validation](../../cloud-ai-accuracy-tuning/references/onnx-baseline-validation.md). The export script itself enforces that reference's [shape and numerical fidelity gate](../../cloud-ai-accuracy-tuning/references/onnx-baseline-validation.md#verify-shapes-and-compare-numerically) and exits non-zero when any output misses it, so a failing export leaves no `_fp16ready.onnx` that has passed the gate. A cosine reported below threshold means the export is unfinished, whatever the task or output modality.

For direct TensorFlow, compare the QPC output with `framework_output_0.raw`
using the same input and output ordering. A successful compile without
execution is not an accuracy result.

## QPC Sanity Check

### Step 4: Sanity Check

Prepare a few representative test inputs. Run the ONNX on CPU with ONNX Runtime for FP32 golden outputs, or run the original TensorFlow model for the TensorFlow direct-compile branch. Run the QPC on Cloud AI with `qaic.Session` — see the `cloud-ai-qaic-python` skill for usage. `qaic-runner` is a CLI alternative — see the `cloud-ai-qaic-runner` skill.

Compare outputs using MSE or cosine similarity. Large discrepancies usually indicate preprocessing, layout, or graph conversion issues rather than QAic itself — run `cloud-ai-accuracy-tuning` [Smoke-test the input pipeline](../../cloud-ai-accuracy-tuning/references/onnx-baseline-validation.md#smoke-test-the-input-pipeline) before assuming the QPC is at fault. Use **Sanity Checks Before Tuning** above for local conversion checks.

Once the QPC runs end-to-end and produces correctly shaped output, proceed to the `cloud-ai-accuracy-tuning` skill for full validation, or `cloud-ai-performance-optimization` to benchmark throughput. This sanity check confirms the QPC executes; it does not accept its numbers. The QAIC pass/fail gate belongs to `cloud-ai-accuracy-tuning`.

## Conversion Manifest

Every model conversion produces a build manifest for the compiled QPC. Write `<qpc_dir>/conversion_manifest.json` after compilation using this fixed schema:

```json
{
  "schema_version": 1,
  "model_name": "<model>",
  "stage": "<full-model|encoder|decoder|...>",
  "created_at": "<ISO-8601 timestamp>",
  "source_format": "<onnx|tensorflow_graphdef|tensorflow_savedmodel>",
  "source_path": "<relative/path/to/source>",
  "source_sha256": "<sha256>",
  "onnx_path": "<relative/path/to/final.onnx-or-null>",
  "onnx_sha256": "<sha256-or-null>",
  "output_nodes": ["<output-node-name>"],
  "input_manifest": "<relative/path/to/input_manifest.json>",
  "input_manifest_sha256": "<sha256>",
  "qpc_program": "programqpc.bin",
  "qpc_program_sha256": "<sha256>",
  "compile_command": "qaic-compile ...",
  "compile_log": "<relative/path/to/compile.log>",
  "precision": "fp16",
  "precision_overrides": null,
  "precision_overrides_sha256": null
}
```

Use `null` for `input_manifest` / `input_manifest_sha256` only when no input manifest exists. If `-node-precision-info` is used, set `precision` to `fp16_with_node_overrides` (or the matching precision mode) and fill `precision_overrides` plus `precision_overrides_sha256`.

The manifest binds the QPC to the source model, I/O contract, output names,
binary hash, compile command, compile log, and precision inputs so later steps
can reject stale or mismatched artifacts. For direct TensorFlow, set the ONNX
fields to `null` and record the TensorFlow source in `source_path`. For a
SavedModel directory, derive `source_sha256` from a deterministic file manifest
(relative paths and file bytes), not from a directory path alone.
