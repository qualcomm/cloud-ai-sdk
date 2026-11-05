# Profile-Guided Quantization and Mixed Precision Reference

PGQ runs representative inputs through a profiling compile pass to collect
per-layer activation statistics, then uses the profile to guide INT8 compilation.
Use this workflow only after the FP16 accuracy and performance baselines exist and
the user has explicitly requested INT8 or higher-throughput quantization work.

## Prepare representative calibration data

Use real inputs that represent the deployment distribution. Read the model source
repository's README and evaluation documentation first: the model's evaluation
dataset is the correct calibration domain.

> **Preprocessing similarity is not domain equivalence.** ViT, DINOv2, ResNet,
> ConvNeXt, and other backbones may use ImageNet mean/std normalization for
> unrelated tasks. Do not infer the dataset from normalization constants.

- If the repository evaluates on ImageNet classification, use
  `download_imagenet_calibration.py` with PASS (default, CC BY 4.0, no auth),
  Open Images v7, Hugging Face ImageNet-1k, or a local validation directory.
- If it evaluates on COCO detection/segmentation, use
  `download_coco80_calibration.py` for YOLO/letterbox or
  `download_coco91_calibration.py` for RCNN/variable-size inputs.
- For every other domain, use the model's own evaluation dataset and the same
  preprocessing as `predict.py` to produce `.raw` files.

Locate a bundled script without assuming one skills installation root:

```bash
find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude \
  -path "*/cloud-ai-accuracy-tuning/scripts/download_imagenet_calibration.py" \
  2>/dev/null | head -1
```

Calibration and test data must never overlap. Download 1,000 images and split
them immediately, before any compile pass:

| Model type | Calibration | Test | Split |
|---|---:|---:|---:|
| Classification | 500 (0-499) | 500 (500-999) | 50/50 |
| Detection, segmentation, transformer, OCR, attention, or other | 750 (0-749) | 250 (750-999) | 75/25 |

```python
import json

batch = json.load(open("batch_inputs.json"))
calib_n = 500  # use 750 for non-classification models
json.dump({"IO-files": batch["IO-files"][:calib_n]}, open("calib_inputs.json", "w"))
json.dump({"IO-files": batch["IO-files"][calib_n:]}, open("test_inputs.json", "w"))
```

Use only `calib_inputs.json` to generate profiles and only `test_inputs.json` for
the final INT8 accuracy evaluation.

If representative calibration data cannot be sourced, use AIMET CLE as the
documented data-free fallback. CLE algebraically rescales weights without forward
passes and is better than `-use-random-input-data=gaussian`, but it does not
calibrate activation ranges and normally trails real-data PGQ. See [AIMET](aimet.md).

## Check INT8 viability before a full calibration run

Use 10-20 representative images from the calibration domain for this gate.

1. Reuse the best validated FP16 throughput result. If no performance sweep
   exists, compile a quick FP16 QPC at 8 cores and benchmark it.
2. Generate a minimal PGQ profile from the 10-20 images.
3. Compile one INT8 QPC with the same performance flags as the FP16 baseline,
   without `-convert-to-fp16`, and capture the complete log:

   ```bash
   qaic-compile ... \
     -quantization-precision=Int8 \
     -quantization-precision-bias=Int32 \
     2>&1 | tee int8_compile.log
   ```

4. Record higher-precision fallbacks:

   ```bash
   grep "Running in higher precision" int8_compile.log | wc -l
   grep "Running in higher precision" int8_compile.log | head -20
   ```

   Record the count, a representative sample, and every grouped
   `node_precision.yaml` region later added to remove a mixed-type regression.

5. Benchmark the QPC, then collect opstats for the FP16 and INT8 QPCs using the
   `cloud-ai-performance-optimization` **Collect diagnostic evidence** workflow
   and its opstats triage reference. Compare the top 20 operations side by side.

For an operation present in both profiles:

- If INT8 ucycles are greater than `1.3×` FP16 and the type annotation mixes
  types such as `float16, i8, ui8`, dequant/requant work is crossing a precision
  boundary. Pin from the nearest upstream precision boundary through the
  regressed operation. **Pinning only the regressed operation is wrong** because
  the INT8 input boundary remains.
- If the operation is absent from the INT8 top 20 or has lower ucycles, leave it
  in INT8; the compiler found an efficient path such as a lookup approximation.

Trace backward to a natural precision boundary: residual Add, LayerNorm,
BatchNorm, embedding/Gather output, or linear projection output. Pin every node
from that boundary through the regressed operation. Recompile, rerun opstats,
verify that the mixed-type annotation is gone, and rebenchmark. Carry the same
`node_precision.yaml` through all later schema, calibration, and mixed-precision
work.

- Throughput `<=` FP16 after grouped pinning: INT8 is not viable; ship FP16.
- Throughput above FP16 but below `1.5×`: the schema/calibration work may
  continue, but record that the production target is not yet met.
- Throughput `>= 1.5×` FP16: continue to schema and accuracy tuning.

INT8 changes the activation working set and can change VTCM spill/fill behavior, so run the full compiler
performance sweep separately with INT8 flags. Use
`cloud-ai-performance-optimization` as the source of truth for sweep dimensions
and carry the winning flags into every later INT8 build. Preserve only measured
flags such as `-aic-num-cores`, `-mos`, `-ols`, or
`-aic-enable-depth-first`; do not invent additions.

## Generate a PGQ profile

`-aic-batch-json-input` uses `IO-files`, not `{"inputs": [...]}`. The latter can
produce `[json.exception.type_error.305]` and skip the profile pass.

```json
{"IO-files": [[{"path": "/abs/path.raw", "io-direction": "in", "data-type": "float32", "map-to": "input_tensor_name"}], ...]}
```

Each inner array is one inference. A minimal builder is:

```python
import json
from pathlib import Path

files = sorted(Path("images").glob("*.raw"))[:500]
items = [[{"path": str(path), "io-direction": "in",
           "data-type": "float32", "map-to": "pixel_values"}]
         for path in files]
json.dump({"IO-files": items}, open("calib.json", "w"))
```

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=calib.json \
  -dump-profile=pgq_profile.yaml \
  -quantization-schema-activations=symmetric_with_uint8 \
  -quantization-calibration=KLMinimization
```

`symmetric_with_uint8` is a good default for ReLU-heavy CNN/vision models;
`symmetric` is useful for signed activation ranges; `asymmetric` captures a full
range but is less hardware-friendly. `KLMinimization` commonly outperforms simple
min-max. Query the installed compiler for supported values rather than assuming
the list is stable.

The output is `pgq_profile.yaml`, containing per-layer range statistics.

Regenerate the profile whenever the activation schema or
`-node-precision-info` changes.

## Compile INT8 with the profile

Start from the FP16 command, keep its measured performance flags, remove
`-convert-to-fp16`, and add the INT8 profile and precision flags. The FP16 flag is
incompatible with INT8 and mixed precision.

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=./inputs.json \
  -load-profile=pgq_profile.yaml \
  -quantization-schema-activations=symmetric_with_uint8 \
  -quantization-precision=Int8 \
  -quantization-precision-bias=Int32 \
  -aic-hw -aic-hw-version=ai100 \
  $PERF_FLAGS \
  -aic-binary-dir=./model_int8_qpc
```

The loaded profile must use the same schema as the profile-generation pass. Add
only performance flags proven by the performance sweep.

## Select the schema with qaic-pytools

Before the full calibration run, compare every supported activation schema and
calibration method on the same 20 test and 20 calibration images. Use
`qaic-acc-evaluator.py` with a `pgq_sweep` configuration; do not write a custom
sweep. The evaluator owns parallel compilation, shared profiles through
`pgq_group`/`reuse_pgq`, CPU FP32 reference outputs, the `cosine_sim` plugin, ONNX
IR 9 compatibility, and local-work-directory requirements.

Do not replace it with a manual sweep because it feels more reliable or gives
more control: the configuration YAML exposes the compiler flags. If the
evaluator rejects a compiler-listed schema, investigate first; omit it only when
it is genuinely unsupported, and record the discrepancy.

Read [qaic-pytools](qaic-pytools.md), especially **Image counts for pgq_sweep vs
production**, **Using CPU FP32 Reference Outputs for pgq_sweep**, and **Known
Issues**. Query valid schemas from the installed SDK:

```bash
/opt/qti-aic/exec/qaic-compile --help -vvv 2>&1 | \
  grep -E "quantization-schema|quantization-calibration"
```

This output is authoritative for valid `-quantization-schema-activations`,
`-quantization-schema-constants`, and `-quantization-calibration` values in the
installed SDK.

Use `-set-global count:20 calib:20 -silent -work-dir /tmp/...`. A 20-image
profile ranks schemas but is never a production profile. If the evaluator rejects
a compiler-listed schema, investigate first; omit it only when it is genuinely
unsupported, and record the discrepancy. Carry the schema with highest
`cosine_mean` into the full calibration run.

## Run full calibration and INT8 accuracy validation

Generate a fresh profile from the complete `calib_inputs.json`, compile with the
winning schema and INT8 performance flags, then run every item in
`test_inputs.json`. Use CPU FP32 outputs for formal accuracy. After the FP16 QPC
is validated, its outputs may also be used to isolate FP16-to-INT8 drift; record
that choice in the model card.

Report the result regardless of outcome. Pure INT8 commonly fails for transformer
and detection models; do not hide the result or weaken the acceptance metric.
Proceed to mixed precision only when the INT8 accuracy gate fails and throughput
still justifies further work.

## Mixed-precision override workflow

`-node-precision-info` names node instances by their first output. A minimal file:

```yaml
FP16NodeInstanceNames:
  - conv0_out
  - layernorm1_out
  - attention_scores
```

Generate a new profile with the override, then compile with the same override:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=inputs.json \
  -node-precision-info=node_precision.yaml \
  -dump-profile=pgq_mixed.yaml

/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=inputs.json \
  -node-precision-info=node_precision.yaml \
  -load-profile=pgq_mixed.yaml \
  -quantization-precision=Int8 \
  -aic-hw -aic-hw-version=ai100 \
  $PERF_FLAGS \
  -aic-binary-dir=./model_int8_mixed_qpc
```

For fused patterns such as Conv plus BatchNorm, keep both nodes in FP16 when
needed to preserve the fusion.

### Verify override names against the final graph

Post-export transforms can rename an output or move its old name onto a wrapper
such as `Squeeze`. Validate every policy against the exact final ONNX passed to
`qaic-compile`:

```python
import onnx
import yaml

model = onnx.load("model_fp16ready.onnx")
policy = yaml.safe_load(open("node_precision.yaml")) or {}
by_first_output = {node.output[0]: node for node in model.graph.node if node.output}

checked = 0
for key in ("FP16NodeInstanceNames", "FP32NodeInstanceNames"):
    for name in policy.get(key, []):
        checked += 1
        hit = by_first_output.get(name)
        print(f"{key} {name}: " +
              (f"{hit.op_type} {hit.name}" if hit else "NO MATCH"))
if checked == 0:
    raise SystemExit("No supported precision-override entries found")
```

Every entry must resolve to the intended operation. `NO MATCH` or an inserted
wrapper match means the policy is stale. Re-derive names from the final graph,
regenerate the profile, recompile, and rerun the accuracy gate. Correcting the
name is a prerequisite, not proof of recovery.

### Select sensitive nodes

Run the [qaic-pytools `fp16sweep` workflow](qaic-pytools.md#fp16sweep-workflow)
first. If the graph topology is unsupported or the result is insufficient, start
from any grouped pins created during the viability gate. If viability created no
grouped pins, build `node_precision.yaml` from scratch. Then widen manually:

1. Pin the nearest natural boundary through attention-score MatMuls and Softmax.
2. Add LayerNorm outputs.
3. Add Erf/GELU activations.
4. Add feed-forward intermediate outputs.
5. Stop when the accuracy thresholds pass without losing the throughput target.

After each change, regenerate the profile, recompile, benchmark throughput first,
then run accuracy. A configuration already below the throughput target is not
worth the more expensive accuracy evaluation.

For coarse control, `-keep-original-precision-for-nodes=<op_kind>` preserves an
operator kind's original precision and `-execute-nodes-in-fp16=<op_kind>` runs all
instances of an operator kind in FP16.

## External quantization profiles

When AIMET or histogram tooling provides encodings, replace
`-load-profile=pgq_profile.yaml` with
`-external-quantization=encodings.yaml` in the INT8 compile. Read
[AIMET](aimet.md) for its compiler-required v0.5.0 YAML schema and conversions.
If a tensor is absent, the compiler falls back to FP16 for that layer; record and
either correct every missing entry or explicitly accept the fallback.
