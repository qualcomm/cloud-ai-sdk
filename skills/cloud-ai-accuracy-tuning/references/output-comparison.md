# Output Comparison and FP16 Localization Reference

Use this reference after artifact identity and the framework-to-ONNX gate have
passed. For framework golden generation, shape checks, and the cosine `>= 0.9999`
gate, read [ONNX baseline validation](onnx-baseline-validation.md). For evaluator,
analyzer, histogram, and comparison commands, read
[qaic-pytools](qaic-pytools.md).

## Per-output comparison and reporting

Compare output shapes before numerical metrics. Judge every tensor independently;
an aggregate score cannot hide one failed output.

For the QAIC FP16 QPC against CPU ONNX Runtime:

- cosine similarity `>= 0.99`: PASS;
- cosine similarity `< 0.99`: FAIL and localize that output; and
- MSE is secondary because its meaning depends on output scale.

One failed output fails the model. Never accept an incorrect output as a "known
limitation." Record output name, shape on both sides, dtype, cosine, MSE, max
absolute error, and PASS/FAIL for every output.

```python
import numpy as np
from scipy.spatial.distance import cosine

def layer_stats(cpu_out: np.ndarray, qaic_out: np.ndarray) -> dict:
    cpu_f = cpu_out.flatten().astype(np.float32)
    qaic_f = qaic_out.flatten().astype(np.float32)
    return {
        "mse": float(np.mean((cpu_f - qaic_f) ** 2)),
        "cosine_sim": 1.0 - cosine(cpu_f, qaic_f),
    }
```

A tensor below `0.99` is a **divergence boundary**, not proof that the operation
producing it is inaccurate: its input may already be wrong. Compare the
operation's immediate inputs and outputs before changing or replacing it.

## Autoregressive decode validation

Single-step cosine at decode step 0 with zero past-KV does not validate
autoregressive fidelity. At that step the KV buffers have no effect, so dangling
KV inputs can produce matching initial logits while later tokens diverge.

After every output passes cosine, run a full end-to-end decode comparison on the
same representative input used for the smoke test:

```python
cpu_text = cpu_model_generate(test_input)
qaic_text = predict_script_run(test_input)
assert cpu_text.strip() == qaic_text.strip(), (
    f"Full decode mismatch: CPU={cpu_text!r}, QAIC={qaic_text!r}. "
    "Check KV-cache wiring in the model-conversion autoregressive export guidance."
)
```

If this fails while per-output cosine passes, inspect KV-cache wiring using the
`cloud-ai-model-conversion` autoregressive decoder export guidance. A near-tied
token still requires the tie analysis below; a tie-triggered abort does not
validate the rest of the decode.

## Cost-aware semantic bisection

Use this workflow when framework-to-ORT fidelity passes, QAIC accuracy fails,
and a full diagnostic compile materially slows iteration. Do not run another
broad full-model compile merely to test one local hypothesis.

For 12 or more homogeneous repeated layers, a four-layer configuration/module
slice is a useful first probe. For heterogeneous graphs or a localized operation,
use semantic boundaries rather than an arbitrary node range. Full-model
validation remains mandatory after the localized deviation is fixed.

1. Expose a sparse set of meaningful intermediate boundaries in the full graph.
   Do not add every node as an output unless the graph is small enough that
   compile time remains practical.
2. Confirm the instrumented build still reproduces the production failure.
   Added outputs block fusions and change tiling. Compare its final output with
   the uninstrumented result on the same inputs. If it now passes, discard its
   intermediates; they describe a different schedule. A passing instrumented
   build is not a fix.
3. Expose the suspect operation's immediate input and output. If the input is
   already inaccurate, move upstream; do not blame, replace, or precision-pin
   the operation.
4. If the input is accurate, build the smallest semantic probe containing the
   operation and its immediate postprocessing. Feed tensors captured from the
   failing full-model run, not synthetic random inputs.
5. Before compiling the probe, require its framework/module and ONNX Runtime
   outputs to meet the ONNX fidelity threshold on those captured inputs. A probe
   that fails this gate has a construction, wiring, or export problem and yields
   no QAIC evidence.
6. Test one hypothesis per compile. Compare the same probe with and without the
   proposed precision override, replacement, or reformulation. Record exact
   inputs, compile command, precision audit, and outputs.
7. Promote only a proven fix through the normal export path or a
   `BaseOnnxTransform` subclass. Validate the enclosing semantic boundary, then
   recompile and validate the full production graph.

Choose the smallest diagnostic form that preserves the suspected failure:

| Graph shape | Diagnostic form |
|---|---|
| Repeated homogeneous blocks | Config/module slice, such as 4 of 12 transformer layers |
| Heterogeneous pipeline | Explicit stage or module boundary |
| Localized operation/expression | Captured-real-input micrograph |
| Input already inaccurate | Sparse upstream semantic bisection |

Do not treat low cosine as proof the op is faulty, pin the full ancestor closure
before a narrow probe proves a precision cause, substitute random inputs for the
captured failing distribution, or ship a diagnostic subgraph as the deployment
artifact.

## Static-shape MOS control

Use this control when a final ONNX/QPC gate fails for one static shape after
framework-to-ONNX fidelity passes and the compiler reports no graph error,
especially when nearby static shapes pass. Rule out a precision tie first.

Start from the exact failing compile command. Change only `-mos` and write each
candidate to a new, non-existent `-aic-binary-dir`:

```bash
for MOS in 1 2; do
  /opt/qti-aic/exec/qaic-compile \
    <same arguments as the failing compile, except omit its -mos flag> \
    -mos=$MOS \
    -aic-binary-dir="./model_qpc_mos$MOS"
done
```

Keep precision, shape bindings, OLS, core count, node precision information,
quantization profiles, and every other option unchanged. If the failing build
omitted `-mos`, include that default build as a control point.

If one MOS value passes and another fails, record MOS as part of that static
shape's accuracy contract. Revalidate every static specialization independently;
a MOS value proven at one shape is not proven at another. Preserve both artifacts
and compile commands as evidence.

## Stochastic outputs

Sample-level comparison against an independently sampled reference is meaningless
even when the learned computation is correct. Models may sample noise vectors,
latent variables, dropout masks, randomized proposals, or domain-specific
excitation.

Before comparing stochastic outputs:

1. Seed both sides identically immediately before the stochastic step, not only
   at script startup.
2. Use identical static shapes. Padded and exact-length paths can consume
   different numbers of random values and diverge despite the same seed.
3. Feed both sides the same captured upstream tensors instead of recomputing
   them independently.

If these controls are impossible, compare deterministic boundaries before
sampling and use a task-appropriate distributional or semantic metric afterward.
A low sample-wise cosine from independently sampled or shape-mismatched runs is
a harness defect, not a device accuracy result.

## Near-tied discrete outputs

Argmax tokens, class labels, and detected-box ordering can differ when two scores
are closer than the compiled precision can represent. The selection differs; the
arithmetic may not.

At the first mismatch record, on both sides:

- top-1 and runner-up scores;
- their margin;
- both disputed candidates' scores; and
- the number of positions compared before the mismatch.

Compare the reference margin with the compiled format's ULP at the observed
magnitude, for example `numpy.spacing(numpy.float16(score))`. ULP scales with
magnitude; never carry one absolute threshold across models or score scales.

| Evidence at first mismatch | Disposition |
|---|---|
| Reference margin is at or below one ULP and both candidates agree across backends to within roughly one ULP | Precision tie-break. Record it. Do not open a MOS or precision investigation; a sweep tie-breaks arbitrarily and would create a false accuracy contract. |
| Reference margin exceeds one ULP or disputed scores disagree by many ULPs | Real divergence. Continue accuracy debugging; the static-shape MOS control is in scope. |

A tie is not an accuracy pass. Re-run to completion with a criterion that cannot
fail on a tie: top-k overlap, score cosine or relative error, or a task metric such
as WER, mAP, or audio distance. If exact match is required, emit the margin for
every mismatch. Ties often cluster in autoregressive tails and saturated or
low-confidence regions, so a mismatch there is weak evidence of a compute defect.

## Narrow CPU offload evidence gate

This gate applies only to a **new CPU offload inside a failing QPC stage** after
semantic bisection proves a candidate boundary. Natural QPC/host stage boundaries
are outside this decision. CPU offload is not a fallback for an inaccurate learned
stage.

All of these conditions are required:

1. The prefix is deterministic, bounded, non-learned preprocessing/control work,
   one-shot parameter lookup, or other non-hot-loop boundary work.
2. Its boundary has a defined shape, dtype, layout, and value range.
3. The CPU prefix matches the framework boundary on representative inputs at the
   ONNX fidelity threshold.
4. The downstream AI 100 QPC suffix meets the QAIC-to-CPU-ONNX gate when fed
   captured framework boundary tensors.
5. End-to-end accuracy and measured bridge cost meet the model requirements.

Never move learned Conv, MatMul, attention, recurrent, decoder, generator, or
residual blocks to CPU merely to bypass an inaccurate QPC. If the gate fails,
keep the stage on AI 100 and continue graph reformulation, targeted precision,
or custom-op evaluation. Report a stage unrecoverable only after each option has
failed with evidence.

Whole-graph FP32 is not a valid target, diagnostic, or fallback. HMX accepts FP16
or INT8 only, so no FP32 op reaches the matrix engine and a whole-network FP32
compile can be extremely slow. Abandon a long-running FP32 compile. Valid remedies
include `FP16ClipTransform` for proven weight underflow, graph reformulation or
replacement, export restructuring, proven targeted precision, or a custom
on-device op.
