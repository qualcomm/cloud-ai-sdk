# QAIC Model Requirements

## Contents

- Static and symbolic shapes
- Constant inputs for shape-affecting operations
- Valid-length inputs for padded variable-length dimensions
- Transformer attention follow-on rules

## 2. Model Requirements for QAic

Cloud AI 100 requires all shapes to be concrete at compile time. Any dim can be made concrete in one of two ways: bake it into the graph as a static value, or leave it as a named ONNX symbol and resolve it at compile time via `-onnx-define-symbol=name,value`. For each axis, make a decision about what would be useful for the downstream consumer to configure — dims that will be varied or swept (batch size, sequence length, image resolution if variable, etc.) must be left as named symbols; dims that are fixed for all deployments can be baked. The batch axis is always configured downstream: it must always be a named symbol — never post-process the ONNX to replace the batch axis with a concrete value.

### 2.1 Static Shapes

- All input tensors must have fully known dimensions.
- No symbolic dims, unless you define them with `-onnx-define-symbol` at compile time.

Check shapes with symbol-aware printing — do not treat `dim_value == 0` as a shape to fix before checking `dim_param`:

```python
import onnx

m = onnx.load("model.onnx")

def describe_dim(dim):
    if dim.HasField("dim_param"):
        return dim.dim_param
    if dim.HasField("dim_value"):
        return dim.dim_value
    return "unknown"

for inp in m.graph.input:
    dims = [describe_dim(d) for d in inp.type.tensor_type.shape.dim]
    print(inp.name, dims)
```

If your model uses symbolic dims (e.g., `batch_size`, `seq_len`), plan to supply concrete values at compile time (see `cloud-ai-model-conversion/references/compiling-to-qpc.md`, "ONNX Compilation Example"). Do not post-process the ONNX to replace symbolic dims that the downstream consumer should configure.

**Symbolic batch is a hard export invariant for the raw export.** Do not set `dynamic_axes=None`, remove `batch_size` from outputs, or bake batch=1 to work around `Shape`/`Gather`/`Concat`/`Reshape`/`Squeeze`/`Unsqueeze` importer errors. Those errors mean the shape or axis subgraph needs a targeted `BaseOnnxTransform`, not that the raw export should become static. If a model-specific transform must bake shape tensors for compilation, parameterize it with `TARGET_BATCH` so the same export script can generate batch-specific ONNX files; never hardcode `batch_size -> 1` in reusable transforms. Performance sweep details belong in `cloud-ai-performance-optimization` → `batch_size_tuning.md`.

### 2.2 Constant Inputs for Shape-Affecting Ops

qaic-compile performs static shape inference at compile time. Any op that uses a tensor input to determine output shape, reduction axes, or data-dependent indexing must have that input as a **constant initializer** — not a graph input or computed tensor.

Apply this principle to every op with shape/axis/index inputs. Common examples (not exhaustive):

- `Reshape`: `shape` must be a Constant
- `Slice`: `starts`, `ends`, `axes`, `steps` must be Constants
- `Resize`: `scales` or `sizes` must be a Constant
- `Pad`: `pads` must be a Constant
- `Tile`: `repeats` must be a Constant
- `TopK`: `k` must be a Constant
- `OneHot`: `depth` must be a Constant
- `ReduceMean`: `axes` (second input, when present) must be a Constant. When ReduceMean is used for global average pooling and constant axes are not sufficient for the compiler to infer the output shape, use `GlobalAveragePoolFusion` from `base_onnx_transform.py` — it is included in the default pipeline and replaces the ReduceMean with `GlobalAveragePool`, whose output shape the compiler always infers correctly.
- `Gather.indices` when used for shape manipulation must be a Constant

If any parameter input is built from `Shape`/`Gather`/`Concat` chains or is a runtime-computed tensor, add a model-specific `BaseOnnxTransform` subclass to the export script and include it in the `run_pipeline()` call. The compiler error for this class of bug typically says *"input shape X and output shape Y have different number of elements"* or *"unable to infer output shape"* — the node named in the error is usually the downstream consumer of the non-constant parameter, not the parameter source itself.

### 2.3 Valid-Length Inputs for Padded Variable-Length Dimensions

**Litmus test — for each input dimension, ask: can a real input be smaller than the compiled dimension?** If yes, the true per-request size MUST be passed into the graph and factored into the computation — via an `input_lengths` tensor (build `mask = arange(seq) < input_lengths` inside the graph) or an explicit boolean `attention_mask` input — and attention, pooling, embeddings, and reductions must be driven from it. This is general to **any encoder or decoder with a variable-length axis padded to a fixed maximum** — tokens in a text encoder / embedding / reranker, frames in a speech or audio model, points in a point-cloud / set model, steps in a time-series model, or the sequence axis of any seq2seq model.

A fixed compiled shape is an **upper bound**, not a promise that every position is real. Baking an all-true mask, or omitting the mask, tells the model the padding is valid content. The symptom differs by architecture but is always wrong: in an **encoder**, padded positions pollute attention and pooled/reduced outputs (wrong embedding or logits); in a **decoder / generative** model, the model treats padding as real context and cannot stop at the true endpoint (trailing repeats or filler). An all-true mask equals `mask=None` **only** for a fully valid, unpadded sequence — which a large fixed upper bound is not. If a ported reference export hardcodes `mask = torch.ones(...)` "for tracing" or "== mask=None", that equivalence breaks the moment you pad to the max — replace it with a length-derived mask.

This mirrors the efficient-transformers (QEfficient) LLM pattern: compile to fixed `prefill_seq_len`/`ctx_len`, then pad `input_ids` and pass a padded `attention_mask`/`position_ids` (pad marked `0`/`-1`) so pad positions are ignored. Use the same pattern for any fixed-shape model with variable real length.

**Verify the mask is consumed — presence is not proof.** A declared `input_lengths`/`attention_mask` can still be silently ignored: a config flag gating attention off (e.g. `attn_mask_enabled=False`), a backend branch, or a reduction over all positions leaves it a *dead input* with the padding bug intact. Confirm behaviorally before compiling: on a genuinely padded (short) input, hold the valid prefix fixed and vary only the padded tail (or swap the true mask for an all-true one); the valid-region output MUST be unchanged by the tail and MUST differ from the all-true case. If not, the mask is not wired through — trace where it is dropped. And run fidelity/accuracy validation (the cosine golden check) on at least one padded input, never only full-length: a full-length golden cannot catch a broken mask, because full-length is exactly where an all-true mask is correct.

**When the litmus is NO, add nothing.** A dimension whose full extent is always real content never triggers it — an image is always supplied at the full compiled H×W (source images are resized/padded during preprocessing, so the tensor is never smaller than the shape), fixed feature dims are always full, and a genuinely fixed-length sequence is always full. Batch is never a validity dimension. Do not add `input_lengths`/`attention_mask` in these cases.

### 2.4 Attention-Specific Follow-On (Transformer Models Only)

**If §2.3 applies to a self-attention or cross-attention layer**, read `cloud-ai-architecture/references/transformer_architecture.md` — its transformer optimization triage (including Rule 4 broadcast-mask + blocked attention) applies at export time, not just during performance debugging. Framework SDPA with `attn_mask` expands to `[B, H, Q, K]` per attention block: a 10× regression that no compiler flag fixes.
