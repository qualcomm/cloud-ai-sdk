# ONNX Graph Cleanup and Surgery

## Contents

- ONNX simplification and constant baking
- Dynamic NonZero and transpose cleanup
- Unsupported-op replacement transforms
- Framework-level patching
- Pre-compilation graph audit

## 3. Graph Cleanup & Surgery (ONNX)

### 3.1 Simplify with onnxsim

`onnxsim` is available as `OnnxSimTransform` in `base_onnx_transform.py`. The default pipeline includes it after `ClipUndefinedInputsTransform` — you do not need to run `python -m onnxsim` as a separate command.

> **Timeout warning:** on transformer-class models with ONNX graphs larger than ~500 MB,
> `OnnxSimTransform` (and therefore the entire export script via `run_pipeline()`) can take
> 30-90 minutes and may OOM inside bash. Run the export script with `task_create`, not `bash`,
> for any model whose raw `.onnx` exceeds 200 MB.

Benefits:

- Fuses BatchNorm into Conv when possible.
- Removes redundant Transposes and no-op patterns.
- Inlines static shape expressions.
- **Topologically sorts the graph** — fixes "undefined node input" errors that `torch.onnx.export` can produce on opset 17+ models.

**`onnxsim` does not replace unsupported ops** — it only simplifies. If `qaic-compile` rejects an op (e.g. `MaxPool3d`, `NonZero`, `ScatterND`), `onnxsim` will not fix it. The op must be replaced with an equivalent sequence of supported ops via a `BaseOnnxTransform` subclass in the export script. See §3.5 for the replacement workflow.

Always use `model_fp16ready.onnx` (produced by `run_pipeline()` in the export script) as the input to `qaic-compile` — never compile the raw export output directly.

### 3.2 Baking Constant Inputs

Use a small helper to insert Constant nodes for shape-affecting attributes:

```python
import onnx
from onnx import helper, numpy_helper
import numpy as np


def bake_constant_tensor(model: onnx.ModelProto, name: str, value_np: np.ndarray) -> onnx.ModelProto:
    const_tensor = numpy_helper.from_array(value_np, name=name + "_tensor")
    const_node = helper.make_node("Constant", inputs=[], outputs=[name], value=const_tensor)
    model.graph.node.insert(0, const_node)
    return model
```

Example: replace dynamic `k` in TopK with a constant value:

```python
k_val = np.array([200], dtype=np.int64)
model = bake_constant_tensor(model, "TopK_k_const", k_val)
# Then adjust the TopK node to consume "TopK_k_const" as its k input.
```

Apply similar logic for `Reshape.shape`, `Slice` indices, `Resize.scales`/`sizes`, `Tile.repeats`, and `OneHot.depth`.

### 3.3 Refactoring Dynamic NonZero

`NonZero` on runtime tensors is often incompatible with efficient QAic execution and may force CPU fallback.

A common pattern:

```text
cond -> NonZero -> Gather -> downstream ops
```

Replace it with a static mask when semantically valid:

```python
# Before: mask = NonZero(cond); y = Gather(x, mask)
# After:  weight = Cast(cond, to=float); y = Mul(x, weight)
```

This keeps the flow on-device without dynamic index computation.

**Always use `MethodPatchTransform` — do not write inline patches:**

When the NonZero is inside a `forward()` method that cannot be replaced by module assignment, always subclass `MethodPatchTransform` from `base_pytorch_transform.py`. Never write inline class-level patching (e.g. `MyClass.forward = lambda ...`) — inline patches are not undoable, not reusable, and leave the class permanently modified if export fails mid-script.

```python
from base_pytorch_transform import MethodPatchTransform
import transformers.models.mymodel.modeling_mymodel as _mod

_orig_forward = _mod.MyDecoderLayer.forward

class FixNonZeroInDecoder(MethodPatchTransform):
    _target_class = _mod.MyDecoderLayer
    _method_name  = "forward"

    @staticmethod
    def replacement(self, hidden_states, attention_mask=None, **kwargs):
        # ... Where-based replacement that avoids NonZero ...
        return _orig_forward(self, hidden_states, attention_mask, **kwargs)

FixNonZeroInDecoder.apply(model)
torch.onnx.export(model, ...)
FixNonZeroInDecoder.undo()
```

Using `MethodPatchTransform` keeps the export script structured, makes patches undoable, and avoids accidental contamination of subsequent imports.

**When the target class cannot be imported cleanly, or the same class has multiple instances that need selective patching — use `ExternalModuleMapperTransform` instead:**

`ExternalModuleMapperTransform` patches at instance level using `types.MethodType`. Unlike `MethodPatchTransform` (which modifies the class globally and requires `undo()`), instance-level patches apply only to matching module objects and require no cleanup. It also supports matching by class name string, so no import is needed:

```python
from base_pytorch_transform import ExternalModuleMapperTransform

def _fixed_forward(self, x, mask=None):
    # ... replacement that avoids NonZero ...
    pass

class FixNonZeroInLayer(ExternalModuleMapperTransform):
    _match_string_replace_method = {
        "MyLayerClassName": {"forward": _fixed_forward},
    }

FixNonZeroInLayer.apply(model)   # no undo() needed
torch.onnx.export(model, ...)
```

Choose `ExternalModuleMapperTransform` when:
- The class is in a private submodule and cannot be imported cleanly (e.g. `transformers.models.foo.modeling_foo.FooAttentionImpl`), or
- The model has multiple instances of the same class and only specific instances need patching (Swin blocks, ViT layers).

### 3.4 Eliminating Redundant Transposes

Double-Transpose patterns waste cycles:

```text
Transpose (NCHW->NHWC) -> Op -> Transpose (NHWC->NCHW)
```

Whenever possible:

- Let the Op run in your primary layout (e.g., NCHW), and
- Avoid layout back-and-forth.

If needed, use **Custom I/O** (see QAic docs) to accept inputs directly in the format your pipeline naturally produces (e.g., NHWC), and avoid host-side Transpose.

### 3.5 Replacing unsupported ops

**Rule: never write ad-hoc graph surgery or inline method patching.** Always check the available transforms first, then subclass:

```bash
python base_onnx_transform.py --list     # post-export ONNX graph surgery
python base_pytorch_transform.py --list  # pre-export PyTorch method/module patches
```

If an existing transform covers the pattern, use it directly. If not, write a new `BaseOnnxTransform` subclass — never raw graph edits outside a transform class, and never attempt an ad-hoc inline fix first "just to check" — checking is done by running `--list` and reading the pattern, not by writing throwaway code.

Consult `../cloud-ai-qaic-compile/references/cloud_ai_onnx_operator_support.md` for the full supported operator table and known limitations. If the model has an unsupported op, replace it with a mathematically equivalent sequence of ops from the supported list. If no suitable standard ONNX replacement exists, implement a custom op using the **`cloud-ai-custom-ops-kernels`** skill.

**Model-specific transforms go in the export script — not in a separate file.** Define the subclass near the top of the export script (after imports), then add it to the `run_pipeline()` call at the bottom, between `OnnxSimTransform` and `FP16ClipTransform`:

```python
from base_onnx_transform import (BaseOnnxTransform, OnnxSimTransform,
                                  FP16ClipTransform,
                                  run_pipeline)

# Step 1 — define the transform (one subclass per unsupported-op pattern)
class MaxPool3dToMaxPool2dTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model):
        # ... pattern match and replace MaxPool3d nodes ...
        return model, changed

# ... export code, framework goldens, torch.onnx.export → "model.onnx" ...

# Step 2 — add the transform to the pipeline
run_pipeline("model.onnx", "model_fp16ready.onnx",
             [OnnxSimTransform,
              MaxPool3dToMaxPool2dTransform,  # ← model-specific, before FP16Clip
              FP16ClipTransform])
```

**Transform ordering inside `run_pipeline()`:**
1. `ClipUndefinedInputsTransform` — always first; removes opset 17+ Clip undefined inputs. Already in the default pipeline — add it explicitly only if building a custom pipeline without defaults.
2. `OnnxSimTransform` — topologically sorts and constant-folds.
3. `GlobalAveragePoolFusion` — replaces ReduceMean global-average-pooling with GlobalAveragePool; run after OnnxSim so shape info is populated.
4. Model-specific transforms — in dependency order (if transform B depends on transform A's output, put A first)
5. `FP16ClipTransform` — always last; must see the final initializer set

```python
run_pipeline("model.onnx", "model_fp16ready.onnx",
             [OnnxSimTransform, GlobalAveragePoolFusion, FP16ClipTransform])
```

Before writing a new transform class, run `python base_onnx_transform.py --list` to see what is already implemented in the base — use existing transforms before writing new ones.

### 3.6 Source-Level Framework Patches

Some frameworks cannot be made ONNX-compatible through submodule replacement or post-export graph surgery alone, because the problematic ops are generated deep inside the framework's own export path — in internal helpers, pooler logic, or proposal utilities that the export script cannot wrap from outside.

**When in-process module replacement is sufficient (preferred):**

If you can replace the problematic submodule by assigning a new `nn.Module` to the parent model:

```python
model.roi_heads.box_roi_pool = NonZeroFreeMultiScaleRoIAlign(...)
```

do that — it requires no patch file and keeps all framework code unmodified.

For models where the problematic module appears in many places (e.g. every Swin block has the same `WindowAttention` class), use **`ModuleMutatorTransform`** to automate the recursive replacement rather than manually traversing the module tree:

```python
from base_pytorch_transform import ModuleMutatorTransform

class ReplaceWindowAttention(ModuleMutatorTransform):
    _match_class = WindowAttentionFused

    @classmethod
    def mutate(cls, original, parent):
        new = WindowAttentionStatic(
            dim=original.dim, num_heads=original.num_heads,
        )
        new.qkv.weight.data.copy_(original.qkv.weight)
        return new

ReplaceWindowAttention.apply(model)
torch.onnx.export(model, ...)
```

`ModuleMutatorTransform` builds a new instance (different `__init__`, repacked weights); use it when `__class__` swap is insufficient.

**When the fix is a known op replacement (GridSample, NonZero in a forward method):**

Use the transforms from `base_pytorch_transform.py` before reaching for source patches:

- **`GridSampleTransform`** — patches `F.grid_sample` globally with a bilinear subgraph safe for ONNX tracing. Covers deformable attention, multi-scale feature sampling, and any module calling `F.grid_sample(...)` with `mode='bilinear'`:

  ```python
  from base_pytorch_transform import GridSampleTransform
  GridSampleTransform.apply(None)   # patches globally — model arg unused
  torch.onnx.export(model, ...)
  GridSampleTransform.undo()
  ```

- **`MethodPatchTransform`** — for NonZero (and other dynamic ops) embedded in a `forward()` method. See §3.3 for the pattern.

Use source patches only when the fix must go inside the framework source (method bodies, config defaults, YAML files, or helper functions with no clean override point).

**Creating a patch file:**

Clone the framework locally, make the edits, then generate with `git diff`:

```bash
git clone <framework-repo> /tmp/<framework> && cd /tmp/<framework> && git checkout <sha>
# make edits
git diff HEAD > /workspace/<model>/patches/<framework>_onnx_export.patch
```

Naming convention: `patches/<framework>_onnx_export.patch`, placed in the model workspace.

**Applying the patch (inside the qaic container):**

Clone at the pinned SHA into the workspace, apply the patch, then install in editable mode
so the export script imports the patched source:

```bash
git clone <framework-repo> /workspace/<model>/src/<framework>
cd /workspace/<model>/src/<framework> && git checkout <sha>
git apply /workspace/<model>/patches/<framework>_onnx_export.patch
pip install -e /workspace/<model>/src/<framework>
```

Key rules:

- **Pin the SHA** — the patch is written against a specific commit; a floating `HEAD` will drift and `git apply` will fail with context mismatches.
- **Use `git apply`**, not `patch -p1` — `git apply` validates cleanly and fails loudly on context mismatch rather than creating silent `.rej` files.
- **`-e` install** after applying the patch so Python imports the patched source, not a stale wheel.

> **Record in MODEL_CARD.md:** the `## ONNX Export → Framework Source Patches` table must
> capture the framework name, pinned SHA, patch filename, and reason for patching. A reader
> reproducing the export needs all four to recreate the exact patched environment.

**Regenerating a patch after further edits:**

```bash
cd /workspace/<model>/src/<framework>
git diff HEAD > /workspace/<model>/patches/<framework>_onnx_export.patch
```

If the upstream framework releases a new version and the patch needs rebasing:

1. Clone fresh at the new SHA.
2. Apply the old patch: `git apply patches/old.patch` (resolve conflicts if any).
3. Re-generate: `git diff HEAD > patches/<framework>_onnx_export.patch`.
4. Update the pinned SHA in the MODEL_CARD and in any references to it.

---

## Pre-compilation Graph Audit

### Step 2: Clean the Graph

**`onnxsim` is built into `base_onnx_transform.py` as `OnnxSimTransform`** — no separate `python -m onnxsim` step needed. Include it in `run_pipeline()` for clean exports.

> **Do not inspect, validate, or apply any transforms to the raw ONNX before calling `run_pipeline()`.** If `onnx.checker` reports a topological sort error or undefined node inputs on the raw export, that is expected for some opset 18+ models — `ClipUndefinedInputsTransform` fixes undefined Clip inputs, and `OnnxSimTransform` (when included) fixes topological sort errors. Running manual surgery before `run_pipeline()` is never the right response to a raw-export validation error.

**Audit the graph before compiling.** After the export script runs and produces `model_fp16ready.onnx`, enumerate every op type and cross-reference against the supported operator table in `../cloud-ai-qaic-compile/references/cloud_ai_onnx_operator_support.md`. Do this proactively — catching unsupported or problematic ops now avoids a compile-fail-fix-recompile loop. If you are unfamiliar with the hardware constraints driving the flags below (VTCM budget, tiling, HMX-unfriendly activations), read the `cloud-ai-architecture` skill before proceeding.

```python
import onnx
from collections import Counter

m = onnx.load("model_fp16ready.onnx")
op_counts = Counter(n.op_type for n in m.graph.node)
for op, count in sorted(op_counts.items()):
    print(f"{op:30s} {count}")
```

Flag any op that appears in the following categories:

**Will fail to compile (unsupported or requires constant inputs):**
- `NonZero` on a runtime tensor — must be constant or replaced; use `MethodPatchTransform` (§3.3) for forward-method fixes or a `BaseOnnxTransform` subclass (§3.5) for post-export ONNX surgery
- `Loop` / `If` with dynamic trip count or condition
- `Upsample` at opset ≥ 10 — replace with `Resize`
- `Resize`, `Reshape`, `Slice`, `Pad`, `Tile`, `TopK`, `OneHot` fed by runtime-computed values — bake as constants (see §3.2 and the full constant-input table in `../cloud-ai-qaic-compile/references/cloud_ai_onnx_operator_support.md` §2)
- `Conv` 3D with dilation, `AveragePool`/`MaxPool` with `ceil_mode=True`, `Einsum` with > 2 operands or ellipsis broadcast, `ScatterElements` with non-2D inputs, `QuantizeLinear` with per-channel scale
- `CumSum`: only axis 0 is supported in all modes. Axis 1 and 2 work only for 2D/3D inputs without `exclusive` or `reverse`. Everything else fails at compile time — decompose or restructure before compiling.

**Supported but inefficient for Cloud AI architecture — fix before compiling:**
- 2D `Conv` with HMX-ineligible dilation geometry — equal x/y dilation <=2 can map to HMX when stride is 1, but unequal dilation, dilation >2, or dilation >1 with stride >1 falls back to HVX. If the model needs those dilations, treat the warning `dilation too large or dilation x/y dims don't match` plus `aicconvhvx` in opstats as a model/export optimization opportunity, not a compiler-flag problem.
- Excessive `Transpose` chains — a single NCHW↔NHWC conversion at the model boundary is fused efficiently into the adjacent op; repeated back-and-forth conversions are not and block Conv fusion. Remove redundant transposes (see §3.4) or use Custom I/O (`cloud-ai-qaic-compile` skill §5.1) to supply inputs in the layout your pipeline naturally produces.
- `Gather` / `GatherElements` on large tensors — small gathers with ≤ 64 indices that fit in VTCM hit the hardware-accelerated path; larger gathers fall back to an unaccelerated memory copy. If the gather is driven by a sparse or conditional selection, restructure as a masked multiply (element-wise multiply by a 0/1 weight tensor) to stay on the vectorized path.
- `NonMaxSuppression` — IoU scoring and score filtering are hardware-accelerated; per-class box suppression is sequential and does not scale with additional cores. For models with few classes or low box counts this is typically fine. For high-class-count or high-box-count models where NMS dominates latency, move it off-device (host CPU) or replace with a batched approximation.
- `TopK` on a non-last axis — last-axis FP16 TopK runs an optimized sort path and is efficient; non-last-axis TopK does not. Insert explicit Transpose→TopK→Transpose in the export script to normalize to the last axis before compiling.

Then verify against `cloud-ai-performance-optimization` Step 1, **Check graph and compile prerequisites**, using its `references/graph-optimization.md` checklist.

For dynamic ops identified above, add a `BaseOnnxTransform` subclass to the export script (see §3.5) to replace dynamic ops (NonZero, TopK, Pad) with static equivalents. See §3 in this skill for surgery patterns.

If after surgery `qaic-compile` still reports an unsupported op, check `../cloud-ai-qaic-compile/references/cloud_ai_onnx_operator_support.md` for the full operator table and fix options. If no standard ONNX replacement exists, implement a custom op using the **`cloud-ai-custom-ops-kernels`** skill.
