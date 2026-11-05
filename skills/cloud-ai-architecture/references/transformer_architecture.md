# Transformer Graph Optimization for Cloud AI 100

Hardware-aware ONNX graph transforms for attention-based models. Each technique is a functional-equivalent graph rewrite that improves execution on Cloud AI 100's HMX/HVX architecture.

**Implementation level:** Each transform can be applied at PyTorch export time (modifying the export script before `torch.onnx.export`) or post-export (editing the ONNX graph directly). Neither is assumed — choose based on what is simpler for the specific transform. One constraint applies to all post-export ONNX rewrites: **apply them before running onnxsim** — onnxsim renames and fuses nodes, breaking patterns written against the original exported graph.

**Reference implementations — QEfficient.** Rules in this doc are tool-agnostic Cloud AI 100 hardware-optimization rules. Where they cite QEfficient as a canonical implementation, they defer the class name and file location to `cloud-ai-llm-onboarding/references/qefficient_transformers_details.md` — the canonical QEfficient inventory. Read the pattern description here for *why* and *what*, then consult that inventory for the current QEfficient class name, then read the source (see `cloud-ai-llm-onboarding` for how to locate the checkout without installing). Port the *pattern* to the target model's own module names; do not vendor QEfficient source into the export script.

---

## Rule 1 — Reduce Memory-Related Operators

Reshape and Transpose nodes on the hot path break the HMX→HVX tile pipeline. Each one between an HMX op output and its HVX consumer inserts an explicit sync event: HMX issues a tile, HVX must drain the interstitial op before HMX can start the next tile.

### 1.1 MHA → Single-Head Attention (SHA) Decomposition

**What:** Split the multi-head attention block into N independent single-head blocks before compiling. Unroll `[B, N_heads, seq, dim]` batch MatMuls into N separate `[B, 1, seq, dim]` MatMul sequences, one per head, with a Concat at the output.

**Why:** When N_heads does not divide evenly across the core count, the compiler creates uneven partitions — some cores receive disproportionate Softmax load while others idle at a cross-head barrier. SHA lets the compiler assign exactly one head per core, balancing the load and eliminating cross-head barrier synchronization overhead.

**Note on barrier overhead as a predictor:** The `barrier HMX + barrier HVX` percentage in opstats understates the true cost of head/core imbalance. Uneven partitioning distorts HMX tile scheduling across the entire inference, not only at explicit sync points — expect the actual throughput gain to be significantly larger than the barrier percentage alone suggests. Apply SHA for all transformer models and benchmark early; gains are largest when `N_heads % N_cores != 0`.

---

### 1.2 Input Layout Pre-Shaping

**What:** Reorder Q/K/V weight matrices so their output is already in the layout that downstream ops expect. The Transpose nodes are removed from the graph; the weight initializers absorb the permutation offline.

**Why:** Runtime Transpose on a large tensor requires multiple HVX gather/scatter passes through vector registers. When VTCM is near capacity, the transpose intermediate must be staged via an intra-VTCM copy, adding further cycles. Pre-shaping the weight initializer at export makes the permutation a one-time offline operation with zero inference cost.

---

### 1.3 H/W Transpose Flip for Attention Inputs

**What:** For NCHW/NHWC inputs where H ≠ 1 and W = 1, transpose the input offline so H = 1. Where possible, make H a multiple of 8.

**Why:** Cloud AI 100 processes tensors in crouton memory layout — a depth-first tiling format that requires the spatial dimensions to satisfy minimum alignment for efficient HMX tile assignment. H ≠ 1 with W = 1 does not map to a crouton-aligned shape, causing the compiler to insert a runtime Transpose. Moving this offline eliminates that op from the inference graph.

---

### 1.4 Move Preprocessing Offline

**What:** Any subgraph that takes a graph input tensor directly and performs only layout or element-wise ops (Reshape, ElementWiseMul, ElementWiseAdd, Div) before the first Conv/MatMul can be moved offline. Precompute the result for the fixed input shape and pass it as a static initializer to the first Conv/MatMul, removing the subgraph from the inference graph.

**Why:** For a fixed-shape compiled QPC, any deterministic subgraph that depends only on input shape — not on input values — is a constant. Running it at inference time adds HVX dispatch overhead and synchronization points on every inference for zero benefit.

---

### 1.5 Split Batch Dimension Offline

**What:** If the model receives inputs with batch dimension > 1, split into batch-1 slices offline before passing to the graph rather than relying on on-device Split nodes.

**Why:** When the compiler handles batch > 1 via SplitPlan, it inserts on-device AICConvertToD32/FromD32 convert nodes to expand and contract the batch dimension at runtime. This adds conversion overhead per inference. Additionally, certain graph-level optimizations in the compiler (e.g. concat fusion) have explicit batch=1 restrictions. Pre-splitting offline avoids these limitations and keeps the compiled graph in the simpler single-inference code path.

---

### 1.6 Precompute Rotary Position Embedding Caches Offline

**What:** For any transformer with rotary position embeddings (RoPE), precompute the `sin` / `cos` frequency tables once at model init and register them as static buffers on the model. In the exported forward, index into those precomputed tables using `position_ids` (or `arange(seq_len)`) instead of recomputing sin/cos from `theta`, `inv_freq`, or `torch.outer` on every forward.

**Why:** The freq/sin/cos math depends only on model config (`base`, `head_dim`) and sequence length — all statically known for a fixed compiled QPC. Leaving it inside `forward()` traces to a subgraph of `Range → Cast → Div → Mul → Sin/Cos → Reshape` that runs on HVX every inference for zero benefit and adds sync points between HMX matmuls and their downstream ops. Moving it to model init makes it a one-time offline computation and an ONNX weight initializer.

**Reference implementation to model after:** QEfficient encodes this as its "RoPE optimization" — see `cloud-ai-llm-onboarding/references/qefficient_transformers_details.md` for the current class name (a subclass of upstream `*RotaryEmbedding` that registers `sin_cached`/`cos_cached` buffers and exposes a `qeff_apply_rotary_pos_emb` helper). The pattern is: subclass the upstream `*RotaryEmbedding`, compute freqs in `__init__`, register `sin_cached`/`cos_cached` as non-persistent buffers, and rewrite the attention forward to index those buffers by `position_ids` instead of recomputing. For non-LLM transformers going through the generic export path, apply this via a `MethodPatchTransform` or model-specific `__init__` monkey-patch in the export script.

**Also applicable to non-RoPE learned positional embeddings:** any positional table that depends only on config and sequence length (learned absolute embeddings, ALiBi bias tables, sinusoidal fixed embeddings) should be a static initializer, not recomputed in `forward()`.

---

## Rule 2 — Run Operators with Efficient Tensor Shapes

### 2.1 Reshape 1D Sequence → 2D Spatial Tile

**What:** Before any batch MatMul or attention op with a 1D sequence dimension, reshape `seq_len` into a 2D tile where both dimensions are ≥ 8:

```
seq_len  →  dim_a × dim_b   (both dim_a ≥ 8, dim_b ≥ 8)
```

Reshape back to 1D after the downstream Softmax before passing to subsequent ops.

**Why:** The HMX tile unit requires ≥ 8 elements along any spatial dimension to fully utilize its MAC arrays. A 1D sequence dimension forces sequential single-strip tile issue with no inter-tile overlap. A 2D factoring enables parallel tile issue along the outer dimension.

A second effect on models with long sequences: the QK attention score matrix grows quadratically with sequence length and can exceed VTCM capacity, forcing tile-by-tile DDR eviction. Tiling the sequence as 2D allows the compiler to schedule smaller 2D tiles that fit in VTCM, reducing or eliminating that DDR round-trip.

**Verification:** After applying this transform, compare the compiled QPC checksum to the untransformed baseline. The compiler may absorb identity-equivalent Reshape→op→Reshape sequences during graph optimization, producing an identical QPC. If checksums match, the transform had no effect; the optimization opportunity may not exist for this model.

---

### 2.2 Avoid ≥5D Tensor Shapes

**What:** Inspect all Reshape output shapes in the ONNX graph. Rewrite any op that introduces a 5th dimension as a 4D equivalent, or decompose it offline.

**Why:** The compiler's tiling engine is designed around 4D tensors. 5D+ shapes prevent crouton-aligned tile assignment and force less-efficient fallback execution paths.

---

## Rule 3 — Operator Transforms for Hardware-Native Forms

### 3.1 Static MatMul → 1×1 Conv2D

**What:** Replace every MatMul where one operand is a constant weight matrix with a 1×1 Conv2D. Applies to Q/K/V projections, output projection, and FFN layers.

For very large linear layers (hidden_dim where a single op creates a weight tile that exceeds per-core VTCM budget), further split into two parallel 1×1 Conv2D ops feeding the same output.

**Why:** When a weight is expressed as a named Conv2D kernel initializer, the compiler can pre-convert it to D32 format once at model load. When expressed as a MatMul initializer, the compiler re-executes weight-to-D32 conversion per tile per inference. The explicit Conv2D form also provides per-output-channel quantization range granularity during INT8 PGQ, improving accuracy recovery. The split-large-linear variant improves core parallelism by allowing the compiler to distribute smaller weight tiles across more cores independently.

---

### 3.2 Dynamic Batch MatMul → Mul + Batch2Depth + 1×1 Conv

**What:** For attention score MatMuls where neither operand is a static weight (Q×K and Score×V), replace with this sequence:

| Step | Transform | Purpose |
|------|-----------|---------|
| 1 | Element-wise Mul with broadcasting | Compute pairwise products |
| 2 | Batch2Depth: Split on batch dim + Concat on depth dim | Regroup head-batch products |
| 3 | 1×1 Conv2D with identity-pattern weight | ReduceSum across depth groups |

**Why:** The unmodified batch MatMul materialises the full attention score matrix in DDR in a single op before the downstream Softmax can read it. For long sequences this matrix is large — it grows quadratically and cannot reside in VTCM. The Mul+Batch2Depth+Conv decomposition computes the same result tile-by-tile, keeping intermediate values in VTCM and HVX registers, eliminating the full-matrix DDR write.

**Implementation note:** Batch2Depth is not a standard ONNX op. Implement as an explicit Split + Concat + Reshape sequence. The sequence dimension must be reshaped to 2D (§2.1) before applying this transform.

---

### 3.3 Scalar Div Fusion (Scale Factor Absorption)

**What:** Fold constant scalar Div nodes into the weight initializer of the preceding MatMul/Conv. Locate each Div whose second input is a constant scalar and whose first input is a MatMul/Conv output. Multiply the MatMul/Conv weight initializer by the reciprocal of the scalar. Remove the Div node.

Common instances:
- Attention scale `÷ √d_k` — fold into Q-projection weight
- GELU denominator `÷ √2` — fold into FFN-intermediate weight **only if** the Div feeds a single downstream op (Erf) and the Div's first input is solely the MatMul/Conv output with no other consumers. If the Div first input is also consumed by a Mul or other op (as in full GELU: `MatMul+bias → Div → Erf → Mul → Mul`), the `÷√2` cannot be folded into the weight — doing so would change the Erf argument but leave the outer `x` in `x * 0.5 * (...)` unscaled, producing incorrect results. In this case mark §3.3 Deferred and proceed to §3.4.

**Why:** A Div node between an HMX MatMul output and a downstream HVX op (Softmax, Erf) requires a scalar-unit dispatch on the Hexagon NSP, inserting an explicit HMX→HVX sync on every tile of every layer. Folding into the constant weight initializer makes it a one-time offline multiply — the Div op disappears from the inference graph entirely.

**Do not attempt to fold Div into the bias alone** — the Div applies to the full MatMul output (dynamic + bias), so folding into only the bias changes semantics.

---

### 3.4 Fused GELU Kernel

**What:** The GELU activation expands to `Div → Erf → Mul → Mul` in ONNX — four separate HVX ops per FFN block per layer. Replace with a single custom HVX kernel that executes all four operations in one pass.

**Why:** Each of the four ops reads and writes the full FFN intermediate tensor. When that tensor exceeds VTCM capacity, each op produces an unnecessary DDR round-trip. A fused kernel keeps all intermediate values in HVX vector registers (1024-bit per lane), eliminating the intermediate writes. The Hexagon VLIW engine can pipeline the polynomial Erf approximation and multiply operations in a single instruction packet, further reducing cycle cost.

Note: After applying §3.3, the leading Div is already folded into the weight. The fused kernel then covers Erf + Mul + Mul only.

---

### 3.5 LayerNorm Fusion

**What:** Verify that LayerNorm is compiled as a single fused op. If the ONNX graph contains the scattered element-wise form (`ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`), replace it with a single `LayerNormalization` ONNX op (opset 17) in the export script. When `affine=False`, insert dummy γ=1.0 and β=0.0 to satisfy the op interface.

**Why:** The compiler fuses `LayerNormalization` into a single HVX kernel. The scattered form runs as separate HVX dispatches with sync points between each sub-op.

**Check first:** If the opstats summary already shows a single fused LayerNorm op, the compiler has already handled this — no action needed.

---

### 3.6 RMSNorm → CustomRMSNorm Custom Op

**What:** For any transformer that uses `RMSNorm` (Llama, Mistral, Qwen, Gemma, T5-style, or any model with `x * rsqrt(mean(x^2) + eps) * weight`), replace the module with the Cloud AI custom op form so the compiler emits a single fused HVX kernel. In ONNX, the target is a `CustomRMSNorm` node in domain `com.qti.aisw.onnx`.

**Why:** The naive PyTorch RMSNorm exports to `Pow → ReduceMean → Add → Sqrt → Reciprocal → Mul → Mul` — six or more separate HVX dispatches with sync points between them, and one FP32→FP16 accumulator round-trip through the `ReduceMean`. The custom op form compiles to a single HVX kernel with intermediate values held in vector registers, avoiding both the sync overhead and the intermediate DDR/VTCM writes.

**Reference implementation to model after:** QEfficient encodes this as its `CustomOpsTransform` (which swaps RMSNorm module classes for a custom-op-emitting variant) plus a custom op kernel in `QEfficient/customop/rms_norm.py` — see `cloud-ai-llm-onboarding/references/qefficient_transformers_details.md` for the current class name and the `CustomRMSNorm` ONNX op domain. The pattern is: subclass or monkey-patch the model's RMSNorm class with a version whose `forward()` calls `torch.autograd.Function.apply()` bound to an `onnxscript` symbolic that emits the `CustomRMSNorm` op. In the generic export path, apply this as a `MethodPatchTransform` on the RMSNorm module class before `torch.onnx.export`, similar to `LayerNormalization` in §3.5.

**Check first:** If opstats top-ops shows a single `rmsnorm` HVX op, the compiler has already picked it up — no action needed. If you see a chain of `pow / reducemean / rsqrt / mul` in top-ops, the custom op form is not being emitted.

---

### 3.7 Split Fused Gate-Up Projection Weights

**What:** For transformers with a gated FFN (SwiGLU / GEGLU / SiLU-gated: Llama-3, Mistral, Qwen, Gemma, etc.), the reference implementation often loads a fused `gate_up_proj` weight tensor and splits it inside `forward()` via `torch.chunk(x, 2, dim=-1)` or an index expression. Rewrite the model so it holds separate `gate_proj` and `up_proj` weight tensors, halved offline. The forward becomes two independent `Linear`/`MatMul`/`Conv2D 1×1` calls whose outputs feed the SiLU + elementwise Mul.

**Why:** A fused `gate_up_proj` forces the compiler to compile one wide MatMul followed by an on-device Split, then two downstream branches. Splitting the weight offline gives two independent MatMuls of half the output width, which:
- The compiler can schedule concurrently onto different cores (better core utilization).
- Removes the intra-forward `Split` node (a sync point between HMX and the SiLU HVX op).
- Halves the per-op VTCM footprint, giving room for larger tile counts or higher `-ols`.
- Aligns with §3.1 (MatMul → 1×1 Conv2D) applied twice, once per branch.

**Reference implementation to model after:** QEfficient encodes this as its `SplitGateUpWeightsTransform` (in the base pytorch_transforms module) — see `cloud-ai-llm-onboarding/references/qefficient_transformers_details.md` for the current class name. The pattern is: walk the state dict, find each `gate_up_proj` (or `gate_up_proj_weight`), split it along the fused dim, copy halves into new `gate_proj` / `up_proj` tensors, and (for models with fused biases) split biases the same way. Applied as a bespoke `PytorchTransform` before ONNX export.

**Check first:** If the model architecture code already has separate `gate_proj` and `up_proj` linears (some HuggingFace configurations do), skip §3.7. Grep the ONNX for a MatMul immediately followed by a Split feeding a SiLU — that's the smoking gun.

---

## Rule 4 — Attention Mask Representation

For models with attention masks over a padded variable-length sequence (see `cloud-ai-model-conversion/references/qaic-model-requirements.md`, "Valid-Length Inputs for Padded Variable-Length Dimensions"), *how* the mask reaches attention determines whether the export is fast or memory-bound. Rule 3's transforms optimize the attention math itself; this rule optimizes the mask consumption path around it.

### 4.1 Broadcast Mask + Blocked Query Attention

**What:** For fixed-shape attention with a per-request validity mask, do not export framework SDPA (PyTorch `F.scaled_dot_product_attention`, HuggingFace `SdpaAttention`, diffusers `AttnProcessor`) with the `attn_mask` argument. Replace the model's attention forward at export time with an explicit implementation that:

1. **Represents the mask in broadcast shape** — `[B, 1, 1, K]` for self/cross-attention over K — materialized once per forward, not expanded to `[B, H, Q, K]`.
2. **Applies the mask as an additive bias** — masked positions → large negative (`-65504.0` in FP16) added to scores before `softmax`. Broadcasting handles the head/query axes at compute time; no expansion tensor is written to memory.
3. **Blocks the query dimension** — tile Q into chunks (typically `block_size=64`) so the per-block score matrix `[B, H, block_size, K]` fits in VTCM and streams rather than materializing full `[B, H, Q, K]` in DDR.

**Why:** Framework SDPA traced with a validity mask typically compiles into `mask.unsqueeze(1).unsqueeze(1).expand(B, H, Q, K)` and materializes both the mask tensor and the full attention score matrix in memory. For `Q = K = 2048`, `H = 16`, FP16, this is ~268 MB of mask + scores per attention block per forward. Multiplied across a 22-block transformer × 32 sampling steps, hundreds of gigabytes of DDR traffic per inference. The mask is fundamentally a `[B, K]` validity signal — expanding it to 4D is wasted DDR bandwidth. Broadcast + blocked attention keeps working set in VTCM.

**Interaction with Rule 3.2 (Dynamic Batch MatMul → Mul + Batch2Depth + Conv):** §3.2 avoids materializing the full attention score matrix. §4.1 avoids materializing the full attention mask tensor. They target different objects; apply both for masked attention.

**Reference implementations** (do not adapt lightly — match numerics exactly, then run the equivalence check below):

- **QEfficient** encodes this as its `BlockingAttentionTransform` and the `blocked_*_attention_forward` helpers (KV, Q, QKV, HQKV, BHQKV modes selected via an `AttentionBlockingConfig`). See `cloud-ai-llm-onboarding/references/qefficient_transformers_details.md` for the current class names and their location in the QEfficient tree.

**Detection signals in an existing export:**

- **Compile.log**: per-core `DDR traffic write` in hundreds of MB, total DDR traffic in gigabytes per inference on a transformer-class model.
- **ONNX graph**: `Expand` nodes producing 4-D shapes with the sequence axis appearing twice (i.e. `[..., Q, K]`) once per attention block. Grep the ONNX by op-type — one Expand per block on a 22-block transformer is a smoking gun.
- **Runtime opstats**: attention-layer share of top-ops disproportionate to arithmetic cost; `sync HMX`/`sync HVX` alongside high `DMAIssue` utilization.

**Validation for the replacement attention (required before compile):**

Run one attention block on identical inputs through both the original masked forward and the replacement blocked forward. Compare valid-region outputs: cosine similarity must be > 0.9999. Anything lower indicates a real numerics bug (scaling forgotten, rotary axis mismatch, mask polarity wrong, softmax NaN handling); do not compile until this passes. This one-block equivalence check is much cheaper than debugging a wrong full-model QPC.

---

## Triage Order

Apply techniques in this sequence. Earlier items are lower risk, validate correctness, and unblock later transforms.

1. **§3.5** LayerNorm fusion — verify explicitly that the compiler produced a single fused kernel; do not assume it did. If not fused, apply the ONNX-level fix before any other transform.
2. **§3.6** RMSNorm → CustomRMSNorm — if the model uses RMSNorm, apply the custom op form (mirrors §3.5 for RMSNorm-based models); zero shape dependency; validate ONNX output on CPU.
3. **§2.2** ≥5D tensor check — pre-flight inspection before graph surgery. 5D+ tensor shapes can prevent §3.1 (MatMul→Conv2D) from succeeding; identify and fix them first.
4. **§1.6** RoPE / positional embedding precompute — if the model uses RoPE (or any position-dependent embedding table), move the sin/cos (or table) computation offline into a static buffer. Weight-space edit; zero shape dependency; validate ONNX output on CPU.
5. **§3.7** Split fused gate-up projection — if the model uses a gated FFN with a fused `gate_up_proj` weight, split offline. Weight-space edit; validate ONNX output on CPU. Skip if the model already has separate `gate_proj`/`up_proj` linears.
6. **§4.1** Broadcast mask + blocked query attention — REQUIRED whenever the model has an attention mask over a padded variable-length axis (`cloud-ai-model-conversion/references/qaic-model-requirements.md`, "Valid-Length Inputs for Padded Variable-Length Dimensions" applies). Framework SDPA with `attn_mask` expands to `[B, H, Q, K]` at compile time; this must be replaced before any other perf tuning is meaningful, because the DDR cost dominates every other optimization on large-seq attention. Validate with the one-block equivalence check in §4.1 before compiling.
7. **§1.1** MHA → SHA decomposition — apply to all transformer models and benchmark; gains are largest when `N_heads % N_cores != 0` but barrier % in opstats understates the true gain regardless of head/core ratio.
8. **§3.3** Scalar Div fusion — weight-initializer edit only; zero shape dependency; validate ONNX output on CPU.
9. **§3.1** Static MatMul → 1×1 Conv2D (incl. large-layer splitting) — weight-space rewrite, layout-independent; must precede layout transforms so Conv tile dimensions are known before §1.2/§1.3 are applied.
10. **§1.2 + §1.3** Transpose pre-shaping and H/W flip — merge into one step; apply after Conv tile dimensions are established from §3.1.
11. **§2.1** Sequence 2D reshape — prerequisite for §3.2; validate output correctness on CPU before proceeding. Verify QPC checksum before/after — if unchanged, the compiler absorbed the Reshape pair during graph optimization and the transform had no effect.
12. **§3.2** Dynamic Batch MatMul decomposition — implement after §2.1 is validated.
13. **§1.4 + §1.5** Move preprocessing offline / split batch dimension — offline input moves are last graph-level edits before custom ops; they depend on final graph shape being stable.
14. **§3.4** Fused GELU custom kernel — custom op; implement last, after all graph surgery gains are measured.

---

## Transformer Optimization Report

**Include this filled-in table in your graph optimization report.** Work through every row in triage order before declaring graph optimization complete. Do not skip a row — mark it Applied, Not applicable, or Deferred (with reason).

```
Transformer Optimization Checklist
───────────────────────────────────────────────────────────────────────────────
1. §3.5  LayerNorm fusion verified (compiler produced single fused kernel)?
         [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

2. §3.6  RMSNorm → CustomRMSNorm custom op applied (if model uses RMSNorm)?
         [ ] Applied  [ ] Not applicable (no RMSNorm)  [ ] Deferred: <reason>

3. §2.2  ≥5D tensor shapes found and resolved?
         [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

4. §1.6  Rotary / positional embedding precompute moved to model init?
         (Static buffers `sin_cached`/`cos_cached` (or equivalent) registered at __init__,
          indexed by position_ids in forward; no per-forward Range/Cos/Sin subgraph.)
         [ ] Applied  [ ] Not applicable (no position-dependent embeddings)  [ ] Deferred: <reason>

5. §3.7  Fused gate-up projection split into separate gate_proj / up_proj weights?
         [ ] Applied  [ ] Not applicable (no gated FFN, or already split)  [ ] Deferred: <reason>

6. §4.1  Broadcast mask + blocked query attention applied (if masked-attention model)?
         (REQUIRED whenever `cloud-ai-model-conversion/references/qaic-model-requirements.md` "Valid-Length Inputs for Padded Variable-Length Dimensions" applies; framework SDPA with `attn_mask` must be replaced
          — validated with one-block equivalence cosine > 0.9999 before compile.)
         [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

7. §1.1  SHA decomposition attempted and benchmarked?
         (REQUIRED for ALL transformer models regardless of head/core ratio or barrier% — do not mark Not applicable)
         [ ] Applied — throughput: <before> → <after> inf/s
         [ ] Deferred: <reason>

8. §3.3  Scalar Div fusion (attention scale ÷√d_k, GELU ÷√2) folded into weights?
         [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

9. §3.1  Static MatMul → 1×1 Conv2D applied?
         [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

10. §1.2+§1.3  Transpose pre-shaping and H/W flip applied?
          [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

11. §2.1  Sequence 1D → 2D reshape applied (QPC checksum verified)?
          [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

12. §3.2  Dynamic Batch MatMul decomposition applied?
          [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

13. §1.4+§1.5  Preprocessing moved offline / batch split offline?
          [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>

14. §3.4  Fused GELU custom kernel implemented?
          [ ] Applied  [ ] Not applicable  [ ] Deferred: <reason>
───────────────────────────────────────────────────────────────────────────────
```
