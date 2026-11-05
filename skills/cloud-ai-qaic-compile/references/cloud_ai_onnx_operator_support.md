# Skill: Cloud AI 100 – ONNX Operator Support

## Overview

This is the authoritative reference for determining whether an ONNX model will compile for Cloud AI 100 without custom ops. It covers:

- Full supported operator table (SDK 1.21, opset 7+)
- Constant tensor requirements per operator
- Unsupported dynamic patterns and their fixes
- ONNX symbol mapping for dynamic shapes

*Source: Qualcomm Cloud AI 100 Compiler Library API Reference, Rev. K (Release 1.9.1, April 2023)*

---

## 1. Supported ONNX Operators (SDK 1.21, Opset 7+)

116 operators are supported as of SDK 1.21. **New in SDK 1.21 (6 additions):** `GroupNormalization`, `MatMulNBits`, `QLinearMatMul`, `SequenceAt`, `SequenceConstruct`, `Trilu`. `LayerNormalization` is also supported natively (was absent from earlier table versions — see row 117).

| # | Operator | Data Types | Key Limitations |
|---|---|---|---|
| 1 | Abs | Float32, Float16 | — |
| 2 | Acos | Float32, Float16 | — |
| 3 | Add | Float32, Float16, Int32, Int64, Int8 | — |
| 4 | And | Bool | — |
| 5 | ArgMax | Float32, Float16, Int64, Int32, Int8 | — |
| 6 | ArgMin | Float32, Float16, Int8 | — |
| 7 | Asin | Float32, Float16 | — |
| 8 | Atan | Float32, Float16 | — |
| 9 | AveragePool | Float32, Float16, Int8 | Int8: 2D only. Float: 1D/2D only. 1D requires kernel dim==1. `ceil_mode` not supported |
| 10 | BatchNormalization | Float32, Float16 | Only Y output valid during inference |
| 11 | Cast | Float32, Float16 | Non-quantized types only |
| 12 | Ceil | Float32, Float16 | — |
| 13 | Clip | Float32, Float16 | — |
| 14 | Concat | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 15 | Constant | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 16 | ConstantOfShape | Shape input: Int64 | — |
| 17 | Conv | Float32, Float16 | 1D/2D/3D only. No dilation for 3D. Dilations must be specified per axis. For 2D Conv performance, HMX requires equal x/y dilation <=2; if dilation >1, stride must be 1. Other dilation geometry falls back to HVX. |
| 18 | ConvTranspose | Float32, Float16 | — |
| 19 | Cos | Float32, Float16 | — |
| 20 | CumSum | Float32, Float16, Int64, Int32 | `axis` must be constant. Axis 0 (all modes); axis 1/2 for 2D/3D inputs without exclusive/reverse |
| 21 | DequantizeLinear | Input: Int8 | — |
| 22 | DepthToSpace | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 23 | Div | Float32, Float16, Int64, Int32, Int8 | — |
| 24 | Dropout | Float32, Float16, Int64, Int32, Int8, Bool | No-op during inference |
| 25 | Einsum | Float32, Float16, Int8 | Max 2 operands. No broadcast by ellipsis. No diagonal |
| 26 | Elu | Float32, Float16, Int8 | — |
| 27 | Equal | Float32, Float16, Int64, Int32, Int8 | — |
| 28 | Erf | Float32, Float16 | — |
| 29 | Exp | Float32, Float16 | — |
| 30 | Expand | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 31 | Flatten | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 32 | Floor | Float32, Float16 | — |
| 33 | GRU | Float32, Float16 | No activation clipping. No variable sequence length. No `clip` attr. No `sequence_lens` |
| 34 | Gather | data: Float32/Float16/Int64/Int32/Int8; indices: Int64/Int32 | — |
| 35 | GatherElements | data: Float32/Float16/Int64/Int32/Int8/Bool; indices: Int64/Int32 | — |
| 36 | GatherND | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 37 | Gemm | Float32, Float16, Int8 | — |
| 38 | GlobalAveragePool | Float32, Float16, Int8 | — |
| 39 | GlobalMaxPool | Float32, Float16, Int8 | 1D/2D only |
| 40 | Greater | Float32, Float16, Int64, Int32, Int8 | — |
| 41 | GreaterOrEqual | Float32, Float16, Int64, Int32, Int8 | — |
| 42 | GroupNormalization | Float32, Float16 | New in SDK 1.21 |
| 43 | HardSigmoid | Float32, Float16, Int8 | — |
| 44 | Identity | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 45 | If | cond: Int8; branches: Graph | Condition must be constant |
| 46 | InstanceNormalization | Float32, Float16 | — |
| 47 | IsInf | Float32, Float16 | — |
| 48 | IsNaN | Float32, Float16 | — |
| 49 | LeakyRelu | Float32, Float16 | — |
| 50 | Less | Float32, Float16, Int64, Int32, Int8 | — |
| 51 | LessOrEqual | Float32, Float16, Int8 | — |
| 52 | Log | Float32, Float16 | — |
| 53 | LogSoftmax | Float32, Float16 | Last dimension only |
| 54 | Loop | M: Int64; cond: Int8/Bool | Static loops only — max trip-count M and cond must be constant or foldable at load time |
| 55 | LpNormalization | Float32, Float16 | — |
| 56 | LRN | Float32, Float16 | — |
| 57 | LSTM | Float32, Float16 | No activation clipping. No variable sequence length. No `clip`. No `sequence_lens` |
| 58 | MatMul | Float32, Float16, Int8 | — |
| 59 | MatMulInteger | Input: Int8, Output: Int8 | Must be preceded by QuantizeLinear, followed by DequantizeLinear |
| 60 | MatMulNBits | Float32, Float16; weights: Int4/Int8 | New in SDK 1.21. Weight-only quantization for LLM inference (GGUF/GGML style) |
| 61 | Max | Float32, Float16, Int64, Int32, Int8 | — |
| 62 | MaxPool | Float32, Float16, Int8 | Int8: 2D only. Float: 1D/2D only. 1D requires kernel dim==1. `ceil_mode` not supported |
| 63 | Mean | Float32, Float16 | — |
| 64 | Min | Float32, Float16, Int64, Int32, Int8 | — |
| 65 | Mod | Float32, Float16, Int64, Int32, Int8 | `fmod` not supported for Int8 |
| 66 | Mul | Float32, Float16, Int8 | — |
| 67 | Neg | Float32, Float16 | — |
| 68 | NonMaxSuppression | boxes/scores/thresholds: Float32/Float16; max_boxes: Int64/Int32 | `max_output_boxes_per_class` ≤ uint16_t max. `iou_threshold` and `score_threshold` must be constants |
| 69 | NonZero | Float32, Float16, Int64, Int32, Int8, Bool | Input must be constant |
| 70 | Not | Bool | — |
| 71 | OneHot | indices/values: Float32/Float16/Int64/Int32; depth: Int32 | `axis` must be -1. `on_value=1`, `off_value=0`. `depth` must be constant |
| 72 | Or | Bool | — |
| 73 | PRelu | Float32, Float16, Int8, Uint8 | — |
| 74 | Pad | Float32, Float16, Int64, Int32, Int8, Bool | Opset > 10: `pads` and `values` must be constant inputs. Only constant mode. No negative pads |
| 75 | Pow | Float32, Float16 | — |
| 76 | QLinearMatMul | Input/output: Int8/Uint8 | New in SDK 1.21. Pre-quantized MatMul with inline scale/zero-point tensors |
| 77 | QuantizeLinear | Float32, Int32 | Scalar `y_scale` only. Scalar `y_zero_point` only |
| 78 | RNN | Float32, Float16 | No activation clipping. No variable sequence length. No `clip`. No `sequence_lens` |
| 79 | Range | Float32, Float16, Int64, Int32 | — |
| 80 | ReduceL2 | Float32, Float16 | `axes` must be unique |
| 81 | ReduceMax | Float32, Float16, Int64, Int32 | — |
| 82 | ReduceMean | Float32, Float16, Int8 | — |
| 83 | ReduceMin | Float32, Float16, Int64, Int32 | — |
| 84 | ReduceProd | Float32, Float16 | — |
| 85 | ReduceSum | Float32, Float16, Int8 | — |
| 86 | Relu | Float32, Float16, Int8 | — |
| 87 | Reshape | Float32, Float16, Int64, Int32, Int8, Bool | `shape` input must be constant |
| 88 | Resize | Float32, Float16, Int64, Int32, Int8 | `scales` must be constant. `align_corners`/`pytorch_half_pixel` for bilinear only. No `exclude_outside`. No `extrapolation_value`. Nearest: `nearest_mode` must be `floor`. Linear and nearest only (no cubic) |
| 89 | RoiAlign | Float32, Float16 | — |
| 90 | Round | Float32, Float16 | — |
| 91 | ScatterElements | data: Float32/Float16/Int8; indices: Int64 | All inputs must be 2D |
| 92 | ScatterND | Float32, Float16, Int32, Int8, Bool; indices: Int32/Int64 | Indices must be 2D or 3D |
| 93 | SequenceAt | Sequence of Float32/Float16/Int64/Int32/Int8 | New in SDK 1.21. Extracts a tensor from a sequence at a given index |
| 94 | SequenceConstruct | Float32, Float16, Int64, Int32, Int8 | New in SDK 1.21. Constructs a sequence from a list of tensors |
| 95 | Shape | Int64 | — |
| 96 | Sigmoid | Float32, Float16 | — |
| 97 | Sign | Float32, Float16, Int8 | — |
| 98 | Sin | Float32, Float16 | — |
| 99 | Slice | Float32, Float16, Int64, Int32 | `start`, `end`, `axes`, `steps` must be constant. Multi-axis step size must be 1 |
| 100 | Softmax | Float32, Float16 | — |
| 101 | Softplus | Float32, Float16 | — |
| 102 | SpaceToDepth | Float32, Float16, Int64, Int32, Int8 | — |
| 103 | Split | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 104 | Sqrt | Float32, Float16 | — |
| 105 | Squeeze | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 106 | Sub | Float32, Float16, Int8 | — |
| 107 | Sum | Float32, Float16, Int32, Int64, Int8 | — |
| 108 | Tanh | Float32, Float16 | — |
| 109 | Tile | Float32, Float16, Int64, Int32, Int8, Bool | `repeats` must be constant |
| 110 | TopK | Float32, Float16, Int8 | `k` must be constant. Axis must be last dimension |
| 111 | Transpose | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 112 | Trilu | Float32, Float16, Int64, Int32, Int8, Bool | New in SDK 1.21. Returns upper or lower triangular part of a matrix |
| 113 | Unsqueeze | Float32, Float16, Int64, Int32, Int8, Bool | — |
| 114 | Upsample | Float32, Float16, Int64, Int32, Int8, Bool | Opset 7/8/9 only — use Resize for opset 10+. Nearest and bilinear modes only |
| 115 | Where | Float32, Float16, Int8, Uint8, Bool | — |
| 116 | Xor | Bool | — |
| 117 | LayerNormalization | Float32, Float16 | Not listed in `OnnxSupportedOperators.txt` (`-operators-supported=onnx`) despite compiling natively — this is a known omission specific to this op |

> **Detectron2 models**: Use `torch.onnx.export` via `torch.jit.trace` to produce a standard ONNX model. **Avoid the Caffe2 export path** — it requires an old Python/Torch environment, and Caffe2 op support in the QAic compiler may be deprecated. The Caffe2 operator list (`GenerateProposals`, `RoIAlign`, etc.) is retained here for reference only.

---

## 2. Constant Tensor Requirements

The following operator inputs **must be constant tensors** (not runtime-computed values). This is the most common cause of compilation failure.

| Operator | Must-Be-Constant Inputs |
|---|---|
| Reshape | `shape` |
| Resize | `scales` |
| Tile | `repeats` |
| Slice | `start`, `end`, `axes`, `steps` |
| Pad (opset > 10) | `pads`, `constant_value` |
| NonZero | entire input tensor |
| TopK | `k` |
| OneHot | `depth` |
| ConstantOfShape | `shape` |
| Loop | max trip-count `M` and condition `cond` |
| If | `condition` |
| NonMaxSuppression | `iou_threshold`, `score_threshold` |

**Fix pattern**: Replace dynamic computations feeding these inputs with `onnx.helper.make_node("Constant", ...)` nodes that bake in the value for your fixed input shape. See `cloud-ai-model-conversion/references/qaic-model-requirements.md`, "Constant Inputs for Shape-Affecting Ops", for the requirement and use the `bake_constant_tensor` helper pattern in `cloud-ai-model-conversion/references/onnx-graph-surgery.md`.

---

## 3. Unsupported Dynamic Patterns and Fixes

| Pattern | Why It Fails | Fix |
|---|---|---|
| NonZero on runtime tensor | Input must be constant | Rewire downstream ops to use TopK indices; eliminate boolean masking |
| Dynamic Pad `pads` | Must be constant for opset > 10 | Evaluate pads for fixed input shape; bake as constant |
| Dynamic Reshape `shape` | Must be constant | Compute shape at export time; bake as constant |
| Loop with dynamic trip count | Only static loops supported | Unroll at export time or use fixed trip count |
| Resize with dynamic `scales` | Must be constant | Compute scales at export time; bake as constant |
| Slice with dynamic indices | All slice params must be constant | Bake `start`/`end`/`axes`/`steps` as constants |
| Upsample opset ≥ 10 | Use Resize instead | Replace Upsample nodes with Resize |
| Conv 3D with dilation | Not supported for 3D Conv | Use 2D convolutions or remove dilation |
| Conv 2D with unequal x/y dilation, dilation >2, or dilation >1 with stride >1 | Supported via HVX fallback, not HMX | Prefer equal x/y dilation <=2 with stride 1 on performance-critical Conv, or redesign/retrain if high-dilation Conv dominates opstats |
| AveragePool/MaxPool `ceil_mode=True` | Not supported | Set `ceil_mode=False` or adjust padding manually |
| CumSum on axis > 2 | Only axis 0/1/2 supported | Restructure computation |
| Einsum with > 2 operands | Max 2 operands | Decompose into multiple 2-operand Einsum nodes |
| Einsum with ellipsis broadcast | Not supported | Expand ellipsis explicitly |
| ScatterElements with non-2D inputs | All inputs must be 2D | Reshape to 2D before scatter |
| QuantizeLinear with per-channel scale | Only scalar `y_scale` supported | Use per-tensor quantization |
| LogSoftmax on non-last axis | Last dimension only | Transpose → LogSoftmax → Transpose |
| TopK on non-last axis | Axis must be last dimension | Transpose → TopK → Transpose |

---

## 4. ONNX Symbol Mapping for Dynamic Shapes

If your ONNX model uses symbolic dimensions (e.g., `batch_size`, `seq_len`), map them to concrete values at compile time using `-onnx-define-symbol`:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -onnx-define-symbol=batch_size,1 \
  -onnx-define-symbol=seq_len,128 \
  ...
```

---

## Related Skills

- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Go to `cloud-ai-model-conversion/references/onnx-graph-surgery.md` for graph surgery patterns to fix the unsupported dynamic patterns listed in this guide.
- **Cloud AI 100 – Hexagon / Custom Ops Compute** (`cloud-ai-custom-ops-kernels` skill) — Go here when an op is not in the supported list and no standard ONNX replacement exists.
- **Cloud AI 100 – `qaic-compile` Detailed Tool Guide** (`cloud_ai_qaic_compile_tool_guide.md`) — Go here for `-onnx-define-symbol` and other compile-time flags for handling shape and operator issues.
