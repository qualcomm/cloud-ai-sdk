# Custom Op Data Types and ONNX Generation

## Contents

- Tensor representation choices
- Float32 path
- INT8 index-type path
- ONNX graph generation
- Data-type best practices

## Data types

Custom op tensor types are determined at ONNX model creation time and flow
through to the kernel via `ctx->inputs[i].data`. Two paths are supported:

| Path | ONNX type | `elem-size` | Kernel cast | Host copy |
|---|---|---|---|---|
| **float32** (default) | `TensorProto.FLOAT` | 4 | `(float *)` | uint8↔float32 loops |
| **INT8 index type** | `TensorProto.INT8` | 1 | `(uint8_t *)` | `memcpy` |

### float32 path (default)

Use this when the kernel needs floating-point arithmetic — normalisation,
weighted blends, neural network activations, etc.

**ONNX model:**

```python
input_t  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [H, W])
output_t = helper.make_tensor_value_info('output', TensorProto.FLOAT, [H, W])
```

**`aic_batch_io.json`:** `"elem-size": 4`

**`customOpInferShape`:**

```cpp
output.dtype = DTFloat32;
```

**Kernel:**

```cpp
const float *src = (const float *)ctx->inputs[0].data;
float       *dst = (float *)ctx->outputs[0].data;
```

**Host-side conversion** (if input data is uint8):

```cpp
// Before inference — uint8 → float32 into DMA buffer
for (int i = 0; i < n; i++)
    float_buf[i] = (float)uint8_buf[i];

// After inference — float32 → uint8 from DMA buffer
for (int i = 0; i < n; i++) {
    float v = std::min(std::max(float_buf[i], 0.0f), 255.0f);
    uint8_buf[i] = (uint8_t)v;
}
```

These conversion loops run on the host CPU, outside the NSP kernel. Use the
host compiler's normal optimization settings, add `__restrict__` only when the
loop pointers cannot alias, and measure conversion cost for the target tensor
shape. This is separate from device-kernel vectorization: `*_aic.cpp` uses the
NSP's HVX path described in [HVX vectorization](hvx-vectorization.md).

Float32 uses four bytes per element; transfer size scales with the tensor shape.

### INT8 index type path

The SDK RST doc states that `int8_t`, `int16_t`, `int32_t`, and `int64_t` can
represent **index types**. Tensors declared as index types are **not quantized**,
regardless of compilation options — the compiler accepts them without a
quantization profile and without the "Type not supported yet" error.

Use this when the kernel operates on raw byte data and floating-point arithmetic
is not required.

**ONNX model:**

```python
input_t  = helper.make_tensor_value_info('input',  TensorProto.INT8, [H, W])
output_t = helper.make_tensor_value_info('output', TensorProto.INT8, [H, W])
```

**`aic_batch_io.json`:** `"elem-size": 1`

**`customOpVerify` and `customOpInferShape`:**

```cpp
// customOpVerify — accept int8 index type
if (opProp->inputs[0].dtype != DTQInt8)
    return false;

// customOpInferShape — propagate dtype to output
output.dtype = opProp->inputs[0].dtype;
```

**Kernel** (cast to `uint8_t*` — the bytes are raw unsigned values, not signed):

```cpp
const uint8_t *src = (const uint8_t *)ctx->inputs[0].data;
uint8_t       *dst = (uint8_t *)ctx->outputs[0].data;
```

**Host-side copy** (no conversion needed):

```cpp
// Before inference — direct copy into DMA buffer
std::memcpy(dma_in_buf, src_uint8, element_count);

// After inference — direct copy from DMA buffer
std::memcpy(dst_uint8, dma_out_buf, element_count);
```

INT8 uses one byte per element, making transfers 4× smaller than float32 for
the same tensor shape and reducing PCIe DMA time proportionally.

For raw byte data, this INT8 index-type path eliminates host-side conversion
loops and preprocessing overhead, reduces the tensor size 4×, enables HVX to
process 128 uint8 elements per vector instead of 32 float32 elements, and
reduces PCIe DMA time proportionally. Use float32 only when the kernel needs
floating-point arithmetic such as normalisation, weighted blends, or
neural-network activations; integer elementwise operations such as subtract,
clamp, and filter are faster with the INT8 index type.

---

## ONNX model generation

Create the ONNX model programmatically with `onnx` + `numpy`. Do not use PyTorch
export — it has limitations with non-standard domains and integer types.

**float32 tensors** (default — use when the kernel needs floating-point arithmetic):

```python
import onnx
from onnx import helper, TensorProto

input_t  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [H, W])
output_t = helper.make_tensor_value_info('output', TensorProto.FLOAT, [H, W])
status_t = helper.make_tensor_value_info('status', TensorProto.FLOAT, [1])
```

**INT8 index-type tensors** (use when the kernel operates on raw byte data):

```python
input_t  = helper.make_tensor_value_info('input',  TensorProto.INT8,  [H, W])
output_t = helper.make_tensor_value_info('output', TensorProto.INT8,  [H, W])
status_t = helper.make_tensor_value_info('status', TensorProto.FLOAT, [1])
# status remains float32 — the compiler does not support int32 as an output dtype
```

**Node and graph construction** (same for both paths):

```python
node = helper.make_node(
    'MyOp',
    inputs=['input'],
    outputs=['output', 'status'],
    domain='QAic',
    width=W,          # node attributes become ctx->params[] in the kernel
    height=H,
    offset=0,
)

graph = helper.make_model(
    helper.make_graph([node], 'my_graph', [input_t], [output_t, status_t]),
    opset_imports=[
        helper.make_opsetid('', 11),
        helper.make_opsetid('QAic', 1),
    ],
)
graph.ir_version = 6
onnx.checker.check_model(graph)
onnx.save(graph, 'custom.onnx')
```

Node attributes (e.g., `width`, `height`) are baked into the QPC at compile time
and appear as `ctx->params[]` in the kernel. They are **not** runtime buffers.
