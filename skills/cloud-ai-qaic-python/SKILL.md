---
name: cloud-ai-qaic-python
description: Python qaic.Session API reference for Cloud AI 100. Use when user needs to run inference from Python, prepare inputs, measure latency/throughput from Python, or set up ProgramGroup for multi-model pipelines sharing a single device.
---

# Skill: Cloud AI 100 – `qaic.Session` Python API Guide

## Overview

This skill explains how to use the **Python QAic API** for running inference with `qaic.Session`:

- Loading QPC binaries and setting up sessions
- Preparing inputs (contiguous arrays, layout/dtype)
- Running single and batched inferences
- Measuring latency and throughput from Python

It assumes you already have compiled QPC binaries (via `qaic-compile`) and prepared input data.

---

## 1. Basic Session Usage

For explicit QPC loading, queues, ExecObj, buffer mappings, asynchronous
completion, or lower-level resource sharing, use the optional
references/qaicrt.md reference. The lower-level binding is qaicrt.

### 1.1 Creating a Session

**Decide first whether this session is standalone or a ProgramGroup member — the initialization differs and the two patterns must not be mixed.**

| Configuration | Initialization |
|---|---|
| No `oversubscription_name` (single QPC) | Construct the session, then call `session.setup()` once. |
| `oversubscription_name` set (multi-stage pipeline) | Construct **every** member with the same name and `dev_id` **before any `run()`**, and call `setup()` on **none** of them. The first `run()` activates the whole group. See Section 6. |

A standalone session:

```python
import qaic

# Standalone session only. Do not copy this setup() call into a helper that
# also builds ProgramGroup members — see Section 6.
session = qaic.Session(
    model_path="./model_qpc/programqpc.bin",
    num_activations=1,  # independent inference instances; 1 for benchmarking
    set_size=1,         # one ExecObj/request slot per activation
)

session.setup()  # initialize resources
```

Important parameters:

- `model_path`: path to the compiled QPC binary (`programqpc.bin`).
- `num_activations`: number of separate program instances loaded on device. Maximum simultaneous in-flight inferences = `set_size × num_activations`.
- `set_size`: ExecObjs per activation. Use `set_size=1` for standard model-onboarding benchmarks.
- `dev_id`: device QID to run on (default 0).
- `oversubscription_name`: name string that groups multiple sessions into a shared ProgramGroup (single-device only — see Section 6).

### 1.3 Inspecting Tensor Names and Shapes

After `session.setup()`, query the model's IO without running inference:

```python
print(session.model_input_names)        # e.g. ['input']
print(session.model_output_names)       # e.g. ['class_id']
print(session.model_input_shape_dict)   # e.g. {'input': ((1, 224, 224, 3), 'float32')}
print(session.model_output_shape_dict)  # e.g. {'class_id': ((1,), 'int64')}
```

Each shape-dict value is `(shape, dtype)` from the compiled QPC. Use these
instead of hardcoding tensor names, shapes, layouts, or dtypes. Output dtypes
may be integral, for example class IDs or indices, rather than floating-point
logits.

### 1.4 Running Inference

```python
import numpy as np

input_name = session.model_input_names[0]
input_shape, input_dtype = session.model_input_shape_dict[input_name]
input_tensor = np.fromfile("input_000.raw", dtype=np.dtype(input_dtype)).reshape(input_shape)
input_tensor = np.ascontiguousarray(input_tensor)  # CRITICAL: must be contiguous

inputs = {input_name: input_tensor}
outputs: dict[str, np.ndarray] = session.run(inputs)

for name, value in outputs.items():
    expected_shape, expected_dtype = session.model_output_shape_dict[name]
    print(name, value.shape, value.dtype)
    assert value.shape == expected_shape
    assert value.dtype == np.dtype(expected_dtype)
```

**CRITICAL:** Ensure all numpy inputs are **contiguous** (`np.ascontiguousarray`). Non-contiguous arrays can cause incorrect reads or performance degradation.

The input layout is whatever the QPC compiled to. Reshape from
`model_input_shape_dict`; do not select NCHW or NHWC from a familiar model
family or byte count alone.

---

## 2. Input Preparation

### 2.1 Layout and Dtype

Match the compiled QPC contract exactly:

- **Layout**: read the compiled shape from `model_input_shape_dict`; the model
  preprocessing/conversion contract determines its dimension semantics.
- **Dtype**: use the dtype paired with that compiled shape.

For image input, take the layout from the model's preprocessing/conversion
contract, then verify the prepared sample against the compiled non-batch shape:

```python
import cv2
import numpy as np


def preprocess_image(image_path, sample_shape, input_dtype, layout):
    if layout == "NCHW":
        channels, height, width = sample_shape
    elif layout == "NHWC":
        height, width, channels = sample_shape
    else:
        raise ValueError(f"unsupported image layout: {layout}")
    if channels != 3:
        raise ValueError(f"example expects three channels, got {channels}")

    img = cv2.imread(image_path)              # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    # Apply the model-specific normalization here.
    tensor = img.transpose(2, 0, 1) if layout == "NCHW" else img
    tensor = np.ascontiguousarray(tensor, dtype=np.dtype(input_dtype))
    if tensor.shape != tuple(sample_shape):
        raise ValueError(f"prepared {tensor.shape}; QPC requires {sample_shape}")
    return tensor
```

### 2.2 Multiple Inputs

If the model has multiple inputs (e.g., `input_ids`, `attention_mask`):

```python
inputs = {
    "input_ids": input_ids_array,      # [batch, seq_len]
    "attention_mask": mask_array,      # [batch, seq_len]
    # ...
}
outputs = session.run(inputs)
```

Ensure each input tensor's shape and dtype match the compiled model.

---

## 3. Running Batches & Concurrency

### 3.1 Batch Dimension

`qaic.Session` treats the batch dimension as part of the compiled input shape.
Prepare exactly that batch size and verify the completed array before `run()`:

```python
input_name = session.model_input_names[0]
input_shape, input_dtype = session.model_input_shape_dict[input_name]
batch_size = input_shape[0]
if len(image_paths) != batch_size:
    raise ValueError(f"QPC requires batch {batch_size}, got {len(image_paths)} images")

# `layout` comes from the model's preprocessing/conversion contract.
batch = np.stack([
    preprocess_image(path, input_shape[1:], input_dtype, layout)
    for path in image_paths
])
batch = np.ascontiguousarray(batch, dtype=np.dtype(input_dtype))
if batch.shape != tuple(input_shape):
    raise ValueError(f"prepared {batch.shape}; QPC requires {input_shape}")

outputs = session.run({input_name: batch})
```

### 3.2 In-Flight Activations (`num_activations` and `set_size`)

Two parameters control on-device concurrency:

- **`num_activations`** — number of separate program instances loaded on device. Each activation is an independent copy of the network weights and state.
- **`set_size`** — number of ExecObj (execution slots) per activation. Each ExecObj can hold one in-flight inference.

**Maximum simultaneous in-flight inferences = `set_size × num_activations`**

- Use `num_activations=1, set_size=1` for unloaded latency and integration checks. Increasing activations does not make one request finish faster.
- For a selected full-SoC throughput row, use its `instances` value for `num_activations`. Let `cloud-ai-performance-optimization` select that row; use `qaic-runner` for the authoritative throughput measurement.
- Use `set_size=1` unless a requested concurrency experiment defines another value, and record the value used.

---

## 4. Measuring Latency & Throughput

### 4.1 Simple Latency Measurement

```python
import time

# Assume session already set up, and single input batch in `inputs`

# Warmup
for _ in range(20):
    session.run(inputs)

# Timed run
n_iter = 100
latencies = []
t_start = time.perf_counter()
for _ in range(n_iter):
    t0 = time.perf_counter()
    session.run(inputs)
    latencies.append(time.perf_counter() - t0)

t_total = time.perf_counter() - t_start

import numpy as np
latencies_ms = np.array(latencies) * 1000.0
print("Throughput: {:.1f} IPS".format(n_iter / t_total))
print("Lat p50: {:.2f} ms  p99: {:.2f} ms".format(
    np.percentile(latencies_ms, 50), np.percentile(latencies_ms, 99)))
```

### 4.2 End-to-End Stage Measurements

In split-model setups (e.g., some stages on device, some on host), measure each stage separately and then overall end-to-end latency.

Example pattern:

```python
def benchmark_stage(session, inputs, n_warmup=20, n_iter=200):
    import time, numpy as np

    for _ in range(n_warmup):
        session.run(inputs)

    latencies = []
    start = time.perf_counter()
    for _ in range(n_iter):
        t0 = time.perf_counter()
        session.run(inputs)
        latencies.append(time.perf_counter() - t0)
    total = time.perf_counter() - start

    lat = np.array(latencies) * 1000
    return {
        "ips": n_iter / total,
        "p50_ms": float(np.percentile(lat, 50)),
        "p99_ms": float(np.percentile(lat, 99)),
    }
```

---

## 5. Error Handling & Pitfalls

Common issues when using `qaic.Session`:

| Symptom | Cause | Fix |
|---|---|---|
| Wrong or garbage outputs | Input layout or dtype differs from the compiled QPC, or array is non-contiguous | Use the compiled input shape and dtype from `model_input_shape_dict`, then call `np.ascontiguousarray` |
| Shape mismatch errors | Input shapes don't match compiled model | Reshape to `model_input_shape_dict[name][0]`; do not rely on byte size alone |
| Type mismatch errors | Dtype mismatch between inputs and model | Use `np.dtype(model_input_shape_dict[name][1])` |
| Poor throughput from Python | Python overhead, single-threaded execution | Use `qaic-runner` for throughput benchmarks; keep Python for integration tests |

---

## 6. ProgramGroup — Python API

For single-SoC pipelines where multiple models run sequentially, use ProgramGroup to share device resources. From Python, this is accessed via the `oversubscription_name` parameter on `qaic.Session`. Sessions with the same name on the same `dev_id` share a ProgramGroup:

```python
# Both sessions share a ProgramGroup on device 0 (single-device QPCs only)
encoder_sess = qaic.Session(
    model_path="encoder_qpc/programqpc.bin",
    num_activations=1,
    set_size=1,
    dev_id=0,
    oversubscription_name="pipeline_group"
)

decoder_sess = qaic.Session(
    model_path="decoder_qpc/programqpc.bin",
    num_activations=1,
    set_size=1,
    dev_id=0,
    oversubscription_name="pipeline_group"  # same name = same ProgramGroup
)
```

Run them sequentially — the shared ProgramGroup reserves device resources once:

```python
enc_out = encoder_sess.run({"input": input_data})
dec_out = decoder_sess.run({"encoder_output": enc_out["output"]})
```

**Constraints:**
- Single-device QPCs only (`mdp_ts_num_devices = 1`). MDP/TS QPCs cannot use ProgramGroup.
- For multi-SoC pipelines, device requirements are additive — no sharing mechanism exists.

**Do not call `session.setup()` on ProgramGroup members.** Construct all sessions sharing an `oversubscription_name`, then call `.run()` — the first `.run()` sets up the group lazily. Eager `setup()` on multiple members can fail with `Failed to allocate NSP resources` / `Activation Reservation failed` / `Couldn't get completed Inference` even when each program fits (group resources are max-sized, not summed — `cloud-ai-qaic-cpp` §4.7). Fix setup order before assuming the pairing is unsupported or recompiling stages at lower core counts.

The group's first `.run()` pays the whole group's load cost, not just that program's — exclude it from timed results or issue a throwaway warmup call.

For the full ProgramGroup concept, C++/C API, lifecycle, resource table, and MDP constraints, see the `cloud-ai-qaic-cpp` skill.

---

## 7. Workflow Summary

1. Compile your model to QPC with `qaic-compile`.
2. Read `model_input_shape_dict`, then prepare tensors with the compiled layout,
   dtype, shape, and **contiguous** memory.
3. Decide whether the sessions are standalone or ProgramGroup members (§1.1). For multi-model pipelines sharing the same SoC, use **ProgramGroup** via `oversubscription_name` (Section 6) instead of independent sessions.
4. Create `qaic.Session` with `model_path` and appropriate `num_activations`. Standalone: call `session.setup()` once. ProgramGroup: construct every member first and call `setup()` on none of them.
5. Run inference via `session.run(inputs)`. A ProgramGroup's first `run()` activates the group and pays its whole load cost — exclude it from timing or issue a throwaway warmup call.
6. Optionally measure latency/throughput from Python for integration testing.

For large-scale performance evaluation, use `qaic-runner` (see `cloud-ai-qaic-runner` skill) and the `cloud-ai-performance-optimization` skill.

---

## Related Skills

- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Go here first to produce the QPC binary that `qaic.Session` loads.
- **Cloud AI 100 – Performance Optimization & Runtime** (`cloud-ai-performance-optimization` skill) — Go here for the benchmark pattern and guidance on interpreting throughput results; use `qaic-runner` for heavy-duty benchmarking.
- **Cloud AI 100 – `qaic-compile` Detailed Tool Guide** (`cloud-ai-qaic-compile` skill) — Go here to recompile with different flags when session results indicate a compile-time issue.
- **Cloud AI 100 – `qaic-runner` Detailed Tool Guide** (`cloud-ai-qaic-runner` skill) — Prefer `qaic-runner` over `qaic.Session` for throughput benchmarking; use `qaic.Session` for Python integration and output comparison.
- **Cloud AI 100 – QAIC C++/C Runtime API** (`cloud-ai-qaic-cpp` skill) — Full ProgramGroup reference: C++/C API, lifecycle, resource table, oversubscription, MDP constraints.
- **Cloud AI 100 – SoC Architecture** (`cloud-ai-architecture` skill) — MDP/TS device counts and Ultra card topology; relevant when sizing ProgramGroup device requirements.
