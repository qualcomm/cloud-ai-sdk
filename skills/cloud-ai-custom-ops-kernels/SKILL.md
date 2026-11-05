---
name: cloud-ai-custom-ops-kernels
description: Use when qaic-compile requires a registered QAic custom op, or when implementing, optimizing, or debugging C++/HVX kernels embedded in an ONNX graph for Cloud AI 100.
---

# Cloud AI 100 Custom Ops Kernels

## Overview

A custom op is a QAic-domain ONNX node whose C++ implementation runs on the Cloud AI 100 NSP. It is compiled into a QPC and executed through the QAIC runtime.

Use a custom op when supported ONNX operators or graph transformations cannot
express the computation efficiently, when fusing operations reduces
intermediate memory traffic or synchronization, or when a specialized
composite block needs kernel-level control.

## When to Use

Use this skill when the model requires:

- An operation the QAIC compiler does not support after checking standard ONNX replacements.
- Device-side preprocessing, postprocessing, or a domain-specific transform.
- Fusing multiple operations to reduce intermediate tensors, memory traffic, or synchronization overhead.
- Implementing specialized attention or another composite block that standard ONNX composition cannot express efficiently.
- Direct control over NSP threads, HVX vectorization, or memory placement.
- An in-graph kernel registered under the `QAic` ONNX domain.

## When Not to Use

- **Supported ONNX replacement:** Check the operator-support material in `cloud-ai-qaic-compile` first.
- **Standalone bare-metal compute app:** Use `cloud-ai-compute-sdk`.
- **JIT eager-mode PyTorch kernel:** Use `cloud-ai-torch-qaic`.
- **Host-side QAIC runtime application:** Use `cloud-ai-qaic-cpp`.
- **General model export or QPC conversion:** Use `cloud-ai-model-conversion`.

## Non-Negotiable Rules

- Keep the ONNX node, YAML registration, kernel casts, and host buffers on the same dtype contract.
- Test the CPU interpreter/reference implementation before running the kernel on hardware.
- The runtime provides four custom-op threads. Do not create threads or use OS services on the NSP.
- Account for the approximate 8 MB per-core VTCM budget across the concurrent
  working set: resident tensors, active tiles, scratch, metadata, and buffers.
- Use unaligned HVX access for row addresses or strides that are not 128-byte aligned, and handle vector tails without out-of-bounds stores.
- Keep profiling QPCs separate from production QPCs; `-stats-level=70` adds instrumentation and runtime overhead.
- On the non-zero-copy QAicApi path, call `setData` before every synchronous `run`. Do not call `waitForCompletion` after synchronous `run`.

## Package Layout

```
CustomOp_Package/
├── custom_op_config.yaml
└── MyOp/
    └── src/
        ├── myop_functions.cpp
        ├── myop_interpreter.cpp
        └── myop_aic.cpp
```

- `custom_op_config.yaml`: registers the op, tensor contracts, implementations, and memory placement.
- `*_functions.cpp`: shape inference, verification, and implementation selection.
- `*_interpreter.cpp`: CPU reference implementation.
- `*_aic.cpp`: NSP kernel, including thread partitioning and HVX code.

## Route to the Right Reference

| Need | Read |
|---|---|
| Register the op, define YAML, or choose VTCM/DDR placement | [Configuration and memory](references/configuration-and-memory.md) |
| Choose float32 versus INT8 index types or generate the QAic-domain ONNX node | [Data types and ONNX](references/data-types-and-onnx.md) |
| Implement the C++ kernel, access `CustomOpContext`, use threading, synchronize, or log | [Kernel authoring](references/kernel-authoring.md) |
| Select HVX intrinsics, handle alignment/tails, or optimize vector code | [HVX vectorization](references/hvx-vectorization.md) |
| Build, compile, validate correctness, benchmark, or collect opstats | [Build, test, and profile](references/build-test-profile.md) |
| Integrate QAicApi, debug stale buffers, use zero-copy, or deploy multiple slots | [Runtime integration](references/runtime-integration.md) |

For hardware concepts, use `cloud-ai-architecture` and its Hexagon v68 DSP and programming references. For the operator-support decision, use `cloud-ai-qaic-compile` before implementing a custom op.

## End-to-End Workflow

1. Confirm that a supported ONNX replacement or graph transform is insufficient.
2. Choose the tensor representation and define the ONNX node, YAML registration, and `aic_batch_io.json` contract.
3. Implement the interpreter/reference path and validate shape, dtype, rank, and dimension bounds.
4. Implement the NSP kernel with the provided four-thread model, correct synchronization, and safe HVX alignment/tail handling.
5. Build the host library, generate the ONNX model and reference outputs, compile the QPC, and compare interpreter and hardware outputs.
6. Profile representative content, distinguish host/DMA time from NSP time, and use opstats to guide placement and vectorization.
7. Integrate the QPC into the QAicApi runtime, validate buffer mapping and repeated-invocation behavior, then keep the production QPC separate from profiling builds.

## Completion Criteria

A custom op is ready to integrate when:

- The interpreter/reference output passes the defined numerical tolerance.
- Hardware output matches the reference on representative inputs.
- Buffer names and shapes are verified after QPC compilation.
- Profiling identifies the dominant cost and uses a separate stats QPC.
- Runtime tests cover repeated synchronous invocations, zero-copy behavior when used, and the intended slot/core configuration.

## Common Mistakes

- **Writing a custom op too early:** Check supported ONNX replacements first.
- **Wrong byte semantics:** INT8 index-type tensors are raw bytes; cast device pointers to `uint8_t *`, not signed arithmetic values. See [Data types and ONNX](references/data-types-and-onnx.md).
- **Silent wrong output:** Unaligned loads/stores and unmasked vector tails can corrupt rows. See [HVX vectorization](references/hvx-vectorization.md).
- **Hardware-only debugging:** Run the interpreter and numerical comparison before hardware. See [Build, test, and profile](references/build-test-profile.md).
- **Stale outputs:** The non-zero-copy path needs `setData` before every synchronous `run`; synchronous `run` must not be followed by `waitForCompletion`. See [Runtime integration](references/runtime-integration.md).
- **Shipping profiling overhead:** Keep the `-stats-level=70` QPC in a separate output directory.

## Related Skills

- `cloud-ai-architecture`: Hardware memory hierarchy, Hexagon v68 DSP behavior, and HVX programming references.
- `cloud-ai-model-conversion`: ONNX export, graph cleanup, and QPC compilation workflow.
- `cloud-ai-performance-optimization`: End-to-end throughput, latency, and opstats analysis.
- `cloud-ai-qaic-cpp`: Host-side QAIC C++/C runtime API.
