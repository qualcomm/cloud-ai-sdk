---
name: cloud-ai-qaic-cpp
description: QAIC C++/C runtime API reference for Cloud AI 100. Use when user needs to write C++ or C inference applications, implement ProgramGroup for multi-model pipelines, understand headers/libraries/build setup, or work with the low-level QAIC runtime API.
---

# Skill: Cloud AI 100 – QAIC C++/C Runtime API Guide

## Overview

This skill covers the **C++ and C runtime API** for Cloud AI 100:

- **Headers and libraries** — what to include and link
- **Build setup** — CMake configuration
- **Minimal working example** — load QPC → run inference → read output
- **ProgramGroup** — sharing a single SoC across multiple models in a pipeline
- C++ and C API usage patterns
- ProgramGroup lifecycle, resource reservation, and oversubscription
- MDP/TS constraints

For Python-based inference with `qaic.Session`, see the `cloud-ai-qaic-python` skill.

---

## 1. Headers and Libraries

### 1.1 Header Files

All headers are installed under `/opt/qti-aic/dev/inc/`.

| Header | Purpose |
|---|---|
| `QAicApi.hpp` | **C++ umbrella header** — include this for all C++ runtime work. Pulls in all sub-headers below. |
| `QAicApi.h` | C-99 runtime API — function table, buffer types, context/program/execobj lifecycle. Include this for C applications. |
| `QAicRuntimeTypes.h` | Core type definitions: `QID`, `QStatus`, `QBuffer`, `QAicBufferIoTypeEnum_t`, `QAicBufferDataTypeEnum_t`, etc. |
| `QAicTypes.h` | Additional type aliases and constants. |
| `QAicApiOsDef.hpp` | OS-specific definitions (included automatically by `QAicApi.hpp`). |
| `QAicApiJit.h` | JIT compilation API (advanced; not needed for standard QPC-based inference). |
| `QAicGraphApi.h` | Graph-level API for dynamic graph manipulation. |
| `QAicMLApi.h` | ML-layer API (higher-level model execution helpers). |
| `QAicFilesystem.h` | Filesystem utilities for QPC loading. |
| `QAicAppCommon/QAicAppJsonIOParserLib.h` | JSON I/O parser for reading `bindings.json` input/output specs. |
| `Common/QAicCommonTypes.h` | Shared common types. |
| `CustomOpAICInterface.h` | Custom op registration interface (see `cloud-ai-custom-ops-kernels` skill). |

**For standard inference in C++, you only need:**
```cpp
#include "QAicApi.hpp"
```

**For C applications:**
```c
#include "QAicApi.h"
```

### 1.2 C++ Namespace Structure

`QAicApi.hpp` exposes the `qaic::rt` namespace with RAII shared-pointer wrappers:

| Type alias | Underlying class | Purpose |
|---|---|---|
| `qaic::rt::shContext` | `shared_ptr<Context>` | Device context (owns device connection) |
| `qaic::rt::shQpc` | `shared_ptr<Qpc>` | Loaded QPC binary |
| `qaic::rt::shProgram` | `shared_ptr<Program>` | Compiled program on a device |
| `qaic::rt::shExecObj` | `shared_ptr<ExecObj>` | Execution object (holds I/O buffers) |
| `qaic::rt::shQueue` | `shared_ptr<Queue>` | Async submission queue |
| `qaic::rt::shProgramGroup` | `shared_ptr<ProgramGroup>` | Multi-model resource group |
| `qaic::rt::shInferenceVector` | `shared_ptr<InferenceVector>` | Buffer set for one inference |
| `qaic::rt::BufferMappings` | `vector<QAicBufferMappings_t>` | I/O buffer metadata from QPC |

Exceptions are in the `qaic` namespace (not `qaic::rt`):
- `qaic::ExceptionInit` — initialization failures
- `qaic::ExceptionRuntime` — runtime execution failures

### 1.3 Libraries

The runtime shared library is at `/opt/qti-aic/dev/lib/x86_64/libQAic.so`.

| Library | Purpose |
|---|---|
| `libQAic.so` | **Main runtime library** — link all C++ and C applications against this. |
| `libQAicCompiler.so` | Compiler library (not needed for inference-only applications). |
| `libQAicML.so` | ML-layer API library (only if using `QAicMLApi.h`). |

The C API uses a **function table** loaded via `dlopen`/`dlsym` at runtime (see `QAicInferenceExampleC99.c`). The C++ API wraps this transparently — just link `-lQAic`.

---

## 2. Build Setup

### 2.1 CMake

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_qaic_app)
set(CMAKE_CXX_STANDARD 20)

find_package(Threads REQUIRED)

# QAIC SDK headers
include_directories("/opt/qti-aic/dev/inc")

# QAIC runtime library
find_library(QAIC_LIB QAic PATHS "/opt/qti-aic/dev/lib/x86_64" REQUIRED)

add_executable(my_app main.cpp)

# QAIC umbrella headers may expose deprecated declarations from optional
# benchmarking APIs. Suppress warnings emitted by those SDK headers.
target_compile_options(my_app PRIVATE -Wno-deprecated-declarations)

# -Wl,--no-as-needed is required: libQAic.so uses pthreads internally
# and the linker may otherwise drop the pthread dependency
target_link_libraries(my_app
    PRIVATE
        ${QAIC_LIB}
        Threads::Threads
        dl
    LINK_FLAGS "-Wl,--no-as-needed"
)
```

Build:
```bash
mkdir build && cd build
cmake ..
make
```

### 2.2 Manual Compilation

```bash
g++ -std=c++20 \
    -I/opt/qti-aic/dev/inc \
    -L/opt/qti-aic/dev/lib/x86_64 \
    -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64 \
    -Wl,--no-as-needed \
    -Wno-deprecated-declarations \
    main.cpp -o my_app \
    -lQAic -lpthread -ldl
```

`QAicApi.hpp` includes optional SDK declarations that may be marked deprecated,
so builds can emit warnings from SDK headers even when the application uses the
current runtime API. Keep `-Wno-deprecated-declarations` in the application
compile options unless you are intentionally auditing SDK header deprecations.

### 2.3 SDK Examples

The SDK ships ready-to-build examples at `/opt/qti-aic/examples/`:

| Path | Description |
|---|---|
| `cpp/QAicInferenceExample.cpp` | Basic synchronous + async inference |
| `cpp/QAicBenchmarkingExample.cpp` | Throughput benchmarking |
| `cpp/QAicInferenceSetExample.cpp` | Inference set (batched async) |
| `cpp/QAicOversubscriptionExample.cpp` | ProgramGroup with oversubscription |
| `cpp/QAicZeroCopyInferenceExample.cpp` | Zero-copy DMA inference |
| `cpp/QAicInferenceMultiModelExample.cpp` | Multiple models on one device |
| `c/QAicInferenceExampleC99.c` | C-99 inference via function table |
| `c/QAicOversubscriptionExampleC99.c` | C-99 ProgramGroup |

Build all examples:
```bash
cd /opt/qti-aic/examples/cpp
mkdir build && cd build
cmake ..
make
```

---

## 3. Minimal Working Example — C++

This example loads a QPC, runs one synchronous inference, and prints output buffer sizes. It mirrors the pattern in `QAicInferenceExample.cpp`.

```cpp
#include "QAicApi.hpp"
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/qpc_dir>\n";
        return 1;
    }
    std::string qpcPath = argv[1];  // directory containing programqpc.bin
    QID qid = 0;                    // device 0

    try {
        // 1. Create a context for the target device
        std::vector<QID> qidList{qid};
        auto context = qaic::rt::Context::Factory(nullptr /*properties*/, qidList);

        // 2. Load the QPC binary
        //    qpcPath is the directory; Factory appends /programqpc.bin
        auto qpc = qaic::rt::Qpc::Factory(qpcPath);
        if (!qpc) {
            std::cerr << "Failed to load QPC from: " << qpcPath << "\n";
            return 1;
        }

        // 3. Inspect I/O buffer layout
        auto bufferMappings = qpc->getBufferMappings();
        for (auto const &m : bufferMappings) {
            std::string dir = (m.ioType == BUFFER_IO_TYPE_INPUT) ? "IN " : "OUT";
            std::cout << dir << " [" << m.index << "] " << m.bufferName
                      << "  size=" << m.size << " bytes\n";
        }

        // 4. Create an InferenceVector (allocates I/O buffers)
        //    USER_BUFFERS: caller fills inputs; RANDOM fills with random data
        auto inferenceVector = qaic::rt::InferenceVector::Factory(
            bufferMappings,
            qaic::rt::InferenceVector::DataSourceType::USER_BUFFERS);

        // 5. Fill input buffers with your data
        //    getVector() returns vector<QBuffer>; QBuffer has .buf (uint8_t*) and .size
        auto &buffers = inferenceVector->getVector();
        for (auto const &m : bufferMappings) {
            if (m.ioType == BUFFER_IO_TYPE_INPUT) {
                // Example: zero-fill; replace with your actual input data
                std::memset(buffers[m.index].buf, 0, buffers[m.index].size);
            }
        }

        // 6. Program::Factory constructs and loads the program. No separate
        //    load() or activate() calls are needed.
        auto program = qaic::rt::Program::Factory(
            context, qid, "my_program", qpc);

        // 7. Create an ExecObj (binds buffers to program)
        auto execObj = qaic::rt::ExecObj::Factory(context, program);
        execObj->setData(inferenceVector);

        // 8. Run synchronously
        execObj->run(inferenceVector);

        // 9. Read output buffers
        for (auto const &m : bufferMappings) {
            if (m.ioType == BUFFER_IO_TYPE_OUTPUT) {
                const QBuffer &out = buffers[m.index];
                std::cout << "Output [" << m.bufferName << "]: "
                          << out.size << " bytes\n";
                // Cast to your expected type, e.g. float*:
                // const float *fp = reinterpret_cast<const float*>(out.buf);
            }
        }

    } catch (const qaic::ExceptionInit &e) {
        std::cerr << "Init error: " << e.what() << "\n";
        return 1;
    } catch (const qaic::ExceptionRuntime &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

In this C++ API, `Program::Factory` constructs and loads the program; no
separate `load()` or `activate()` calls are needed. The lower-level Python
`qaicrt` binding exposes those lifecycle calls explicitly.

### 3.1 Async Inference

For throughput-oriented workloads, use a `Queue` to submit inferences asynchronously:

```cpp
// Create a queue for async submission
auto queue = qaic::rt::Queue::Factory(context, qid);

// Submit
queue->enqueue(execObj);

// Wait for completion
execObj->waitForCompletion();
```

Multiple `ExecObj` instances can be in-flight simultaneously (one per activation slot). See `QAicBenchmarkingExample.cpp` for a full pipelined throughput loop.

---

## 4. ProgramGroup — Sharing a Single Device Across Multiple Models

### 4.1 What ProgramGroup Is

A **ProgramGroup** is a QAIC runtime concept that allows multiple compiled programs (QPCs) to share the same reserved resources on a **single SoC (QID)** — Virtual Channels, semaphores, NSP cores, and DDR memory. Programs within a group run sequentially on that one device rather than each holding an independent device allocation.

> **Critical constraint: ProgramGroup is single-device only.**
> `qaicCreateProgramGroup` / `ProgramGroup::Factory` takes a single `QID`. There is no multi-device or device-list variant. All programs in the group must be compiled for **`mdp_ts_num_devices = 1`** (no MDP/TS). Attempting to add a tensor-sliced QPC to a ProgramGroup is unsupported and will not work correctly.

**When to use it:** Single-SoC pipelines where multiple models run sequentially — e.g. a classification backbone + post-processing head, or a small encoder + decoder that each fit on one SoC. Without ProgramGroup, each session independently reserves the SoC's resources. With ProgramGroup, the group reserves once and both programs reuse that allocation.

**When NOT to use it:** Any model compiled with `mdp_ts_num_devices > 1` (TS2, TS4, TS8, TS16). Those QPCs span multiple SoCs and cannot be placed in a ProgramGroup. For multi-SoC pipelines (e.g. Llama 3.1 70B prefill at TS16 + decode at TS16), there is currently no equivalent resource-sharing mechanism — device requirements are additive.

### 4.2 Resource Reservation Model

Without ProgramGroup (single-SoC models):
```
model_A  → reserves 1 SoC independently
model_B  → reserves 1 SoC independently
Total required: 2 SoCs (additive)
```

With ProgramGroup (both single-SoC):
```
ProgramGroup reserves 1 SoC once
model_A  runs → uses the group's SoC
model_B  runs → reuses the same SoC
Total required: 1 SoC
```

### 4.3 C++ API Usage

```cpp
#include "QAicApi.hpp"

// 1. Create context and ProgramGroup on a single QID
std::vector<QID> qidList{0};
auto context = qaic::rt::Context::Factory(nullptr, qidList);

QAicProgramGroupProperties_t pgProps{};
pgProps.enableOversubscription =
    QAIC_PROGRAM_GROUP_PROPERTIES_OVERSUBSCRIPTION_ENABLED;  // optional

QID device = 0;  // single SoC — ProgramGroup does not accept a device list
auto pg = qaic::rt::ProgramGroup::Factory(
    context, device, "pipeline_group", &pgProps);

// 2. Load QPCs and add programs (configure state)
// Both QPCs must be compiled with mdp_ts_num_devices=1
auto qpc_encoder = qaic::rt::Qpc::Factory("encoder_qpc/");
auto qpc_decoder = qaic::rt::Qpc::Factory("decoder_qpc/");

qaic::rt::shProgram prog_encoder, prog_decoder;
pg->addProgram(qpc_encoder, prog_encoder, nullptr /*properties*/);
pg->addProgram(qpc_decoder, prog_decoder, nullptr /*properties*/);

// 3. Enable — reserves resources on the single QID, transitions to active state
pg->enable();

// 4. Create ExecObjs
auto exec_encoder = qaic::rt::ExecObj::Factory(context, prog_encoder);
auto exec_decoder = qaic::rt::ExecObj::Factory(context, prog_decoder);
exec_encoder->setData(inferenceVector_encoder);
exec_decoder->setData(inferenceVector_decoder);

// 5. Run sequentially via ProgramGroup queue
pg->enqueue(exec_encoder);
exec_encoder->waitForCompletion();

pg->enqueue(exec_decoder);
exec_decoder->waitForCompletion();

// 6. Cleanup
pg->disable(true /*waitForCompletion*/, 0 /*timeoutUs — 0 = infinite*/);
```

See `/opt/qti-aic/examples/cpp/QAicOversubscriptionExample.cpp` for the full reference implementation.

### 4.4 C API Usage

```c
#include "QAicApi.h"

QAicProgramGroupProperties_t props;
qaicProgramGroupPropertiesInitDefault(&props);

QAicProgramGroup *pg;
QAicQueue *queue;
// Single QID — no multi-device variant exists
qaicCreateProgramGroup(context, &pg, &queue, &props, "pipeline_group", /*QID=*/0);

// Add programs (both must be single-device QPCs)
QAicProgram *prog_encoder, *prog_decoder;
qaicProgramGroupAddProgram(pg, qpc_encoder, &prog_encoder, NULL);
qaicProgramGroupAddProgram(pg, qpc_decoder, &prog_decoder, NULL);

qaicProgramGroupEnable(pg);

// Enqueue ExecObjs to queue sequentially
qaicQueueEnqueue(queue, execObj_encoder);
qaicExecObjWaitForCompletion(execObj_encoder, 0);

qaicQueueEnqueue(queue, execObj_decoder);
qaicExecObjWaitForCompletion(execObj_decoder, 0);

qaicProgramGroupDisable(pg, true, 0);
qaicReleaseProgramGroup(pg);
```

See `/opt/qti-aic/examples/c/QAicOversubscriptionExampleC99.c` for the full reference implementation.

### 4.5 ProgramGroup Lifecycle

| State | Transitions to | Allowed operations |
|---|---|---|
| **Configure** | → Active (via `enable()`) | `addProgram()`, `reset()` |
| **Active** | → Configure (via `disable()`) | `enqueue()` |

- `reset()` removes all programs but keeps VC/semaphore assignments (configure state only)
- `disable(waitForCompletion=true, timeoutUs=0)` blocks until all pending inferences finish before releasing resources. **`timeoutUs` is in microseconds** (0 = infinite wait). Note: the Python docstring incorrectly says milliseconds — the parameter name and C++ header are authoritative.
- `release()` frees everything — only valid when not in use (`QS_BUSY` returned otherwise)

### 4.6 Oversubscription Mode

Setting `enableOversubscription = ENABLED` allows the group to time-multiplex programs on the same cores — useful when programs have different active core counts and you want idle cores reclaimed between steps. For purely sequential pipelines with no concurrency, the default (disabled) is sufficient.

### 4.7 Resources Reserved by ProgramGroup

When `enable()` is called, the group reserves the following on the single QID:

| Resource | Notes |
|---|---|
| Virtual Channels | Communication channel to device |
| Semaphores | Synchronization primitives |
| NSP cores | Max across all programs in group |
| DDR memory (static) | Weight/constant storage |
| DDR memory (dynamic) | Activation buffers |
| L2 memory | On-chip staging memory |
| Multicast IDs | Weight broadcast across cores |
| Networks | Network descriptor slots |

All resources are sized to the **maximum** across all programs in the group.

In the Python binding this reservation happens lazily on the group's first `.run()`, not at construction — see `cloud-ai-qaic-python` §6 for the eager-`setup()` failure mode.

### 4.8 MDP/TS Pipelines — No ProgramGroup Support

For multi-SoC pipelines (any model with `mdp_ts_num_devices > 1`), ProgramGroup cannot be used. Each MDP session independently reserves its SoC set, so device requirements are additive:

```
prefill QPC (TS16, 16 SoCs) + decode QPC (TS16, 16 SoCs) = 32 SoCs = 8 Ultra cards
```

There is no current QAIC runtime mechanism to share SoC reservations across MDP programs. Disaggregated serving pipelines (e.g. GPT-OSS 120B with separate prefill and decode QPCs) are an example — each QPC requires its own independent SoC reservation.

---

## Related Skills

- **Cloud AI 100 – `qaic.Session` Python API** (`cloud-ai-qaic-python` skill) — Python-side ProgramGroup usage via `oversubscription_name`; inference, batching, and latency measurement from Python.
- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Go here first to produce the QPC binaries that this API loads.
- **Cloud AI 100 – SoC Architecture** (`cloud-ai-architecture` skill) — MDP/TS device counts and Ultra card topology; relevant when sizing ProgramGroup device requirements.
- **Cloud AI 100 – `qaic-compile` Detailed Tool Guide** (`cloud-ai-qaic-compile` skill) — Compile QPCs with `mdp_ts_num_devices=1` for ProgramGroup compatibility.
- **Cloud AI 100 – `qaic-runner` Detailed Tool Guide** (`cloud-ai-qaic-runner` skill) — CLI tool for quick inference testing before writing C++ code.
