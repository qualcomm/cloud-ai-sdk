---
name: cloud-ai-architecture
description: Cloud AI 100 accelerator architecture, NSP cores, memory hierarchy, and Hexagon v68 DSP details. Use when user asks about hardware architecture, core specs, memory model, or low-level compute units.
---

# Cloud AI 100 – Architecture & Hardware Reference

This skill provides the foundational hardware architecture knowledge for Cloud AI 100 accelerators. Use this alongside:

- **Performance Optimization** (`cloud-ai-performance-optimization`) — Flag tuning and profiling workflows
- **Graph and compile prerequisites** (`cloud-ai-performance-optimization`, Step 1, `references/graph-optimization.md`) — Channel alignment, graph structure, VTCM, and partitioning checks
- **Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — ONNX/TF to QPC workflow
- **Accuracy Tuning** (`cloud-ai-accuracy-tuning` skill) — FP16/INT8 quantization

## Reference Documents

**Always read the relevant reference before analysing opstats, compiler metrics, or answering architecture questions. This SKILL.md alone contains no hardware detail.**

| Reference | Path | When to read |
|-----------|------|--------------|
| SoC Architecture | `references/cloud_ai_soc_architecture.md` | Step 1 ONNX export (identifying hardware-unfriendly ops and graph structures that need replacement or custom kernels), Step 5 graph optimization (interpreting opstats — HMX/HVX utilisation, VTCM sizing, barrier overhead, sync cycles), compiler flag selection (`-mos`, `-ols`) |
| Hexagon v68 DSP Architecture | `references/cloud_ai_hexagon_v68_dsp_architecture.md` | Custom op development, HVX vector programming, DSP pipeline details |
| Hexagon v68 Programming Reference | `references/cloud_ai_hexagon_v68_programming_reference.md` | Writing HVX intrinsics, custom kernel authoring |
| Transformer Architecture | `references/transformer_architecture.md` | Any model with attention mechanisms or transformer encoder/decoder blocks. |

**If the model under onboarding has transformer or attention blocks** (ViT, BEiT, Swin, DETR, LayoutLM, any model with multi-head attention or encoder/decoder structure), you MUST:

```
read(path="/skills/cloud-ai-architecture/references/transformer_architecture.md", permanent=True)
```

Do this at Step 1 (immediately after identifying the architecture, before writing any export code) and again at Step 5 (before interpreting opstats or choosing graph transforms). Do not treat the Step 0 `soc_architecture.md` read as satisfying this requirement — it does not.

Read `references/cloud_ai_soc_architecture.md` at **Step 1** (before graph surgery — to know which op patterns are HMX-friendly and which require replacement or custom kernels) and again at **Step 5** (before interpreting opstats — to understand HMX sync protocol, HVX barrier overhead, VTCM capacity, and data layout constraints).
