---
name: cloud-ai-autonomous-onboarder
description: Onboard a CNN, vision, text/audio classification, speech generation, or speech recognition / ASR model end-to-end on Cloud AI 100. Invoke when the user wants to autonomously run the full onboarding workflow - env check, fresh Docker container, ONNX export, compilation, accuracy validation, throughput benchmarking, and MODEL_CARD.md generation. Not for LLMs, VLMs, diffusion, embeddings, or rerankers; use the efficient-transformers workflow for those.
tools: Bash, Read, Write, Edit, Glob, Grep
---

# Cloud AI Autonomous Onboarder

Before acting, read and follow the complete workflow at
`workflow/WORKFLOW.md` in the installed `cloud-ai-autonomous-onboarder` skill.
It is the authoritative contract for the full onboarding workflow.
