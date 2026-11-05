# Working in this repository

This repository contains Qualcomm Cloud AI developer resources: model recipes,
tutorials, utilities, samples, and reusable agent skills. Use this file as a
map to the detailed documentation; the README files and skills remain the
source of truth for commands and workflows.

## Start here

- Read [README.md](README.md) for the repository overview and its main content
  areas.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution conventions.
- Read [skills/README.md](skills/README.md) to find the Cloud AI skill that
  matches the task.

## Where to go

- `qaic_models/` is the installable, canonical recipe package for focused
  vision and speech workflows. Start with
  [qaic_models/README.md](qaic_models/README.md) for its recipe catalog, then
  read the relevant recipe README and source files. Its recipes include image
  classification and detection, OCR, instance segmentation, and F5-TTS. For
  recipe work, also use
  [cloud-ai-qaic-models](skills/cloud-ai-qaic-models/SKILL.md).
- `models/` contains the older model recipe tree and model-specific examples.
  For work there, read the nearest README in that subtree.
- `tutorials/` contains notebook-based learning and workflow material for
  model onboarding, performance, profiling, and integrations. Start with
  [tutorials/README.md](tutorials/README.md), then follow the nearest tutorial
  README and its neighboring requirements or scripts.
- `utils/` contains operational helpers for benchmarking and Cloud AI device
  workflows. Start with [utils/README.md](utils/README.md), then read the
  README in the utility being used.
- `samples/` contains small Python and C++ examples showing inference APIs and
  Cloud AI usage. Read the nearest sample README.

For any other directory, start with its nearest README. If there is no more
specific documentation, return to [README.md](README.md) and
[skills/README.md](skills/README.md).

For LLM, VLM, embedding, diffusion, video-generation, or speech-recognition
work, follow the root README's direction to the separate Efficient Transformers
project rather than treating `qaic_models/` as the source for those workflows.

## Skills

Skills provide the procedural detail for specialized work. Select them by the
task, not just by the directory being edited. Use the catalog in
[skills/README.md](skills/README.md), then read the matching `SKILL.md` before
following a workflow such as model onboarding, conversion, compilation,
accuracy, performance, runtime, serving, or device work.

## Workspace and temporary files

Do not use `/tmp` as a workspace, run root, artifact store, cache, clone
location, worktree parent, or report destination. Large or retained project
data belongs in an explicit, user-approved directory under this `cloud-ai`
project root. `/tmp` is appropriate only for small, short-lived intermediate
files that are removed by the command that created them. When a command may
produce model artifacts, logs, downloads, compiled output, media, or other
material data, require an explicit disk-backed project path rather than
silently defaulting to `/tmp`.
