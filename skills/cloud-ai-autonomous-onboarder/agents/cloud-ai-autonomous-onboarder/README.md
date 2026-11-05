# cloud-ai-autonomous-onboarder

Autonomous onboarding agent for bringing CNN/vision models to production on Qualcomm Cloud AI 100.

## Making the Agent Fully Autonomous

By default Claude Code prompts for approval before running each Bash command. To let this agent run without interruption, choose one of the two options below.

### Option A — `allowedTools` in project settings (recommended)

Add specific command allowances to `.claude/settings.local.json`. Scoped to the project; does not affect other projects or global settings.

```json
{
  "allowedTools": [
    "Bash(docker run *)",
    "Bash(docker exec *)",
    "Bash(docker stop *)",
    "Bash(docker images *)",
    "Bash(curl *)",
    "Bash(lspci *)",
    "Bash(uname *)",
    "Bash(mkdir *)",
    "Bash(python3 *)"
  ]
}
```

Merge with any existing content in that file.

### Option B — `permissionMode` in AGENTS.md

Add `permissionMode: bypassPermissions` to the frontmatter of `AGENTS.md`:

```yaml
---
name: cloud-ai-autonomous-onboarder
description: ...
tools: Bash, Read, Write, Edit, Glob, Grep
permissionMode: bypassPermissions
---
```

This bypasses all permission prompts for this agent only — it does not affect the main session or other agents.

## Usage

Invoke from any Claude Code session:

```
Onboard my lprnet model — weights at /workspace/checkpoints/lprnet.pth
```

Or explicitly:

```
Use the cloud-ai-autonomous-onboarder agent to onboard resnet50
```

Or by @-mention (type `@` in the UI and select from typeahead).

The agent needs two things to start:
- `model_name` — short lowercase identifier (e.g. `lprnet`)
- `model_source` — weights location (checkpoint path, torchvision name, HuggingFace repo)

It runs the full workflow autonomously and reports back when all checklist items are complete.
