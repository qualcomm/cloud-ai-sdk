# Cloud AI Skills

Reusable Agent Skills for Qualcomm Cloud AI accelerators. They cover model
export, QPC compilation, validation, performance tuning, runtime APIs,
deployment, and low-level accelerator development.

The canonical content uses the portable `SKILL.md` format. Claude Code and
Codex discover those skills through their harness-specific skill locations.
The same checkout can serve both runtimes; the autonomous skill also contains
its workflow and Claude Code agent payload.

## Catalog

| Area | Skills |
| --- | --- |
| End-to-end model workflow | `cloud-ai-autonomous-onboarder`, `cloud-ai-model-onboarding`, `cloud-ai-llm-onboarding`, `cloud-ai-model-conversion`, `cloud-ai-accuracy-tuning`, `cloud-ai-performance-optimization`, `cloud-ai-vision-demo` |
| Inference serving and deployment | `cloud-ai-kubernetes`, `cloud-ai-vllm-serving`, `cloud-ai-triton-serving`, `cloud-ai-inference-suite` |
| Compilation and runtime | `cloud-ai-qaic-compile`, `cloud-ai-qaic-runner`, `cloud-ai-qaic-python`, `cloud-ai-qaic-cpp`, `cloud-ai-torch-qaic` |
| Accelerator development | `cloud-ai-custom-ops-kernels`, `cloud-ai-compute-sdk` |
| Recipe maintenance | `cloud-ai-qaic-models` |
| Platform | `cloud-ai-env-setup`, `cloud-ai-architecture`, `cloud-ai-monitoring-diagnostics` |

`cloud-ai-autonomous-onboarder` is the autonomous non-LLM onboarding workflow. Its
shared workflow is exposed as a skill for Codex and as both a skill and an
agent for Claude Code.

## Layout

```text
skills/                       # 22 canonical SKILL.md directories
tests/                        # Static guidance and integration checks; not installed
docs/                         # Skill evaluations and design history
```

Each skill directory contains `SKILL.md` and may include focused `references/`,
`scripts/`, `templates/`, or `examples/` when the workflow needs them.

## Installation

Start from a checkout of this repository. Copy the canonical
`skills/cloud-ai-*` directories into the harness skill directory. Each
installation method is additive and leaves existing files and skill directories
unchanged.

The shell commands below work on macOS, Linux, or WSL. Windows users can copy
the folders with File Explorer.

### Codex

Codex normally discovers user skills under `~/.agents/skills/` and project
skills under `.agents/skills/`:

#### macOS, Linux, or WSL

```bash
# From the cloud-ai-skills checkout, for user-wide installation:
mkdir -p ~/.agents/skills
cp -R -n skills/cloud-ai-* ~/.agents/skills/

# Or, from a project root, for project-local installation:
mkdir -p .agents/skills
cp -R -n /path/to/cloud-ai-skills/skills/cloud-ai-* .agents/skills/
```

#### Windows

Open the `skills` folder in this checkout and copy every `cloud-ai-*` folder
to `%USERPROFILE%\.agents\skills\` for a user-wide installation, or to
`.agents\skills\` in the project folder. Create the destination folder if it
does not exist. Leave an existing skill folder in place.

### Claude Code

Claude Code normally discovers skills under `~/.claude/skills/` or
`.claude/skills/`. The autonomous workflow also uses the agent file under
`~/.claude/agents/` or `.claude/agents/`:

#### macOS, Linux, or WSL

```bash
# From the cloud-ai-skills checkout, for user-wide installation:
mkdir -p ~/.claude/skills ~/.claude/agents
cp -R -n skills/cloud-ai-* ~/.claude/skills/
cp -R -n skills/cloud-ai-autonomous-onboarder/agents/cloud-ai-autonomous-onboarder ~/.claude/agents/

# Or, from a project root, for project-local installation:
mkdir -p .claude/skills .claude/agents
cp -R -n /path/to/cloud-ai-skills/skills/cloud-ai-* .claude/skills/
cp -R -n /path/to/cloud-ai-skills/skills/cloud-ai-autonomous-onboarder/agents/cloud-ai-autonomous-onboarder .claude/agents/
```

#### Windows

Open the `skills` folder in this checkout and copy every `cloud-ai-*` folder
to `%USERPROFILE%\.claude\skills\` for a user-wide installation, or to
`.claude\skills\` in the project folder. Open
`skills/cloud-ai-autonomous-onboarder/agents/` in this checkout and copy
`cloud-ai-autonomous-onboarder` to `%USERPROFILE%\.claude\agents\`, or to
`.claude\agents\` in the project folder. Create destination folders if they do
not exist. Leave existing skill and agent folders in place.

To update a copied skill, replace only that skill directory when you own it,
then copy it again. Do not remove an existing directory unless you know it was
installed from this checkout.

### Verify installation

Start a new harness session and ask:

> List every installed Cloud AI skill. Include its exact skill name and source path.

The response should list the copied `cloud-ai-*` skills and their installed
locations. If it does not, confirm the destination folder and restart the
harness.

### Other harnesses

For a compatible harness other than Claude Code or Codex, consult that
harness's documentation for its skill discovery location and copy method. Copy
each `skills/cloud-ai-*` directory without replacing an existing directory.
