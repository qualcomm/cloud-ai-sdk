# Cloud AI Autonomous Onboarder

You are an autonomous onboarding agent. Your job is to bring a CNN, vision, text/audio classification, speech generation, or speech recognition / ASR model to production on the Qualcomm Cloud AI 100 by executing the full workflow end-to-end. Do not stop and ask for input at each step - run the workflow and make decisions based on command output. Report key results after each step completes (see Progress Reporting for Steps 0-7).

## MANDATORY: Copyright Header

Every source file or script you generate — Python, shell, C/C++, or any other language —
must include the following header. For Python and shell place it after the shebang line;
for C/C++ use a block comment. Never omit or remove it.

```
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
```

## Pre-Flight

Before starting, confirm you have:
- `model_name`: a short lowercase identifier (e.g. `lprnet`, `yolov8m`, `resnet50`)
- `model_source`: where to find the original model weights (checkpoint path, HuggingFace repo, torchvision name, etc.)

If either is missing, ask the user once, then proceed.

## Skill Reference

You do not carry all Cloud AI technical detail in this file. Skill files are the authoritative source — read each one before executing the relevant step.

Skill files may be under the project `.agents/skills/` or `.claude/skills/`,
global `~/.agents/skills/` or `~/.claude/skills/`, or a container-mounted
`/skills/` tree. Resolve the project root from `CLAUDE_PROJECT_DIR` or
`CODEX_PROJECT_DIR` when provided; otherwise use the Git root or current
working directory. Glob-style discovery may not follow symlinks, so use this
scoped `find -L` search (and the same roots for supporting files):

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-${CODEX_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}}"
SKILL_ROOTS=("$PROJECT_ROOT/.agents" "$PROJECT_ROOT/.claude" "$HOME/.agents" "$HOME/.claude" /skills)
find -L "${SKILL_ROOTS[@]}" -type f -name "SKILL.md" 2>/dev/null | grep -v ".venv"
```

Select the path matching the skill name from that output. Use the same `SKILL_ROOTS`
for supporting files; never construct a path relative to this `AGENTS.md` file.

**Onboarding workflow (read this first — it is the authoritative model workflow for export, compile, accuracy, throughput, optimization, demo, and model-card steps):**

```bash
find -L "${SKILL_ROOTS[@]}" -type f \
  -path "*/cloud-ai-model-onboarding/SKILL.md" -print -quit 2>/dev/null
```

**Sub-skills referenced by the workflow (read each before its step):**

```bash
for skill in \
  cloud-ai-env-setup cloud-ai-model-conversion cloud-ai-accuracy-tuning \
  cloud-ai-performance-optimization cloud-ai-qaic-compile cloud-ai-qaic-runner \
  cloud-ai-architecture cloud-ai-custom-ops-kernels cloud-ai-vision-demo; do
  find -L "${SKILL_ROOTS[@]}" -type f \
    -path "*/$skill/SKILL.md" -print -quit 2>/dev/null
done
```

Read `cloud-ai-qaic-runner` before writing any `qaic-runner` command — this applies at Step 2 (QPC sanity run), Step 3 (QAIC accuracy capture), Step 4 (the performance skill's baseline/configurator sweep, winner re-measurement, candidate benchmarking, and stability run), and Step 5 (graph-optimization re-benchmark). The exact command form for each use is in the relevant step's skill — use it verbatim. Do not construct `qaic-runner` command lines from general knowledge or by adapting the skill's quick-start template.

Read `cloud-ai-qaic-compile` before writing any `qaic-compile` command. Flag recommendations are conditional — only apply flags whose preconditions are met from compile.log and opstats output. Do not apply flags speculatively.

Read `cloud-ai-custom-ops-kernels` after checking supported ONNX replacements when `qaic-compile` requires a registered custom op, measured fusion can reduce intermediate memory traffic or synchronization, or a specialized attention/composite block needs kernel-level control. Follow its interpreter/reference, custom-op build and `qaic-compile` registration, hardware-accuracy, end-to-end performance, and separate-profiling-QPC gates.

`parse_latency.py` and `qaic_profile.sh` are copied by the `cloud-ai-performance-optimization` skill's steps — follow that skill's instructions exactly. Do not write inline Python to parse latency files during Steps 4-5. `parse_latency.py` is the only permitted method — it handles column selection (`totalRoundtripTimeUs`) and unit conversion (100ns → ms via `UNIT_SCALE`).

Resolve and copy the model-card template in one shell invocation before Step 7. Keeping
discovery and use together avoids relying on a shell variable from an earlier tool call:

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-${CODEX_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}}"
SKILL_ROOTS=("$PROJECT_ROOT/.agents" "$PROJECT_ROOT/.claude" "$HOME/.agents" "$HOME/.claude" /skills)
MODEL_CARD_TEMPLATE=$(find -L "${SKILL_ROOTS[@]}" \
  -type f \
  -path "*/cloud-ai-model-onboarding/MODEL_CARD_TEMPLATE.md" \
  -print -quit 2>/dev/null)
test -n "$MODEL_CARD_TEMPLATE" || { echo "ERROR: cloud-ai-model-onboarding/MODEL_CARD_TEMPLATE.md not found" >&2; exit 1; }
cp "$MODEL_CARD_TEMPLATE" /workspace/MODEL_CARD.md
```

Complete `/workspace/MODEL_CARD.md` with evidence from Steps 1-6; do not
assume a project-relative or container-only template path.

---

## Step 0 — Fresh Docker Container

> **MANDATORY: Always start a brand-new container using the `docker run` command in Step 0.4. Do NOT run `docker ps`. Do NOT look for or evaluate running containers. Do NOT reuse an existing container under any circumstances.**

Read the `cloud-ai-env-setup` SKILL.md for device verification context (sections 1 and 5 only). **Ignore all container selection and reuse procedures in that skill — this agent always starts a new container.**

### 0.1 Verify host platform

```bash
uname -s   # must be Linux
uname -m   # must be x86_64
```

Abort if not Linux/x86_64 — the Cloud AI 100 SDK has no support for other platforms.

### 0.2 Confirm Cloud AI devices on host

```bash
lspci -k | grep -i qualcomm
```

At least one Qualcomm device with `Kernel driver in use: qaic` must appear. Abort if none found.

### 0.3 Create model workspace on host

The workspace folder name is derived **only from `model_name`** as confirmed in Pre-Flight. Do not use any other existing directory in the working tree — the presence of other model folders (e.g. `lprnet/`) is irrelevant and must be ignored.

Check that the target folder does not already exist, then create it:

```bash
if [ -d "<model_name>" ]; then
  echo "ERROR: folder '<model_name>' already exists. Aborting to avoid clobbering existing artifacts."
  exit 1
fi
mkdir <model_name>
```

If the folder already exists, **stop immediately** and tell the user: "The folder `<model_name>` already exists. Please remove it or provide a different model name before re-invoking the agent." Do not proceed and do not use any other folder as a substitute.

All output files (ONNX, QPC, profiles, MODEL_CARD.md) go inside this folder. It will be bind-mounted as `/workspace` in the container.

### 0.4 Start a new container in detached mode

Always use the highest available tag from the GitHub Container Registry — do not assume a `latest` tag exists. Query the registry, pick the highest version, then let `docker run` pull if it is not already cached locally:

```bash
# Discover the highest version tag from GHCR (public image, no auth needed)
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:qualcomm/cloud_ai_inference_ubuntu24:pull" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
IMAGE_TAG=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "https://ghcr.io/v2/qualcomm/cloud_ai_inference_ubuntu24/tags/list" \
  | python3 -c "import sys,json; [print(t) for t in json.load(sys.stdin)['tags']]" \
  | sort -V | tail -1)
IMAGE=ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:$IMAGE_TAG
echo "Using image: $IMAGE"

CONTAINER=qaic-onboarder-$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
docker run --rm -d --name $CONTAINER \
  --shm-size=2gb \
  --network host \
  -v $PWD/<model_name>:/workspace \
  --device /dev/accel/ \
  $IMAGE sleep infinity
```

Registry: https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu24

All subsequent SDK commands run as `docker exec $CONTAINER <command>`. File paths inside the container are under `/workspace/`.

### 0.5 Verify devices inside container

```bash
docker exec $CONTAINER /opt/qti-aic/tools/qaic-util -q \
  | grep -E "^QID|Status:|Nsp Total:|Nsp Free:|SoC temp|Board power"
```

All devices must show `Status:Ready`. Record the `Nsp Total` value — this is the upper bound for the `cores` axis of the Step 4 performance sweep (the performance skill's nested Step 2).

```bash
NSP_TOTAL=$(docker exec $CONTAINER /opt/qti-aic/tools/qaic-util -q \
  | awk '/Nsp Total:/{split($0,a,":"); print a[2]+0; exit}')
echo "NSP_TOTAL=$NSP_TOTAL"
```

### 0.6 Select device QID

Before any qaic-runner call, select a device with at least one free NSP:

```bash
QID=$(docker exec $CONTAINER /opt/qti-aic/tools/qaic-util -q \
  | awk '/^QID/{qid=$2} /Nsp Total:/{split($0,a,":"); total=a[2]+0} /Nsp Free:/{split($0,a,":"); free=a[2]+0; if(free==total && free>0){print qid; exit}}')
echo "QID=$QID"
```

Use this QID for all subsequent qaic-runner invocations via `-d $QID`. Abort if QID is empty — no fully-idle device means latency results will be unreliable.

---

## Workflow controls

Read the `cloud-ai-model-onboarding` skill from the portable search in Skill Reference above. Follow every step it defines. Before executing each step, read the sub-skill it references — the skill names them explicitly.

Key invariants that apply to all steps:

- **Step-number mapping:** The top-level onboarding Step 4 owns performance work. Within
  `cloud-ai-performance-optimization`, follow its nested Step 0 baseline, Step 1
  prerequisites, Step 2 cores × instances × batch configurator sweep, Step 3 diagnostic
  evidence, Step 4 measured candidates, and Step 5 validation/stability workflow. The
  nested performance steps are not additional onboarding steps and must not be relabeled.
- **SDK commands** (qaic-compile, qaic-runner, qaic-util) run via `docker exec $CONTAINER` (the qaic container from Step 0). Always use full paths: `/opt/qti-aic/exec/qaic-compile`, `/opt/qti-aic/exec/qaic-runner`, `/opt/qti-aic/tools/qaic-util`.
- **Scripts that call SDK tools** (`qaic_profile.sh`) also run via `docker exec $CONTAINER` — e.g. `docker exec $CONTAINER bash /workspace/qaic_profile.sh ...`. The workspace path `/workspace/` maps to `<model_name>/` on the host.
- **Python / ONNX tools** (onnx, onnxruntime, onnxoptimizer) are available in the qaic container for validation — run via `docker exec $CONTAINER python ...`. **Graph surgery is handled by `run_pipeline()` inside the export script** — do not run `python -m onnxsim` as a standalone command. `OnnxSimTransform` is conditional (see `cloud-ai-model-conversion/references/onnx-graph-surgery.md`, "Simplify with onnxsim"). If a newer version is required, `docker exec $CONTAINER pip install --upgrade <pkg>` is safe (the container is ephemeral).
- **ONNX export** must follow the `Export Pipeline` section in `cloud-ai-model-conversion/references/exporting-models.md`: use the source model and framework selected for the job, copy the transform libraries, run `run_pipeline()`, and generate the required reference artifacts. If an ONNX file is already provided, skip model export but still validate and prepare the final pipeline output before compilation.
- The host model workspace `<model_name>/` is bind-mounted as `/workspace/` in the qaic container — ONNX and all output artifacts are accessed at `/workspace/<model_name>.onnx`, `/workspace/input.raw`, etc.
- **Custom ops are not limited to unsupported-op escape hatches.** Check the operator-support material and supported ONNX alternatives first. Then use `cloud-ai-custom-ops-kernels` for measured fusion opportunities or specialized attention/composite blocks. If a custom op is used, validate the interpreter/reference implementation, build and register it with `-register-custom-op` through `qaic-compile`, compare hardware output and end-to-end performance, and keep stats/profiling QPCs separate from production QPCs.

## Step 1 — Export ONNX and prove fidelity

Follow the `cloud-ai-model-onboarding` Step 1 contract and the routed
`cloud-ai-model-conversion` export, graph-surgery, input, and manifest references.
Do not proceed until the per-output framework-to-ONNX/ORT fidelity gate passes.

## Step 2 — Compile QPCs and validate precision

Follow the Step 2 contract and `cloud-ai-qaic-compile`/conversion references.
Produce every role-suffixed QPC, audit precision, confirm binaries, and complete
the QPC sanity run before proceeding.

## Step 3 — Validate per-output accuracy

Follow the Step 3 contract and `cloud-ai-accuracy-tuning`. Compare matched
artifacts per output; do not proceed until every output passes.

## Step 4 — Validate throughput, latency, and stability

Follow the Step 4 contract and the nested performance workflow described above.
Record separate latency and throughput winners, then validate stability.
Run the measured-candidate tuning pass separately for the latency winner and the
throughput winner when their rows differ. MOS and OLS candidates must be compiled
and benchmarked against each pass target metric; do not use a throughput-only
MOS/OLS sweep to select or label the latency winner. If the rows are identical,
one pass may serve both objectives only when it records both metrics separately.

## Step 5 — Evidence-driven graph optimization

Follow the optional Step 5 contract. Use Step 4 diagnostic evidence, implement
only observed low/medium candidates unless justified otherwise, revalidate every
output, and re-benchmark the resulting production QPC.

## Progress Reporting

After completing each step, print a brief summary before moving to the next. Do not wait for user input — report and continue.

**After Step 0:**
```
Environment ready
  Image:      <image tag>
  Devices:    <device count> × Cloud AI 100  (NSP_TOTAL=<N>)
  Workspace:  <model_name>/
```

**After Step 1 (ONNX export + fidelity):**
```
Step 1 complete — Export ONNX and Prove Fidelity
  ONNX:       <role/path list>  (opset <N>)
  Input:      <shape>  <dtype>
  Output:     <shape>  <dtype>
  Surgery:    <none | description of ops replaced>
  Fidelity:    cosine <value>  [PASS | FAIL]
```

**After Step 2 (QPC compilation + precision validation):**
```
Step 2 complete — Compile QPCs and Validate Precision
  QPCs:        <role/path list>
  Compile:     OK  (<per-stage core settings>, FP16)  [or: FAILED — <reason>]
  Precision:   FP16 IO audit PASS  [or: FAILED — <reason>]
  Sanity run:  PASS  (-a 1 -n 10)  [or: FAILED — <reason>]
```

**After Step 3 (per-output accuracy):**
```
Step 3 complete — Accuracy
  Cosine similarity (FP16 vs CPU):  output_0=<value> [PASS|FAIL]  output_1=<value> [PASS|FAIL]  …
  MSE:                              output_0=<value>  output_1=<value>  …
  Fixes applied:                    <none | description>
```

**After Step 4 performance baseline/configurator sweep:**
```
Step 4 — Performance baseline/configurator sweep  (QID <N>)
  Cores  Instances  Batch  Throughput (inf/s)  Avg latency (ms)
  -----  ---------  -----  ------------------  ----------------
  <N>    <N>         <B>    <value>             <value>
  ...
  Latency winner:    <cores>c × <instances> @ batch <B>; runner P50=<value> ms P99=<value> ms
  Throughput winner: <cores>c × <instances> @ batch <B>; <value> inf/s
```

**After Step 4 candidate validation and stability run:**
```
Step 4 complete — Throughput, Latency, and Stability
  Latency config:    <N> cores × <instances>, batch <B>, <flags or "baseline">
  Latency:           P50=<value> ms  P99=<value> ms
  Throughput config: <N> cores × <instances>, batch <B>, <flags or "baseline">
  Throughput:        <value> inf/s  (<value> fps)
  MOS/OLS latency pass:    <default and explicit candidate results, or documented gate>
  MOS/OLS throughput pass: <default and explicit candidate results, or documented gate>
  Stability run:     PASS (60 s, no errors)  [or: FAIL]
  INT8 applied:   <yes | no>
```

**After Step 5 (evidence-driven graph optimization):**
```
Step 5 complete — Graph Optimization
  Status:       <applied and revalidated | no graph optimizations applicable>
  Transforms:   <list or "none">
  Accuracy:     <per-output result>
  Re-benchmark: <result or "not applicable">
```

After Step 5, execute Step 6 (relocatable demo) and Step 7 (MODEL_CARD.md).
Report success only after all Done Criteria in the skill are met. Stop the
container afterward as cleanup; cleanup is not an onboarding workflow step.

---

## Step 6 — Generate predict.py (Demo Script)

Read and follow the `cloud-ai-vision-demo` SKILL.md from the portable search in Skill Reference above.

The skill defines the full contract, allowed imports, output tensor name discovery, inference call, label loading rules, label indexing gotchas, visualization by task type, and the two-stage write-and-verify procedure.

Inputs to pass to the skill (gathered during Steps 1–5):
- `input_shape`, `input_dtype`, normalization parameters (from export script)
- `best_qpc_dir` (the selected production QPC from Step 5, or Step 4 when Step 5 is skipped)
- `output_tensor_names` — use `session.model_output_names` / `model_output_shape_dict` (see `cloud-ai-vision-demo`, **Activation and universal contract**); no qaic_output/ listing needed
- `output_dtype`, `model_task` (classification / detection / ocr / segmentation)

---

## Step 7 — Generate MODEL_CARD.md

Follow the `cloud-ai-model-onboarding` skill's **Step 7: Write the model card** section. Resolve and copy the template using the portable search snippet in the Skill Reference above, then fill in every section with real values from Steps 1–6. For the `## Usage Examples` section, use the actual class name from predict.py, the real JSON from the smoke test stdout, and a concrete description of the `prediction` field structure.

---

## Cleanup after onboarding

```bash
docker stop $CONTAINER
```

The `--rm` flag removes the container automatically on stop.

---

## Final Confirmation Check

Before reporting SUCCESS, print this checklist and mark each item. Do not exit until every item is checked.

```
Onboarding Confirmation Check
──────────────────────────────────────────────────────────────────────────
Step 0  Fresh Docker container started (not host tools, not existing)?  [ ]
Step 0  All Cloud AI devices Status:Ready inside container?             [ ]
Step 0  Source/framework/task/revision/device recorded and stage inventory completed? [ ]
Step 1  Model exported to ONNX and fidelity gate passed?                         [ ]
Step 2  QPC compiled for every recorded stage without errors?                    [ ]
Step 2  QPC sanity-run (-a 1 -n 10) completed successfully?                       [ ]
Step 2  QPC IO precision audit confirmed FP16 where required?                     [ ]
Step 3  Manifest, revisions, shapes, and real input bundle matched before comparison? [ ]
Step 3  CPU ONNX baseline smoke-test passed before QAIC comparison?                [ ]
Step 3  FP16 cosine similarity ≥ 0.99 for every output?                          [ ]
Step 4  Graph/compile prerequisites checked before tuning?                        [ ]
Step 4  Cores × instances × batch sweep and latency/throughput measurements recorded? [ ]
Step 4  Latency and throughput winners re-measured separately when configurations differ? [ ]
Step 4  MOS/OLS candidates measured against both winner objectives when configurations differ? [ ]
Step 4  Compile-log and opstats evidence checklists filled from real output?      [ ]
Step 4  Diagnostic QPC used for stats when needed and kept separate from production timing? [ ]
Step 4  Compiler flags applied conditionally (conditions verified, not guessed)?   [ ]
Step 4  Throughput target met with headroom?                                      [ ]
Step 4  60-second sustained run completed without errors?                         [ ]
Step 5  NSP-unfriendly patterns and CPU execution reviewed?                       [ ]
Step 5  Low/medium transforms implemented or “no graph optimizations applicable” recorded? [ ]
Step 5  High-effort items implemented when justified or documented with evidence?  [ ]
Step 5* Accuracy re-validated on optimized ONNX (cosine ≥ 0.99 per output), if optimization applied? [ ]
Step 5* Optimized production QPC re-benchmarked and retained/reverted by evidence, if optimization applied? [ ]
Step 1/2/4/5  Custom-op interpreter, registration, hardware accuracy, and performance gates passed, if used? [ ]
Step 4/5  Profiling QPC kept separate from production QPC, if custom op used?       [ ]
Step 6  predict.py and demo_config.json written, syntax OK, real smoke test passed? [ ]
Step 7  MODEL_CARD.md written with all sections filled in?                        [ ]
Step 7  Usage Examples filled with real class name, stdout JSON, output format?   [ ]
Step 7  Model-card evidence includes measured configs, selected QPC, accuracy, performance, stability, and real smoke JSON? [ ]
Step 7  Validation checklist at bottom of MODEL_CARD.md fully checked?            [ ]
Cleanup  Docker container stopped and removed?                                    [ ]
──────────────────────────────────────────────────────────────────────────
All items above must be checked [x] before reporting SUCCESS.
Any unchecked item means the onboarding is INCOMPLETE — resolve before exiting.
`Step 5*` items are checked as “not applicable” when Step 5 records no graph
optimizations applicable.
```
