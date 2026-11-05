---
name: cloud-ai-env-setup
description: Set up the Cloud AI 100 development environment. Use when user asks to check for devices, enable PCIe P2P for Ultra MDP/TS workloads, start a Docker container, install the SDK, verify device status, or configure tool paths and Python packages for Cloud AI 100 work.
---

# Cloud AI 100 - Environment Setup

## Overview

Run SDK tools in a container while the host provides the QAIC kernel driver and
accelerator device firmware. This quick start creates a workspace, starts a current
inference container, and verifies a ready device.

## Search Tool Availability

Before using rg on the host or in a container, verify it is available in that exact execution environment; use an appropriate fallback if it is not.

## Quick Reference

A Cloud AI workload needs all of the following:

- The host owns the QAIC kernel driver and accelerator devices under
  `/dev/accel/`; the driver gives the container access to the hardware.
- A persistent host workspace preserves work after the container exits.
- An SDK inference container provides the QAIC tools, compiler, and runtime.

## Choose the Container Path

Choose the execution context before performing host setup:

| State | Action |
|---|---|
| Agent is already running in the work container | Verify the current environment, then continue directly. |
| The user specifies a container, or the current task already created one | Verify it, then continue with `docker exec`. |
| Otherwise | Create a new container using the host setup below. |

For the current environment:

Verify the workspace, device access, and SDK tools. Run subsequent QAIC
commands directly, without `docker exec`:

```bash
test -d /workspace && test -w /workspace
test -e /dev/accel/accel0
test -x /opt/qti-aic/exec/qaic-compile
test -x /opt/qti-aic/exec/qaic-runner
test -x /opt/qti-aic/tools/qaic-util
/opt/qti-aic/tools/qaic-version-util --platform
/opt/qti-aic/tools/qaic-util -q \
  | grep -E '^QID|Status:|Nsp Total:|Nsp Free:'
```

Every QID selected for work must report `Status:Ready`; use `Nsp Free` to
ensure it has enough free NSPs for the workload. If any check fails, use the
applicable reference below rather than starting another container.

For a named existing container, inspect its mount:

```bash
CONTAINER=<user-specified-or-current-task-container>
docker inspect "$CONTAINER" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'
```

Confirm that the intended host workspace is mounted at `/workspace`, then run
the current-environment validation commands with `docker exec "$CONTAINER"`
prefixed to each command. If they pass, keep using `docker exec "$CONTAINER"
<command>` and do not create a new container.

## Create a Container

Only when no usable container exists, run these checks on the host. The platform
must be `Linux` / `x86_64`. If `lspci -k` does not show `Kernel driver in use:
qaic` for each accelerator, follow the Platform SDK installation reference,
then rerun these checks. If no accelerator is listed, resolve host PCIe/device
availability before creating a container:

```bash
set -euo pipefail

test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
lspci -k

WORKSPACE="${WORKSPACE:-$HOME/qaic-workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
docker version
```

Prefer the highest local inference image. If none is available, discover the
highest public version and let Docker pull it. The fallback uses `curl` and
`python3`, not an extra registry client:

For SDK 1.21 and earlier, use matching tags from the legacy `ghcr.io/quic`
repositories: `cloud_ai_inference_ubuntu24`, `cloud_ai_triton_server`, and
`cloud_ai_inference_vllm`. For example:
`docker pull ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0`. The discovery
command below queries `ghcr.io/qualcomm`; pull a legacy tag explicitly when
targeting an older SDK.

```bash
IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep 'cloud_ai_inference_ubuntu24' | sort -V | tail -1 || true)
if [ -z "$IMAGE" ]; then
  TOKEN=$(curl -fsSL 'https://ghcr.io/token?scope=repository:qualcomm/cloud_ai_inference_ubuntu24:pull' \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
  IMAGE_TAG=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
    'https://ghcr.io/v2/qualcomm/cloud_ai_inference_ubuntu24/tags/list' \
    | python3 -c 'import json, sys; print("\\n".join(json.load(sys.stdin)["tags"]))' \
    | sort -V | tail -1)
  test -n "$IMAGE_TAG"
  IMAGE="ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:$IMAGE_TAG"
fi
CONTAINER="qaic-dev-$(date +%s)"

docker run --rm -d --name "$CONTAINER" \
  --shm-size=2gb \
  --network host \
  -v "$WORKSPACE":/workspace \
  --device /dev/accel/ \
  "$IMAGE" sleep infinity
```

Verify the mounted workspace, SDK, and devices. Every QID selected for work
must report `Status:Ready`; use `Nsp Free` to ensure it has enough free NSPs
for the workload:

```bash
docker exec "$CONTAINER" test -d /workspace
docker exec "$CONTAINER" test -e /dev/accel/accel0
docker exec "$CONTAINER" /opt/qti-aic/tools/qaic-version-util --platform
docker exec "$CONTAINER" /opt/qti-aic/tools/qaic-util -q \
  | grep -E '^QID|Status:|Nsp Total:|Nsp Free:'
```

Run SDK commands as `docker exec "$CONTAINER" <command>` and keep generated
files under `/workspace`. Use the task-specific onboarding, compilation, or
runtime skill after this check succeeds.

If a check fails or the task needs a specialized workflow, use the applicable
reference below rather than altering the standard setup blindly.

## Reference Routing

1. [Host preflight and local SDK access](references/host-preflight.md): platform,
   driver, workspace, host tools, and `qaic` group access.
2. [Platform SDK installation and components](references/platform-sdk-installation.md):
   full bare-metal runtime or container-only driver installation.
3. [Container workflow](references/container-workflow.md): existing containers,
   non-root containers, and container-startup failures.
4. [Device validation and smoke test](references/device-validation.md): inspect
   QIDs and run the bundled ResNet-50 fixture.
5. [PCIe P2P reference](references/pci-p2p.md): Ultra MDP/TS VM preflight, ACS
   enablement, and synthetic P2P tests.
6. [Tool paths and Python environment](references/tool-paths-python.md): QAIC
   paths, `qaic.Session`, and model-work Python dependencies.
