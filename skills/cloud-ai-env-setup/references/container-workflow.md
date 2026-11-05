# Container Workflow

## Reuse or Start an SDK Container

| State | Action |
|---|---|
| Agent is already running in the work container | Verify the current environment, then continue directly. |
| The user specifies a container, or the current task already created one | Verify it, then continue with `docker exec`. |
| Otherwise | Create a new container using the start procedure below. |

For the current environment:

```bash
test -d /workspace && test -w /workspace
test -e /dev/accel/accel0
test -x /opt/qti-aic/exec/qaic-compile
test -x /opt/qti-aic/exec/qaic-runner
test -x /opt/qti-aic/tools/qaic-util
```

For a named container:

```bash
CONTAINER=<user-specified-or-current-task-container>
docker inspect "$CONTAINER" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'
docker exec "$CONTAINER" test -d /workspace
docker exec "$CONTAINER" test -e /dev/accel/accel0
docker exec "$CONTAINER" test -x /opt/qti-aic/exec/qaic-compile
docker exec "$CONTAINER" test -x /opt/qti-aic/exec/qaic-runner
docker exec "$CONTAINER" test -x /opt/qti-aic/tools/qaic-util
```

For a named container, confirm that the mount source is the intended host workspace
and its destination is `/workspace`. If verification fails, create a new container
using the start procedure below.
Commands later in this reference omit `docker exec "$CONTAINER"` when the agent is
already running in the work container.

## Start the Docker Container

First confirm Docker is installed and the daemon is running:

```bash
docker version 2>&1 | head -5
```

If this fails with `command not found` or `Cannot connect to the Docker daemon`, Docker is not available. Inform the user: Docker must be installed and the daemon started before this path is available. Direct them to [Docker Engine installation](https://docs.docker.com/engine/install/) for their distro. If Docker cannot be installed, the only path forward is a bare-metal SDK installation on the host combined with `qaic` group membership. See [Platform SDK installation](platform-sdk-installation.md) and [host preflight](host-preflight.md).

The official Cloud AI inference container image is available at:
<https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu24>

Before starting a container, find the latest locally available image:

```bash
docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep "cloud_ai_inference_ubuntu24" \
  | sort -V | tail -1
```

Use the tag returned. If no local image is present, check the [container registry](https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu24) for the latest tag and pull it. Do not hardcode a version — always use the latest available.

Start the container:

```bash
IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep "cloud_ai_inference_ubuntu24" | sort -V | tail -1)
CONTAINER=qaic-dev-$(date +%s)

docker run --rm -d --name "$CONTAINER" \
  --shm-size=2gb \
  --network host \
  -v $PWD:/workspace \
  --device /dev/accel/ \
  "$IMAGE" sleep infinity
```

**Mount only the project directory (`$PWD`), never a parent folder.** Mounting a parent exposes unrelated files to the container. If the user's working directory is `/foo/bar/myproject`, the mount must be `-v /foo/bar/myproject:/workspace`, not `-v /foo/bar:/workspace` or `-v /foo:/workspace`.

**Do not add extra `-v` mounts** beyond the workspace. In particular, never mount `/dev`, `/opt/qti-aic`, or any other host system directory — these override the container's own SDK and device management. Use `--device /dev/accel/` for device access, not `-v /dev:/dev`.

Flag reference:

| Flag | Purpose |
|---|---|
| `--rm` | Remove the container automatically when it exits |
| `-d` | Runs the container in the background for subsequent `docker exec` commands |
| `--name` | Assigns the container name retained for the current task |
| `--shm-size=2gb` | Shared memory required for QAic data transfers |
| `--network host` | Direct host networking (required for Triton gRPC clients and similar) |
| `-v $PWD:/workspace` | Bind-mounts the current host directory as `/workspace` inside the container |
| `--device /dev/accel/` | Passes all Cloud AI accelerator devices into the container |

Run subsequent commands from the host with `docker exec "$CONTAINER" ...`. Keep
generated files under `/workspace` so they persist in the host workspace.

**SDK version:**
```bash
docker exec "$CONTAINER" /opt/qti-aic/tools/qaic-version-util --platform
docker exec "$CONTAINER" /opt/qti-aic/tools/qaic-version-util --apps
```

**Non-root containers:** The default inference container runs as root — no extra steps needed. If the container runs as a non-root user (e.g., `-u $(id -u):$(id -g)`), the user must be in the `qaic` group to open `/dev/accel/` devices (`crw-rw---- root:qaic`). Fix — choose one:
- **Run as root (simplest):** omit the `-u` flag.
- **Keep non-root user:** add `--group-add $(getent group qaic | cut -d: -f3)` to the `docker run` command:

```bash
CONTAINER=qaic-dev-$(date +%s)
docker run --rm -d --name "$CONTAINER" \
  --shm-size=2gb \
  --network host \
  -v $PWD:/workspace \
  --device /dev/accel/ \
  --group-add $(getent group qaic | cut -d: -f3) \
  -u $(id -u):$(id -g) \
  "$IMAGE" sleep infinity
```

If `getent group qaic` returns nothing on the host, install the Platform SDK first — it creates the `qaic` group. See [Platform SDK installation](platform-sdk-installation.md).
