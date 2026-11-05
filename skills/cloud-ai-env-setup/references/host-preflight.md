# Host Preflight and Local SDK Access

## Verify Host Platform

Run these two checks first. Do not proceed if the platform is not supported.

```bash
uname -s   # operating system
uname -m   # CPU architecture
```

If the output is `Linux` / `x86_64`, continue to check available devices.

---

## Check for Available Devices

Run the following on the host to confirm Cloud AI accelerator devices are present and the kernel driver is loaded:

```bash
lspci -k
```

Look for Qualcomm device entries like the following. You should see one block per device. The `Kernel driver in use` field must read `qaic` for each.

```
03:00.0 Processing accelerators: Qualcomm Device a100
        Subsystem: Qualcomm Device a100
        Kernel driver in use: qaic
        Kernel modules: qaic
```

If `Kernel driver in use` is not `qaic`, install the Platform SDK from the
[Qualcomm Software Center](https://softwarecenter.qualcomm.com/catalog/item/aic_platform)
as described in the [Platform SDK installation reference](platform-sdk-installation.md).

If no devices appear at all, the host driver is not loaded or no cards are present. Do not proceed until at least one device is visible with the `qaic` driver.

---

## Prepare a Persistent Workspace Folder

Create a folder on the host to store models, QPC binaries, calibration data, and outputs. This folder will be bind-mounted into the container so all work survives container restarts:

```bash
mkdir -p ~/qaic-workspace
cd ~/qaic-workspace
```

Run the `docker run` command from this folder so that `$PWD` resolves to your workspace.

---

## Locate Local SDK Tools

> **SDK tools are installed ONLY under `/opt/qti-aic/`.** They will never be found at any other path on the host — not in `/usr/`, `/usr/local/`, `/home/`, or anywhere else. Do NOT use `which`, `find`, or any other filesystem search to locate them. The only valid checks are the two `ls` commands below.

Check for tools at their only known host locations:

```bash
# Platform SDK (qaic-runner)
ls /opt/qti-aic/exec/qaic-runner 2>/dev/null || echo "Platform SDK not installed"
# Apps SDK (qaic-compile)
ls /opt/qti-aic/exec/qaic-compile 2>/dev/null || echo "Apps SDK not installed"
```

Both must be present for compile and run operations. If either is missing, check whether Docker is available before deciding how to proceed:

```bash
docker version 2>&1 | head -1
```

- If Docker is available, continue with the [container workflow](container-workflow.md) to either use the task's container or create a new one. Do not search elsewhere on the host for SDK tools.
- If Docker is not available, the bare-metal SDK must be installed. See the [Platform SDK installation reference](platform-sdk-installation.md). Without either the SDK on the host or Docker, no qaic tools are accessible.

**Host device access — verify before running any qaic tool:** `/dev/accel/` devices are `crw-rw---- root:qaic`. Only root or members of the `qaic` group can open them without sudo. Before running `qaic-runner` or `qaic-util` on the host, check whether the current user is in the group:

```bash
groups | grep qaic
```

If `qaic` does not appear in the output, stop and inform the user of the following before asking them to proceed:

- **What `qaic` group membership grants:** direct read/write access to all Cloud AI accelerator devices (`/dev/accel/accel*`). Any process running as that user can submit inference workloads or query device state without sudo.
- **Privilege scope:** membership does **not** grant sudo or any other elevated system privilege — it is scoped solely to accelerator device access.
- **Alternative (no group change needed):** run all qaic work inside Docker. The container runs as root but uses Linux namespaces and a reduced capability set — it has no access to the host filesystem or devices beyond what you explicitly pass (`-v $PWD:/workspace` and `--device /dev/accel/`). Following the mount guidelines in the [container workflow](container-workflow.md) keeps the exposure well-bounded. Docker root is not equivalent to host root, though it is not without risk: overly broad mounts or kernel vulnerabilities can narrow the gap, which is why the container workflow instructs you to mount only `$PWD` and never `/dev` or `/opt/qti-aic` directly.

If the user decides to proceed with host access, they must run the following in their own terminal and start a new login shell before continuing:

```bash
sudo adduser $(whoami) qaic
newgrp qaic   # applies group in current shell without full logout
```

Do not attempt to run any `qaic-runner` or `qaic-util` commands until the user confirms that `groups | grep qaic` passes. If the `qaic` group does not exist on the host, the Platform SDK has not been installed — see the [Platform SDK installation reference](platform-sdk-installation.md).
