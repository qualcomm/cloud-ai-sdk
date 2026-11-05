---
name: cloud-ai-kubernetes
description: Use when adding Cloud AI workers to Kubernetes-compatible clusters, deploying the QAIC device plugin, or scheduling pods that request QAIC devices.
---

# Cloud AI Accelerators on Kubernetes

Set up QAIC workloads on Kubernetes by verifying each worker's device access,
registering capacity with the device plugin, and proving a non-root pod can
request a QAIC device. Run device commands on the worker; run `kubectl` commands from an administrative context.

## Add a QAIC Worker

1. From a `kubectl` administrative context, confirm it can label nodes and
   create a privileged DaemonSet. Enroll the node with the cluster's approved
   node-enrollment procedure. Do not invent a `kubeadm join`, `k3s agent`, or
   managed-cluster enrollment command. Verify Ready:

   ```bash
   kubectl wait --for=condition=Ready node/<worker> --timeout=5m
   kubectl get node <worker> -o wide
   ```

2. On each eligible worker, **REQUIRED SUB-SKILL:** Use
   `cloud-ai-env-setup` to install and verify Platform SDK driver and firmware.
   Confirm `lspci`, `/dev/accel/accel*`, and `qaic-util -q` report healthy
   devices. If a non-root operator needs node CLI access, add it to the group
   that owns the device node:

   ```bash
   QAIC_DEVICE_GROUP="$(stat -c '%G' /dev/accel/accel0)"
   sudo usermod -aG "$QAIC_DEVICE_GROUP" "$USER"
   ```

   Relog or run `newgrp "$QAIC_DEVICE_GROUP"` afterward. This does not grant
   pod access.

   On a VM- or container-backed node, configure PCIe passthrough or device mapping so the driver and `/dev/accel` are visible in the node runtime, then preflight there. For a Minikube-like container node, this belongs in the cluster node configuration, not a workload `hostPath`.

3. On each eligible worker, derive the device GID and kubelet root:

   ```bash
   ps -ef | grep '[k]ubelet.*--root-dir'
   QAIC_GID="$(stat -c '%g' /dev/accel/accel0)"
   QAIC_DEVICE_GROUP="$(stat -c '%G' /dev/accel/accel0)"
   QAIC_KUBELET_ROOT=/var/lib/kubelet # replace with the value after --root-dir, if shown
   test -S "$QAIC_KUBELET_ROOT/device-plugins/kubelet.sock"
   printf 'QAIC_GID=%s QAIC_DEVICE_GROUP=%s QAIC_KUBELET_ROOT=%s\n' \
     "$QAIC_GID" "$QAIC_DEVICE_GROUP" "$QAIC_KUBELET_ROOT"
   ```

   Start with the default above; replace with the `--root-dir` value, if shown,
   before the socket check. k3s normally has no `--root-dir` override, so
   `/var/lib/kubelet` is correct when the socket check passes. A single
   DaemonSet can cover only workers with the same `QAIC_GID` and
   `QAIC_KUBELET_ROOT`; use separate labels and manifests for different values.
   Use the numeric GID in workload manifests. Do not hardcode `995`. Use
   `supplementalGroups`, not `runAsGroup`.

   The driver may not create a host group named `qaic`. Numeric device access
   alone is insufficient for SDK images whose runtime calls
   `getgrnam("qaic")`: the smoke template creates an in-container `qaic` group
   with `QAIC_GID` through an `emptyDir` overlay. Do not use
   `getent group qaic` as discovery.

4. From the administrative context, label the compatible workers and render the
   plugin manifest:

   ```bash
   export QAIC_WORKER_LABEL=workload
   export QAIC_WORKER_VALUE=qaic-worker
   kubectl label node <worker> "$QAIC_WORKER_LABEL=$QAIC_WORKER_VALUE"
   ```

   In the shell that renders manifests, export every header variable from the
   template, including the common `QAIC_GID` and `QAIC_KUBELET_ROOT` from step
   3. `envsubst` reads only exported variables. Create
   `qaic-device-plugin.yaml` from
   [templates/qaic-device-plugin.yaml](templates/qaic-device-plugin.yaml).
   Select exactly one resource mode: generic (`0`, `None`), SKU (`1`, `None`),
   or fractional (`0`, `Half` or `Quarter`). Do not deploy a second device
   plugin if one already advertises `qualcomm.com/*` resources.

   Pod Security must permit its privileged, host-networked, host-path
   DaemonSet. Use an explicit plugin image tag; it need not match the Kubernetes version.
   Public images can lag Platform SDK releases, so validate the
   selected image on the target cluster. Create the plugin and smoke namespaces
   if needed, then render and client-validate:

   ```bash
   kubectl get namespace "$QAIC_PLUGIN_NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$QAIC_PLUGIN_NAMESPACE"
   kubectl get namespace "$QAIC_SMOKE_NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$QAIC_SMOKE_NAMESPACE"
   envsubst < templates/qaic-device-plugin.yaml > qaic-device-plugin.yaml
   if grep -n '\${' qaic-device-plugin.yaml; then
     echo "unresolved template variable" >&2
     exit 1
   fi
   kubectl apply --dry-run=client -f qaic-device-plugin.yaml
   kubectl apply -f qaic-device-plugin.yaml
   kubectl rollout status daemonset/"$QAIC_PLUGIN_NAME" -n "$QAIC_PLUGIN_NAMESPACE"
   kubectl get nodes -l "$QAIC_WORKER_LABEL=$QAIC_WORKER_VALUE"
   kubectl get pods -n "$QAIC_PLUGIN_NAMESPACE" -l "app=$QAIC_PLUGIN_LABEL" -o wide
   kubectl describe node <worker>
   ```

   Confirm plugin-pod node names and DaemonSet desired and ready counts cover every listed worker. Use the discovered `qualcomm.com/*` resource, not an assumed SKU.

5. After capacity appears, create `qaic-smoke-job.yaml` from
   [templates/qaic-smoke-job.yaml](templates/qaic-smoke-job.yaml). Set its QAIC
   resource and numeric GID. Discover the image's default non-root UID rather
   than guessing it:

   From a Docker-capable administrative machine that can pull the image, not
   necessarily from a worker node:

   ```bash
   docker run --rm --entrypoint id "$QAIC_SMOKE_IMAGE"
   ```

   Set `QAIC_SMOKE_UID` to the reported `uid`. Smoke namespace admission must
   allow the root-only init container; it creates the named `qaic` group file
   before `qaic-util` runs as that non-root UID. Render, run, and preserve the
   result:

   ```bash
   envsubst < templates/qaic-smoke-job.yaml > qaic-smoke-job.yaml
   if grep -n '\${' qaic-smoke-job.yaml; then
     echo "unresolved template variable" >&2
     exit 1
   fi
   kubectl apply --dry-run=client -f qaic-smoke-job.yaml
   kubectl apply -f qaic-smoke-job.yaml
   kubectl wait --for=condition=complete job/"$QAIC_SMOKE_NAME" -n "$QAIC_SMOKE_NAMESPACE" --timeout=5m
   kubectl logs job/"$QAIC_SMOKE_NAME" -n "$QAIC_SMOKE_NAMESPACE"
   ```

   `qaic-util -q` can emit non-fatal `hwmon` read warnings as a non-root user.
   Treat the Job as successful when it exits zero and reports ready QIDs. If
   the log says `qaic group does not exist` or reports insufficient privileges,
   verify the generated `/etc/group` overlay and `QAIC_GID`; changing only
   `supplementalGroups` does not satisfy the runtime's named-group lookup.
   A pending Job indicates a label, taint, image-pull, or resource-capacity
   problem.

For a workload-specific serving deployment after the worker smoke test, use `cloud-ai-vllm-serving`.
