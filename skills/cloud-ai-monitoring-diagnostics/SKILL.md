---
name: cloud-ai-monitoring-diagnostics
description: Use when monitoring or diagnosing Cloud AI 100 devices, querying AICM or the low-level QMonitor gRPC diagnostics server (qaic-monitor-grpc-server), collecting sysfs/debugfs telemetry, investigating PCIe MSI allocation or interrupt-related performance, collecting firmware logs with qaic-log, tracing with qaic-trace-ctl, or configuring Prometheus/Grafana and alert policies.
---

# Cloud AI 100 – Monitoring and Diagnostics

## 7. AIC-Manager (AICM)

AICM provides REST telemetry, health monitoring, policies, time-series
metrics, and alerts.

Run it from `/opt/qti-aic/tools/aic-manager/`.

### Starting AICM

```bash
# Service mode
sudo bash scripts/start_aicm_agent.sh

# Direct mode
python aicm_agent.py --ip 127.0.0.1 --port 9000

# Stop
sudo bash scripts/stop_aicm_agent.sh
```

Run `python aicm_agent.py --help` for startup options.

Hash a password: `python scripts/hash_password.py <password>`

### API Access

- Swagger UI: `http://<IP>:<PORT>/docs`
- Redoc: `http://<IP>:<PORT>/redoc`
- All requests require Basic Auth credentials.

### AICMI (interactive CLI)

```bash
python aicmi.py
```

Commands: `health`, `group`, `policy`, and `list`. Use `help` or
`<command> -h` for subcommands and options.

### Key Metrics

AICM tracks device and firmware state, resources, clocks, power, temperature,
DDR/PCIe traffic, and correctable/uncorrectable RAS.

Docker image: `ghcr.io/qualcomm/cloud_ai_mgmt_aicm:1.23.2.0`

### Prometheus & Grafana Integration

AICM endpoints:

| Endpoint | Content |
|---|---|
| `/aicm/health_data/prometheus_health_metrics` | Health + PCIe metrics |
| `/aicm/health_data/prometheus_ras_metrics` | RAS error metrics |

Examples include a basic-auth `prometheus.yml`; use 5 seconds for a few cards
and as little as 250 ms at larger scale.

A Grafana dashboard in `/examples` covers NSP, SoC, power, and RAS.

## 8. vLLM Prometheus + Grafana (Disaggregated Serving)

Each vLLM worker exposes `/metrics`; the `qaic_disagg` proxy returns `404`.
Scrape `--encode-port` and `--decode-port`. Use `cloud-ai-vllm-serving` for
the disaggregated-serving layout.

### Autogenerate the stack from the launch command

Add `--build-grafana-json` once per EPD topology. It writes:

- `qaic_disagg/examples/prometheus_grafana/prometheus.yaml`: worker scrape targets
- `qaic_disagg/examples/prometheus_grafana/grafana_disagg.json`: matching dashboard

Targets default to `host.docker.internal:<vllm_worker_port>`. Regenerate only
after a topology change.

### Stack launch

```bash
cd qaic_disagg/examples/prometheus_grafana

export GRAFANA_ADMIN_USER=<username>
export GRAFANA_ADMIN_PASSWORD=<password>

mkdir -p prometheus-vllm-data prometheus-qaic-data grafana-data
sudo chown -R 65534:65534 prometheus-vllm-data prometheus-qaic-data
sudo chown -R 472:472 grafana-data

docker compose up
```

To switch EPD topology: `docker compose down`, regenerate the artifacts, then
`docker compose up`.

### Pairing with AICM (device-side metrics)

To add NSP utilization, DDR bandwidth, and RAS, start AICM
(`aicm_agent.py --ip 0.0.0.0`) and append its endpoints to `prometheus.yaml`.
Use `prometheus-vllm` (`:9090`) for application metrics and `prometheus-qaic`
(`:9091`) for AICM; Grafana exposes them as `DS_VLLM` and `DS_QAIC`.

### Key vLLM Metrics

| Metric | Type | Notes |
|---|---|---|
| `vllm:request_success_total` | counter | Disaggregated runs increment twice per e2e request (encode + decode). Halve for e2e throughput or filter by worker role. |
| `vllm:e2e_request_latency_seconds_bucket` | histogram | `histogram_quantile(0.5, sum(rate(...[5m])) by (le))` for p50 |
| `vllm:time_to_first_token_seconds_bucket` | histogram | TTFT distribution |
| `vllm:queue_wait_time_ms` | hist/gauge | Time the request sat in queue before prefill |
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | counter | Use 5–15 min rate for input/output tok/s |

### Dashboard import

At `http://<server>:3000`, import `grafana_disagg.json`. Default credentials
are `admin/admin`; `DS_VLLM` and `DS_QAIC` must exist.

### Health-checking the proxy

The proxy supports `GET /docs`, `GET /openapi.json`, and
`POST /v1/chat/completions`. Its `/health`, `/v1/models`, and `/metrics`
return `404`; probe a chat completion or worker `/health` ports.

## 9. QMonitor gRPC Server

QMonitor is the low-level device diagnostics server for device information,
RAS/PCIe, ULog/QDSS, debug controls, and Sysmon/PMU telemetry.

Use the client matching the operation. Check `--help` and source for
version-specific operations:

| Client or tool | Scope |
|---|---|
| `qaic-log` | Configure and read ULog/DIAG logs; route logs to QDSS when required. |
| `qaic-trace-ctl` | Configure, start, stop, and read QDSS traces. |
| `qaic-monitor-json` | Send JSON-formatted QMonitor requests, including device information. |
| `QAicMonitorClient.py` | Device info, RAS/PCIe, and event streams. |
| `QMonitorGRPCTestClient.py` | Connectivity, device info, and diagnostic loopback. |
| `QAicMonitorLLDB.py` | Use PVS/PTrace and debug/probe-agent operations. |
| `QAicMonitorSysmonExample.cpp` | Collect Sysmon PMU telemetry and asynchronous events. |

Examples are under `/opt/qti-aic/examples/python/` and
`/opt/qti-aic/examples/cpp/`; protobuf definitions are under
`/opt/qti-aic/dev/proto/`.

`qaic-log` connects to `127.0.0.1:62472` and has no host or port option, so it
and QMonitor must share a network namespace. `qaic-trace-ctl` accepts host and
port options.

Only one QMonitor can run at a time per physical accelerator host.

On a systemd host, `qaic-rt` post-install starts `qmonitor-proxy.service`.
Manage it with:

```bash
sudo /opt/qti-aic/scripts/qaic-monitor-service.sh start
sudo /opt/qti-aic/scripts/qaic-monitor-service.sh status
```

For a container deployment, run the standalone QMonitor image with host
networking and accelerator access:

```bash
docker run --rm -d --name qmonitor \
  --network host \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_mgmt_qmonitor:1.23.2.0
```

Docker bridge clients cannot reach another container's `127.0.0.1`. Run
`qaic-log` with host networking or in QMonitor's network namespace.

For a direct, single-container session, start it directly:

```bash
export LD_LIBRARY_PATH="/opt/qti-aic/dev/lib/$(uname -m)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/opt/qti-aic/tools/qaic-monitor-grpc-server -p 62472 -n 127.0.0.1 \
  >/tmp/qaic-monitor-grpc-server.log 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
/opt/qti-aic/tools/qaic-log -g -d 0 | tee /tmp/qmonitor-qaic-log-check.txt
if ! grep -Fq '"status": "SUCCESS"' /tmp/qmonitor-qaic-log-check.txt; then
  tail -n 100 /tmp/qaic-monitor-grpc-server.log
  exit 1
fi
```

If `"status": "SUCCESS"` is absent, inspect the server log and fix the deployment.

### QAIC sysfs telemetry

**REQUIRED REFERENCE:** Read
`references/qaic-sysfs-telemetry.md` when collecting or interpreting QAIC
sysfs or debugfs telemetry. It lists the supported nodes, units, DBC states,
MSI/MSI-X allocation, single-MSI fallback, BIOS prerequisites, and
interrupt-related performance diagnostics.

The QAIC driver exposes telemetry under the dynamic path
`/sys/class/accel/accelN/device/hwmon/hwmon*/`; discover `N` instead of
assuming a fixed accelerator or hwmon number.

```bash
accel=accel0
device="/sys/class/accel/$accel/device"
readlink -f "$device"
hwmon=$(find -L "$device/hwmon" -maxdepth 1 -type d -name 'hwmon*' -print -quit)
```

The reference defines the read-only `dbc*_state` values and their meaning.
Treat missing nodes, `EOPNOTSUPP`, `ENODEV`, and other read errors as
unavailable, not as zero readings. Do not write device power, TDP, or thermal
settings during routine diagnosis.

## 10. qaic-log — Firmware Log Collection

`qaic-log` requires QMonitor.

Run `/opt/qti-aic/tools/qaic-log -h` for all options.

```bash
# Set log level (d=debug, i=info, w=warning, e=error, f=fatal)
/opt/qti-aic/tools/qaic-log -l i -d 0

# Collect logs for 10 seconds
/opt/qti-aic/tools/qaic-log -r -d 0 -t 10

# Write to file
/opt/qti-aic/tools/qaic-log -r -d 0 -p /tmp/aic_logs.txt

# List available log sources on the device
/opt/qti-aic/tools/qaic-log -g -d 0

# Collect until interrupted
/opt/qti-aic/tools/qaic-log -r -d 0 -t 0
```

## 11. qaic-trace-ctl — Device Tracing

QMonitor is required. Configure → start → *run workload* → stop → read.

Run `/opt/qti-aic/tools/qaic-trace-ctl --help` for options.

```bash
# Configure
/opt/qti-aic/tools/qaic-trace-ctl --config trace_config.json --qid=0 configure

# Start
/opt/qti-aic/tools/qaic-trace-ctl --config trace_config.json --qid=0 start

# (run your workload here)

# Stop
/opt/qti-aic/tools/qaic-trace-ctl --config trace_config.json --qid=0 stop

# Read
/opt/qti-aic/tools/qaic-trace-ctl --config trace_config.json --qid=0 read
```

The configuration file is a JSON array of component objects. A sink alone
produces no event records. Use this software-event ETR configuration to trace
a workload:

```json
[
  {
    "component": "sw_event",
    "config": [
      {
        "subsystem": ["neural_proc_all"],
        "events": ["all"]
      }
    ]
  },
  {
    "component": "sink",
    "sink": "etr",
    "buffer_size": 32,
    "mode": "circular"
  }
]
```

Sink types are `etr` (device DRAM), `pcie` (host-side sink), and `etb` (the
internal 64 KiB buffer). `buffer_size` is in MiB. For `etr` or `etb`, run
`read` after `stop`; output is written under `/var/log/qti-aic/QID_X/`. For
`pcie`, stop after the workload; do not run `read`.

Use `qaic-log` for ULog collection. For hardware-event or ETM tracing, obtain
the matching SDK trace configuration recipe; QMonitor request JSON under
`test-data/json/qmonitor/qdss` is not a `qaic-trace-ctl` configuration file.

## Related Skills

- **REQUIRED SUB-SKILL:** Use `cloud-ai-env-setup` when the SDK, tools, or
  devices are not ready.
- Use `cloud-ai-kubernetes` when diagnosing a Kubernetes-deployed workload.
