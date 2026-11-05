#!/bin/bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

exec /opt/vllm-env/bin/vllm serve "$model" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --additional-config '{
    "device_group": [0, 1, 2, 3],
    "override_qaic_config": {
      "qpc_path": "/model/qpc"
    }
  }'
