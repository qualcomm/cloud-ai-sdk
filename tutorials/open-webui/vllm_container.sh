# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

image=ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0
qpc_path=/path/to/qpc

chmod +x serve.sh

docker run -dit \
  --workdir /model \
  --name qaic-vllm \
  --network host \
  --mount type=bind,source=${PWD}/serve.sh,target=/model/serve.sh \
  --mount type=bind,source=${qpc_path},target=/model/qpc \
  -v qaic-vllm:/model/data \
  --env VLLM_QAIC_MAX_CPU_THREADS=8 \
  --env HF_HOME=/model/data/huggingface \
  --env QEFF_HOME=/model/data/qeff_models \
  --device=/dev/accel/ \
  --entrypoint=/model/serve.sh \
  ${image}
