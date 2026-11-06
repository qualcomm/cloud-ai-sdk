####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# Stable Diffusion XL container environments (Cloud AI SDK 1.23)
#
# Two build targets, mirroring the two README sections:
#   export   - README part 2: ONNX generation (env_sdxl_onnxgen venv:
#              requirements.txt + patched diffusers for ONNX export)
#   pipeline - README part 4: end-to-end inference on AI100 (env_sdxl_infer
#              venv: requirements.txt + qaic wheel + pipeline-patched
#              diffusers)
#
# The venvs are named env_sdxl_onnxgen / env_sdxl_infer under /opt/recipe to
# match the README commands (the block-size sed in the export flow targets
# env_sdxl_onnxgen by path). The compose workflow bind-mounts only the output
# folders (hf_home/, HF_repo/, onnx_files/, qpc/, compile_logs/, images/), so
# results persist on the host while the venvs stay in the image layer.
#
# With utils/ on PATH, from this recipe directory:
#   mkdir -p hf_home HF_repo onnx_files qpc compile_logs images
#   qaic-container build
#   qaic-container run --rm export
#   qaic-container run --rm inference --prompt "<your prompt>" --negative_prompt "<your negative prompt>"
FROM ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0 AS recipe-base

WORKDIR /opt/recipe

COPY requirements.txt ./

# ---- export env (README part 2) ----
FROM recipe-base AS export
# moreutils provides `ts`, which compile_models.sh pipes qaic-compile output through.
RUN apt-get update \
 && apt-get install -y --no-install-recommends moreutils \
 && rm -rf /var/lib/apt/lists/*
# python3.12 ships inside /opt/venv_py312 in the SDK image; build a fresh env.
RUN /opt/venv_py312/bin/python -m venv env_sdxl_onnxgen
ENV VIRTUAL_ENV=/opt/recipe/env_sdxl_onnxgen
ENV PATH=/opt/recipe/env_sdxl_onnxgen/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt
COPY attention_patch.patch pipeline_patch.patch ./
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-dev \
 && cd diffusers-dev \
 && git apply --reject --whitespace=fix ../attention_patch.patch \
 && pip install --no-cache-dir . \
 && cd ..
ARG HOST_UID
ARG HOST_GID
ARG QAIC_GID
COPY --chown=${HOST_UID}:${HOST_GID} . .
RUN set -eux; \
    ! getent passwd recipe; \
    ! getent group recipe; \
    for value in "$HOST_UID" "$HOST_GID" "$QAIC_GID"; do \
      case "$value" in ''|*[!0-9]*) exit 1 ;; esac; \
      test "$value" -gt 0; \
    done; \
    if getent group qaic >/dev/null; then \
      groupmod -o -g "$QAIC_GID" qaic; \
    else \
      groupadd -o -g "$QAIC_GID" qaic; \
    fi; \
    groupadd -o -g "$HOST_GID" recipe; \
    useradd -o -u "$HOST_UID" -g recipe -G qaic \
      -m -d /home/recipe -s /bin/bash recipe; \
    chown "$HOST_UID:$HOST_GID" /opt/recipe /home/recipe \
    && chown "$HOST_UID:$HOST_GID" \
      env_sdxl_onnxgen/lib/python3.12/site-packages/diffusers/models \
      env_sdxl_onnxgen/lib/python3.12/site-packages/diffusers/models/attention_processor.py
ENV HOME=/home/recipe
USER recipe
CMD ["bash"]

# ---- pipeline/inference env (README part 4) ----
FROM recipe-base AS pipeline
RUN /opt/venv_py312/bin/python -m venv env_sdxl_infer
ENV VIRTUAL_ENV=/opt/recipe/env_sdxl_infer
ENV PATH=/opt/recipe/env_sdxl_infer/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
COPY attention_patch.patch pipeline_patch.patch ./
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-dev \
 && cd diffusers-dev \
 && git apply --reject --whitespace=fix ../pipeline_patch.patch \
 && pip install --no-cache-dir . \
 && cd ..
ARG HOST_UID
ARG HOST_GID
ARG QAIC_GID
COPY --chown=${HOST_UID}:${HOST_GID} . .
RUN set -eux; \
    ! getent passwd recipe; \
    ! getent group recipe; \
    for value in "$HOST_UID" "$HOST_GID" "$QAIC_GID"; do \
      case "$value" in ''|*[!0-9]*) exit 1 ;; esac; \
      test "$value" -gt 0; \
    done; \
    if getent group qaic >/dev/null; then \
      groupmod -o -g "$QAIC_GID" qaic; \
    else \
      groupadd -o -g "$QAIC_GID" qaic; \
    fi; \
    groupadd -o -g "$HOST_GID" recipe; \
    useradd -o -u "$HOST_UID" -g recipe -G qaic \
      -m -d /home/recipe -s /bin/bash recipe; \
    chown "$HOST_UID:$HOST_GID" /opt/recipe /home/recipe
ENV HOME=/home/recipe
USER recipe
CMD ["bash"]
