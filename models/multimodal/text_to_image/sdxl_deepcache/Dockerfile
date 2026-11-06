####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# SDXL DeepCache container environments (Cloud AI SDK 1.23)
#
# Two build targets, mirroring the two README sections:
#   export   - README part 1: ONNX generation (env_onnx venv: requirements.txt
#              + patched diffusers for ONNX export + patched DeepCache UNet)
#   pipeline - README part 2: end-to-end inference on AI100 (env_pipeline venv:
#              requirements.txt + qaic wheel + clean diffusers + pipeline-
#              patched DeepCache)
#
# The venvs are named env_onnx / env_pipeline under /opt/recipe to match the
# paths the run_config_*.sh scripts already use. The compose workflow bind-mounts only
# the output folders (cache/, onnx_files/, qpc/, compile_logs/, images/), so
# results persist on the host while the venvs stay in the image layer.
#
# With utils/ on PATH, from this recipe directory:
#   mkdir -p cache onnx_files qpc compile_logs images
#   qaic-container build
#   qaic-container run --rm export
#   qaic-container run --rm inference --prompt "<your prompt>"
FROM ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0 AS recipe-base

WORKDIR /opt/recipe

COPY requirements.txt ./

# ---- export env (README part 1) ----
FROM recipe-base AS export
# moreutils provides `ts`, which utils.execute pipes qaic-compile output through.
RUN apt-get update \
 && apt-get install -y --no-install-recommends moreutils \
 && rm -rf /var/lib/apt/lists/*
# python3.12 ships inside /opt/venv_py312 in the SDK image; build a fresh env.
RUN /opt/venv_py312/bin/python -m venv env_onnx
ENV VIRTUAL_ENV=/opt/recipe/env_onnx
ENV PATH=/opt/recipe/env_onnx/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt
COPY patches/ ./patches/
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers_onnx \
 && cd diffusers_onnx \
 && git apply --reject --whitespace=fix ../patches/attention_patch.patch \
 && pip install --no-cache-dir . \
 && cd .. \
 && git clone https://github.com/horseee/DeepCache.git DeepCache \
 && cd DeepCache \
 && git checkout 1691a3ab22308fe06685e4ff151a9ab175be2dc4 \
 && git apply --reject --whitespace=fix ../patches/deepcache_unet.patch \
 && pip install --no-cache-dir .
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
      env_onnx/lib/python3.12/site-packages/DeepCache/sdxl \
      env_onnx/lib/python3.12/site-packages/DeepCache/sdxl/unet_2d_condition.py
ENV HOME=/home/recipe
USER recipe
CMD ["bash"]

# ---- pipeline/inference env (README part 2) ----
FROM recipe-base AS pipeline
RUN /opt/venv_py312/bin/python -m venv env_pipeline
ENV VIRTUAL_ENV=/opt/recipe/env_pipeline
ENV PATH=/opt/recipe/env_pipeline/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
COPY patches/ ./patches/
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers_pipeline \
 && cd diffusers_pipeline \
 && pip install --no-cache-dir . \
 && cd .. \
 && git clone https://github.com/horseee/DeepCache.git deepcache_pipeline \
 && cd deepcache_pipeline \
 && git checkout 1691a3ab22308fe06685e4ff151a9ab175be2dc4 \
 && git apply --reject --whitespace=fix ../patches/deepcache_pipeline.patch \
 && pip install --no-cache-dir .
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
