####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# SDXL Turbo container environments (Cloud AI SDK 1.23)
#
# Two build targets, mirroring the two README sections:
#   export   - README part 1: ONNX generation (env_onnx venv: requirements.txt
#              + patched diffusers for ONNX export)
#   pipeline - README part 2: end-to-end inference on AI100 (env_pipeline venv:
#              requirements.txt + qaic wheel + pipeline-patched diffusers)
#
# The venvs are named env_onnx / env_pipeline under /opt/recipe to match the
# paths the run_config_*.sh scripts already use. The compose workflow bind-mounts
# only the output folders (cache/, onnx_files/, qpc/, compile_logs/, images/),
# so results persist on the host while the venvs stay in the image layer.
#
# Recommended entry point is the bundled compose launcher:
#   docker compose build
#   docker compose run --rm export
#   docker compose run --rm inference --prompt <prompt_in_quotes>
FROM ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0 AS recipe-base

WORKDIR /opt/recipe

COPY requirements.txt ./

# ---- export env (README part 1) ----
FROM recipe-base AS export
# python3.12 ships inside /opt/venv_py312 in the SDK image; build a fresh env.
RUN /opt/venv_py312/bin/python -m venv env_onnx
ENV VIRTUAL_ENV=/opt/recipe/env_onnx
ENV PATH=/opt/recipe/env_onnx/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt
COPY patches/ ./patches/
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-onnx \
 && cd diffusers-onnx \
 && git apply --reject --whitespace=fix ../patches/attention_patch.patch \
 && pip install --no-cache-dir . \
 && cd ..
COPY . .
CMD ["bash"]

# ---- pipeline/inference env (README part 2) ----
FROM recipe-base AS pipeline
RUN /opt/venv_py312/bin/python -m venv env_pipeline
ENV VIRTUAL_ENV=/opt/recipe/env_pipeline
ENV PATH=/opt/recipe/env_pipeline/bin:${PATH}
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
COPY patches/ ./patches/
RUN git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-pipeline \
 && cd diffusers-pipeline \
 && git apply --reject --whitespace=fix ../patches/pipeline_patch_separate.patch \
 && pip install --no-cache-dir . \
 && cd ..
COPY . .
CMD ["bash"]
