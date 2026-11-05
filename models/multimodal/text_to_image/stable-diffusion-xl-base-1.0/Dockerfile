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
# Recommended entry point is the bundled compose launcher:
#   docker compose build
#   docker compose run --rm export
#   docker compose run --rm inference --prompt <prompt_in_quotes> --negative_prompt <prompt_in_quotes>
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
COPY . .
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
COPY . .
CMD ["bash"]
