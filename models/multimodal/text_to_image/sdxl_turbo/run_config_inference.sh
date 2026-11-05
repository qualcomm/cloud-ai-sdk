#!/usr/bin/env bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

PYTHON=$1
echo $PYTHON

# model configs
MODEL_PATH="stabilityai/sdxl-turbo"
PROMPT="\"photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece\""
VAE_TYPE="vae"
IMAGE_SIZE=512
BLOCK_SIZE=256
BATCH_SIZE=1

# onnx configs
GENERATE_ONNX=false
ONNX_TEXT_ENCODER=false
ONNX_UNET=false
ONNX_VAE=false

# compile configs
NUM_CORES=16
VAE_MOS=2
VAE_OLS=1
UNET_MOS=2
UNET_OLS=1
COMPILE_TEXT_ENCODER=false
COMPILE_UNET=false
COMPILE_VAE=false

# inference configs
RUN_ONLY=true
DEVICE=0
NUM_STEPS=1
WARMUP_ITERS=3
REPEAT_ITERS=3

# mode
TOGETHER=false

if [ ${GENERATE_ONNX} == true ]
then
    GENERATE_ONNX_CMD="--generate-onnx"
else
    GENERATE_ONNX_CMD=""
fi

if [ ${ONNX_TEXT_ENCODER} == true ]
then
    ONNX_TEXT_ENCODER_CMD="--onnx-text-encoder"
else
    ONNX_TEXT_ENCODER_CMD=""
fi

if [ ${ONNX_UNET} == true ]
then
    ONNX_UNET_CMD="--onnx-unet"
else
    ONNX_UNET_CMD=""
fi

if [ ${ONNX_VAE} == true ]
then
    ONNX_VAE_CMD="--onnx-vae"
else
    ONNX_VAE_CMD=""
fi

if [ ${COMPILE_TEXT_ENCODER} == true ]
then
    COMPILE_TEXT_ENCODER_CMD="--compile-text-encoder"
else
    COMPILE_TEXT_ENCODER_CMD=""
fi

if [ ${COMPILE_UNET} == true ]
then
    COMPILE_UNET_CMD="--compile-unet"
else
    COMPILE_UNET_CMD=""
fi

if [ ${COMPILE_VAE} == true ]
then
    COMPILE_VAE_CMD="--compile-vae"
else
    COMPILE_VAE_CMD=""
fi

if [ ${RUN_ONLY} == true ]
then
    RUN_ONLY_CMD="--run-only"
else
    RUN_ONLY_CMD=""
fi

if [ ${TOGETHER} == true ]
then
    TOGETHER_CMD="--together"
else
    TOGETHER_CMD=""
fi

export HF_HOME="cache"

rm run.sh

RUNARGS=""
# $1 is the python interpreter; remaining args pass through to main.py
[ $# -gt 1 ] && RUNARGS="$(printf '%q ' "${@:2}")"
scripts="$PYTHON main.py \
--model-path $MODEL_PATH \
--prompt $PROMPT \
--vae-type $VAE_TYPE \
--batch-size $BATCH_SIZE \
--image-size $IMAGE_SIZE \
--block-size $BLOCK_SIZE \
--num-cores $NUM_CORES \
--vae-mos $VAE_MOS \
--vae-ols $VAE_OLS \
--unet-mos $UNET_MOS \
--unet-ols $UNET_OLS \
--device $DEVICE \
--num-steps $NUM_STEPS \
--num-warmup-iters $WARMUP_ITERS \
--num-repeat-iters $REPEAT_ITERS \
$ONNX_TEXT_ENCODER_CMD \
$ONNX_UNET_CMD \
$ONNX_VAE_CMD \
$COMPILE_TEXT_ENCODER_CMD \
$COMPILE_UNET_CMD \
$COMPILE_VAE_CMD \
$GENERATE_ONNX_CMD \
$RUN_ONLY_CMD \
$TOGETHER_CMD $RUNARGS"

echo $scripts >> run.sh

bash run.sh
