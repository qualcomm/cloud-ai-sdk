#!/usr/bin/env bash

####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# model configs
MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT="\"A cinematic shot of a baby racoon wearing an intricate italian priest robe.\""
VAE_TYPE="vae"
UNET_TYPE="deep"
IMAGE_SIZE=1024
BLOCK_SIZE_DEEP=256
BLOCK_SIZE_SHALLOW=128
BATCH_SIZE=1
PRECISION=fp16,fp16,fp16,fp16

# onnx configs
GENERATE_ONNX=true
ONNX_TEXT_ENCODER=true
ONNX_UNET=true
ONNX_VAE=true

# compile configs
NUM_CORES=16
VAE_MOS=2
VAE_OLS=1
UNET_MOS_DEEP=2
UNET_OLS_DEEP=1
UNET_MOS_SHALLOW=1
UNET_OLS_SHALLOW=2
COMPILE_TEXT_ENCODER=true
COMPILE_UNET=true
COMPILE_VAE=true

# inference configs
RUN_ONLY=false
DEVICE=0
DEVICE_2=1
NUM_STEPS=20
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
sed -i 's/query_block_size = 128/query_block_size = 256/g' ./env_onnx/lib/python3.12/site-packages/diffusers/models/attention_processor.py

rm run.sh

RUNARGS=""
[ $# -gt 0 ] && RUNARGS="$(printf '%q ' "$@")"
scripts="python main.py \
--model-path $MODEL_PATH \
--prompt $PROMPT \
--unet-type $UNET_TYPE \
--vae-type $VAE_TYPE \
--batch-size $BATCH_SIZE \
--image-size $IMAGE_SIZE \
--block-size-deep $BLOCK_SIZE_DEEP \
--block-size-shallow $BLOCK_SIZE_SHALLOW \
--num-cores $NUM_CORES \
--vae-mos $VAE_MOS \
--vae-ols $VAE_OLS \
--unet-mos-deep $UNET_MOS_DEEP \
--unet-ols-deep $UNET_OLS_DEEP \
--unet-mos-shallow $UNET_MOS_SHALLOW \
--unet-ols-shallow $UNET_OLS_SHALLOW \
--device-id $DEVICE \
--device-id-2 $DEVICE_2 \
--num-steps $NUM_STEPS \
--num-warmup-iters $WARMUP_ITERS \
--num-repeat-iters $REPEAT_ITERS \
--precision $PRECISION \
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
