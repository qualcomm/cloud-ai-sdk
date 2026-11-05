#!/usr/bin/env bash

####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# model configs
MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"
PROMPT="\"$1\""
NEG_PROMPT="\"$2\""
GUIDANCE=7.0
VAE_TYPE="vae"
IMAGE_SIZE=1024
BLOCK_SIZE=64
BATCH_SIZE=1

# onnx configs
GENERATE_ONNX=false
ONNX_TEXT_ENCODER=false
ONNX_TRANSFORMER=false
ONNX_VAE=false

# compile configs
NUM_CORES=16
VAE_MOS=2
VAE_OLS=1
TRANSFORMER_MOS=1
TRANSFORMER_OLS=2
COMPILE_TEXT_ENCODER=false
COMPILE_TRANSFORMER=false
COMPILE_VAE=false

# inference configs
RUN_ONLY=true
TEXT_ENCODER_3=true
DEVICE=0
DEVICE2=1
NUM_STEPS=28
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

if [ ${ONNX_TRANSFORMER} == true ]
then
    ONNX_TRANSFORMER_CMD="--onnx-transformer"
else
    ONNX_TRANSFORMER_CMD=""
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

if [ ${COMPILE_TRANSFORMER} == true ]
then
    COMPILE_TRANSFORMER_CMD="--compile-transformer"
else
    COMPILE_TRANSFORMER_CMD=""
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

if [ ${TEXT_ENCODER_3} == true ]
then
    TEXT_ENCODER_3_CMD="--text-encoder-3"
else
    TEXT_ENCODER_3_CMD=""
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
# $1 and $2 are the prompt and negative prompt; remaining args pass through to main.py
[ $# -gt 2 ] && RUNARGS="$(printf '%q ' "${@:3}")"

scripts="python main.py \
--model-path $MODEL_PATH \
--prompt $PROMPT \
--neg_prompt $NEG_PROMPT \
--guidance $GUIDANCE \
--vae-type $VAE_TYPE \
--batch-size $BATCH_SIZE \
--image-size $IMAGE_SIZE \
--block-size $BLOCK_SIZE \
--num-cores $NUM_CORES \
--vae-mos $VAE_MOS \
--vae-ols $VAE_OLS \
--transformer-mos $TRANSFORMER_MOS \
--transformer-ols $TRANSFORMER_OLS \
--device-id $DEVICE \
--device-id2 $DEVICE2 \
--num-steps $NUM_STEPS \
--num-warmup-iters $WARMUP_ITERS \
--num-repeat-iters $REPEAT_ITERS \
$ONNX_TEXT_ENCODER_CMD \
$ONNX_TRANSFORMER_CMD \
$ONNX_VAE_CMD \
$COMPILE_TEXT_ENCODER_CMD \
$COMPILE_TRANSFORMER_CMD \
$COMPILE_VAE_CMD \
$GENERATE_ONNX_CMD \
$RUN_ONLY_CMD \
$TEXT_ENCODER_3_CMD \
$TOGETHER_CMD \
$RUNARGS"

echo $scripts >> run.sh

bash run.sh
