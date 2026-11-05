####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#!/bin/bash

BINARY_FOLDER="./qpc/"
LOG_FOLDER="./compile_logs/"
BATCH_SIZE=1
BATCH_SIZE_2=$(expr 2 \* $BATCH_SIZE)
SEQ_LEN=77
LATENT_CHANNELS=4
LATENT_HEIGHT=128
LATENT_WIDTH=128
NUM_CORES=16
VAE_MOS=2
VAE_OLS=1
UNET_MOS_BS1=2
UNET_OLS_BS1=1
UNET_MOS_BS2=1
UNET_OLS_BS2=2

mkdir ${BINARY_FOLDER}
mkdir ${LOG_FOLDER}

########################################################################################################################

# 1. Compile the text encoder - self-generated
rm -rf ${BINARY_FOLDER}text_encoder
/opt/qti-aic/exec/qaic-compile \
    -aic-hw -aic-hw-version=ai100 -aic-perf-warnings -aic-perf-metrics \
    -convert-to-fp16 \
    -m=./onnx_files/text_encoder/model.onnx \
    -onnx-define-symbol=batch_size,${BATCH_SIZE} \
    -stats-batchsize=${BATCH_SIZE} \
    -onnx-define-symbol=sequence_length,${SEQ_LEN} \
    -aic-num-cores=${NUM_CORES} \
    -aic-binary-dir=${BINARY_FOLDER}text_encoder \
    2>&1 | ts > ${LOG_FOLDER}text_encoder.log &

########################################################################################################################

# 2. Compile the text encoder 2 - self-generated
rm -rf ${BINARY_FOLDER}text_encoder_2
/opt/qti-aic/exec/qaic-compile \
    -aic-hw -aic-hw-version=ai100 -aic-perf-warnings -aic-perf-metrics \
    -convert-to-fp16 \
    -m=./onnx_files/text_encoder_2/model.onnx \
    -onnx-define-symbol=batch_size,${BATCH_SIZE} \
    -stats-batchsize=${BATCH_SIZE} \
    -onnx-define-symbol=sequence_length,${SEQ_LEN} \
    -aic-num-cores=${NUM_CORES} \
    -aic-binary-dir=${BINARY_FOLDER}text_encoder_2 \
    2>&1 | ts > ${LOG_FOLDER}text_encoder_2.log &

########################################################################################################################

# 3a. Compile the UNet with batchsize=1, blocksize=256
rm -rf ${BINARY_FOLDER}unet-bs${BATCH_SIZE}
/opt/qti-aic/exec/qaic-compile \
    -aic-hw -aic-hw-version=ai100 -aic-perf-warnings -aic-perf-metrics \
    -convert-to-fp16 \
    -mos=${UNET_MOS_BS1} -ols=${UNET_OLS_BS1} \
    -m=./onnx_files/unet_bs1/unet/model.onnx \
    -onnx-define-symbol=batch_size,${BATCH_SIZE} \
    -stats-batchsize=${BATCH_SIZE} \
    -onnx-define-symbol=sequence_length,${SEQ_LEN} \
    -onnx-define-symbol=steps,1 \
    -onnx-define-symbol=num_channels,${LATENT_CHANNELS} \
    -onnx-define-symbol=height,${LATENT_HEIGHT} \
    -onnx-define-symbol=width,${LATENT_WIDTH} \
    -aic-num-cores=${NUM_CORES} \
    -aic-binary-dir=${BINARY_FOLDER}unet-bs${BATCH_SIZE} \
    2>&1 | ts > ${LOG_FOLDER}unet-bs${BATCH_SIZE}.log &


# 3b. Compile the UNet with batchsize=2, blocksize=128
rm -rf ${BINARY_FOLDER}unet-bs${BATCH_SIZE_2}
/opt/qti-aic/exec/qaic-compile \
    -aic-hw -aic-hw-version=ai100 -aic-perf-warnings -aic-perf-metrics \
    -convert-to-fp16 \
    -mos=${UNET_MOS_BS2} -ols=${UNET_OLS_BS2} \
    -m=./onnx_files/unet_bs2/unet/model.onnx \
    -onnx-define-symbol=batch_size,${BATCH_SIZE_2} \
    -stats-batchsize=${BATCH_SIZE_2} \
    -onnx-define-symbol=sequence_length,${SEQ_LEN} \
    -onnx-define-symbol=steps,1 \
    -onnx-define-symbol=num_channels,${LATENT_CHANNELS} \
    -onnx-define-symbol=height,${LATENT_HEIGHT} \
    -onnx-define-symbol=width,${LATENT_WIDTH} \
    -aic-num-cores=${NUM_CORES} \
    -aic-binary-dir=${BINARY_FOLDER}unet-bs${BATCH_SIZE_2} \
    2>&1 | ts > ${LOG_FOLDER}unet-bs${BATCH_SIZE_2}.log &


########################################################################################################################

# 4. Compile the VAE Decoder
rm -rf ${BINARY_FOLDER}vae_decoder
/opt/qti-aic/exec/qaic-compile \
    -aic-hw -aic-hw-version=ai100 -aic-perf-warnings -aic-perf-metrics \
    -convert-to-fp16 \
    -mos=${VAE_MOS} -ols=${VAE_OLS} \
    -m=./onnx_files/vae_decoder/model_fixed_128.onnx \
    -onnx-define-symbol=batch_size,${BATCH_SIZE} \
    -stats-batchsize=${BATCH_SIZE} \
    -onnx-define-symbol=num_channels_latent,${LATENT_CHANNELS} \
    -onnx-define-symbol=height_latent,${LATENT_HEIGHT} \
    -onnx-define-symbol=width_latent,${LATENT_WIDTH} \
    -aic-num-cores=${NUM_CORES} \
    -aic-enable-depth-first -aic-depth-first-mem=32 \
    -aic-binary-dir=${BINARY_FOLDER}vae_decoder \
    2>&1 | ts > ${LOG_FOLDER}vae_decoder.log &

########################################################################################################################

echo Waiting for qaic-compile processes to finish ...
wait

