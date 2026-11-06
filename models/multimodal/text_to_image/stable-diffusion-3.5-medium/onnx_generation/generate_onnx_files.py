####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
import time
import torch
import os
import shutil
from pathlib import Path

import onnx
from diffusers import StableDiffusion3Pipeline
from onnx import external_data_helper, numpy_helper
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from onnx_generation.t5_onnx_gen import export_t5
from onnx_generation.onnx_gen_utils import onnx_export, fix_onnx_fp16
from utils import scale_conv


def export_onnx_vae(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
        latent_size = None,
        latent_channels = 4,
        vae_type = 'vae',
    ):
    if vae_type == 'tiny':
        print("Generating tiny vae...")
        from diffusers import AutoencoderTiny
        pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3")

    vae_path = output_path / f"vae_decoder_{block_size}b_{image_size}i_{vae_type}" / "model.onnx"
    vae_decoder = pipeline.vae
    vae_decoder.forward = pipeline.vae.decode
    onnx_export(
        vae_decoder,
        model_args=(torch.randn(1, latent_channels, latent_size, latent_size)),
        output_path=vae_path,
        ordered_input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
        },
        opset=opset,
    )
    return True


def export_onnx_transformer(
        pipeline, 
        output_path, 
        opset,
        image_size = None,
        block_size = None,
        latent_channels = 4,
        latent_size = None,
        sequence_length = 77,
        joint_attention_dim = None,
        pooled_projection_dim = None,
    ):
    # MMDIT
    transformer_path = output_path / f"transformer_{block_size}b_{image_size}i" / "model.onnx"
    onnx_export(
        pipeline.transformer,
        model_args=(
            torch.randn(2, latent_channels, latent_size, latent_size),
            torch.randn(2, sequence_length, joint_attention_dim),
            torch.randn(2, pooled_projection_dim),
            torch.randint(0, 20, (2,), dtype=torch.int64), 
        ),
        output_path=transformer_path,
        ordered_input_names=["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"],
        output_names=["output"],  
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0:"steps"},
            "output": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
        },
        opset=opset,
        use_external_data_format=True,  
    )
    transformer_model_path = str(transformer_path.absolute().as_posix())
    transformer_dir = os.path.dirname(transformer_model_path)
    transformer = onnx.load(transformer_model_path)
    # clean up existing tensor files
    shutil.rmtree(transformer_dir)
    os.mkdir(transformer_dir)
    # collate external tensor files into one
    onnx.save_model(
        transformer,
        transformer_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return True # success


def export_onnx_text_encoder(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
    ):
    # TEXT ENCODER
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden_state_list = [f"hidden_states.{i}" for i in range(13)]
    onnx_export(
        pipeline.text_encoder,
        model_args=(text_input.input_ids.to(torch.int32), None, None, None, True),
        output_path=output_path / f"text_encoder_{block_size}b_{image_size}i" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"] + hidden_state_list,
        dynamic_axes={
            key: {0: "batch_size"} if key == "pooler_output" else {0: "batch_size", 1: "sequence_length"}
            for key in ["input_ids","last_hidden_state", "pooler_output"] + hidden_state_list
        },
        opset=opset,
    )
    return True # success


def export_onnx_text_encoder_2(
        pipeline, 
        output_path, 
        opset,
        block_size = None,
        image_size = None,
    ):        
    # TEXT ENCODER 2
    text_enc_2_path = output_path / f"text_encoder_2_{block_size}b_{image_size}i" / "model.onnx"
    text_input = pipeline.tokenizer_2(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden_state_list = [f"hidden_states.{i}" for i in range(33)]
    onnx_export(
        pipeline.text_encoder_2,
        model_args=(text_input.input_ids.to(torch.int64), None, None, None, True),
        output_path=text_enc_2_path,
        ordered_input_names=["input_ids"],
        output_names=["text_embeds", "last_hidden_state"] + hidden_state_list,
        dynamic_axes={
            key: {0: "batch_size", 1: "sequence_length"}
            for key in ["input_ids", "text_embeds", "last_hidden_state"] + hidden_state_list
        },
        opset=opset,
    )
    text_enc_2_path = str(text_enc_2_path.absolute().as_posix())
    text_enc_2_dir = os.path.dirname(text_enc_2_path)
    text_enc_2 = onnx.load(text_enc_2_path)
    # clean up existing tensor files
    shutil.rmtree(text_enc_2_dir)
    os.mkdir(text_enc_2_dir)
    # collate external tensor files into one
    onnx.save_model(
        text_enc_2,
        text_enc_2_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return True # success


@torch.no_grad()
def convert_models(
        output_path: str, 
        model_path: str, 
        opset: int = 16,
        image_size = None,
        block_size = 128,
        latent_size = None,
        seq_len = None,
        t5_seq_len = None,
        vae_type = 'vae',
        latent_channels = None,
        joint_attention_dim = None,
        pooled_projection_dim = None,
        export_transformer: bool = False,
        export_text_encoder: bool = False,
        export_text_encoder_2: bool = False,
        export_text_encoder_3: bool = False,
        export_vae: bool = False,
        together: bool = False,
    ):
    print(f"Model: {model_path}")
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, text_encoder_3=None, tokenizer_3=None)
    output_path = Path(output_path)
    
    if not together:
        if export_text_encoder:
            start_time = time.perf_counter()
            export_onnx_text_encoder(pipeline, output_path, opset, image_size=image_size, block_size=block_size)
            print(f"Completed Text Encoder ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting Text Encoder ONNX file")
            
        if export_text_encoder_2:
            start_time = time.perf_counter()
            export_onnx_text_encoder_2(pipeline, output_path, opset, image_size=image_size, block_size=block_size)
            print(f"Completed Text Encoder 2 ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting Text Encoder 2 ONNX file")
        
        if export_text_encoder_3:
            start_time = time.perf_counter()
            export_t5(model_path, block_size, image_size, opset)
            print(f"Completed Text Encoder 3 ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting Text Encoder 3 ONNX file")

        if export_transformer:
            start_time = time.perf_counter()
            export_onnx_transformer(
                    pipeline, 
                    output_path, 
                    opset,
                    image_size = image_size,
                    block_size = block_size,
                    latent_channels = latent_channels,
                    latent_size = latent_size,
                    sequence_length = seq_len + t5_seq_len,
                    joint_attention_dim = joint_attention_dim,
                    pooled_projection_dim = pooled_projection_dim,
            )
            print(f"Completed MMDiT ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting MMDiT ONNX file")

        if export_vae:
            start_time = time.perf_counter()
            export_onnx_vae(
                    pipeline, 
                    output_path, 
                    opset,
                    block_size = block_size,
                    image_size = image_size,
                    latent_size = latent_size,
                    latent_channels = latent_channels,
                    vae_type = vae_type,
            )
            print(f"Completed VAE ONNX export in {time.perf_counter() - start_time:.2f} seconds")
        else:
            print(f"Not exporting VAE ONNX file")
    else:
        raise NotImplementedError

