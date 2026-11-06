####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
from typing import Optional
import numpy as np
import torch
import transformers
import shutil
from pathlib import Path
import onnxruntime
import onnxsim
import onnx
import sys
from typing import Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusion3Pipeline
from onnx import external_data_helper, numpy_helper

def export_onnx(
    pt_model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    gen_models_path: str,
    model_base_name: str,
    opset: int,
) -> str:
    # Inspect the model's forward method arguments
    pt_model_code = pt_model.forward.__code__
    pt_input_names = pt_model_code.co_varnames[1 : pt_model_code.co_argcount]
    
    # Arrange the inputs in proper order to make tracing work properly
    pt_inputs = []
    input_names = []
    for input_name in pt_input_names:
        if input_name in inputs:
            if input_name == "past_key_values":
                if isinstance(inputs[input_name][0], torch.Tensor):
                    for i in range(len(inputs[input_name])):
                        input_names.append(f"past_key_value.{i}")
                else:
                    for i in range(len(inputs[input_name])):
                        input_names.append(f"past_key.{i}")
                        input_names.append(f"past_value.{i}")
            else:
                input_names.append(input_name)
            pt_inputs.append(inputs[input_name])
        else:
            pt_inputs.append(None)
    
    # Create dynamic axes dict for inputs that need to have dynamic input shapes
    seq_len_inputs = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "encoder_outputs",
    }
    
    seq_len_outputs = {
        "last_hidden_state"
    }

    decoder_seq_inputs = {"decoder_input_ids", "decoder_attention_mask"}
    dynamic_axes = {}
    for iname in input_names:
        if iname in seq_len_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "sequence_length"}
        elif iname.startswith("past_"):
            if len(inputs["past_key_values"][0][0].shape) == 4:
                # Normal attention (batch_size, num_heads, past_len, embed_dim)
                dynamic_axes[iname] = {0: "batch_size", 2: "past_sequence"}
            else:  # Multi-Query attention (batch_size, past_len, embed_dim)
                dynamic_axes[iname] = {0: "batch_size", 1: "past_sequence"}
    if (
        "past_key.0" in input_names or "past_key_value.0" in input_names
    ) and "attention_mask" in input_names:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "sequence+past_sequence"}
    for oname in output_names:
        if oname in seq_len_outputs:
            dynamic_axes[oname] = {0: "batch_size", 1: "sequence_length"}
    torch.onnx.export(
        pt_model,
        tuple(pt_inputs),
        f"{gen_models_path}/{model_base_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return model_base_name


def save_onnx(
    model: Union[onnx.ModelProto, str], gen_models_path: str, model_base_name: str
) -> str:
    if isinstance(model, str):
        model = onnx.load(f"{gen_models_path}/{model}.onnx")
    # Load the external tensors into the ModelProto, so the right size is calculated
    # and re-exported into right external tensor file
    onnx.load_external_data_for_model(model, gen_models_path)
    GB = 2**30
    if model.ByteSize() <= 2 * GB:
        onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")
    else:
        file_num = 0
        current_file_size = 0
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data") and ((tsize := sys.getsizeof(tensor.raw_data)) >= 1024):
                current_file_size += tsize
                if current_file_size > 10 * GB:
                    file_num += 1
                    current_file_size = tsize
                external_data_helper.set_external_data(
                    tensor, f"{model_base_name}_{file_num}.onnx.data"
                )
        onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")
    return model_base_name


def simplify_onnx(gen_models_path: str, model_base_name: str, **kwargs) -> str:
    model_path = Path(gen_models_path) / f"{model_base_name}.onnx"
    model = onnx.load(model_path, load_external_data=False)
    has_external_data = any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in external_data_helper._get_all_tensors(model)
    )
    simplified_name = f"{model_base_name}_simplified"

    if has_external_data:
        simplified_path = Path(gen_models_path) / f"{simplified_name}.onnx"
        check = onnxsim.onnx_simplifier.C.simplify_path(
            onnxsim.onnx_simplifier.C.ModelExecutor(),
            str(model_path),
            str(simplified_path),
            kwargs.get("skipped_optimizers", []),
            not kwargs.get("skip_constant_folding", False),
            not kwargs.get("skip_shape_inference", False),
            int(1.5 * 2**30),
            None,  # target_opset_version
            None,  # rewriter
        )
        if not check:
            raise RuntimeError("Failed verification of simplified model")
        return simplified_name

    simple_model, check = onnxsim.simplify(str(model_path), **kwargs)
    assert check, "Failed verification of simplified model"
    return save_onnx(simple_model, gen_models_path, simplified_name)


def fix_onnx_fp16(
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    ort_outputs: List[np.ndarray],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    finfo = np.finfo(np.float16)
    fp16_max = finfo.max
    fp16_min = finfo.min
    model = onnx.load(f"{gen_models_path}/{model_base_name}.onnx")
    fp16_fix = False
    
    for tensor in external_data_helper._get_all_tensors(model):
        nptensor = numpy_helper.to_array(tensor, gen_models_path)
        if nptensor.dtype == np.float32 and (
            np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)
        ):
            nptensor = np.clip(nptensor, fp16_min, fp16_max)
            new_tensor = numpy_helper.from_array(nptensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            fp16_fix = True

    if fp16_fix:
        # Save FP16 model
        print("Found constants out of FP16 range, clipped to FP16 range")
        model_base_name += "_fp16"
        save_onnx(model, gen_models_path, model_base_name)

        # Check if the FP16-fixed model can be used for FP32
        close_outputs = []
        _, ort_outputs_fixed = run_model_on_ort(
            f"{gen_models_path}/{model_base_name}.onnx", inputs, output_names
        )
        print("ONNXRT vs. ONNXRT fixed (MAD):")
        for oname, orto, ortof in zip(output_names, ort_outputs, ort_outputs_fixed):
            fix_diff = np.abs(orto - ortof).max()
            print(oname, fix_diff)
            close_outputs.append(fix_diff < 1e-5)

        # Replace FP32 model with FP16-fixed model
        if all(close_outputs):
            print("Using FP16 model for FP32")
            model_base_name = model_base_name[:-5]
            # Remove the FP32 onnx to avoid appending the external data
            os.remove(f"{gen_models_path}/{model_base_name}.onnx")
            i = 0
            while os.path.exists(f"{gen_models_path}/{model_base_name}_{i}.onnx.data"):
                os.remove(f"{gen_models_path}/{model_base_name}_{i}.onnx.data")
                i += 1
            # Save the fixed onnx
            save_onnx(model, gen_models_path, model_base_name)
            # Remove the temporary FP16 onnx
            os.remove(f"{gen_models_path}/{model_base_name}_fp16.onnx")
            i = 0
            while os.path.exists(f"{gen_models_path}/{model_base_name}_fp16_{i}.onnx.data"):
                os.remove(f"{gen_models_path}/{model_base_name}_fp16_{i}.onnx.data")
                i += 1
    return model_base_name


def run_model_on_ort(
    onnx_path: str,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
) -> Tuple[List[str], List[np.ndarray]]:
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_names = [x.name for x in ort_session.get_inputs()]
    ort_outputs = ort_session.run(
        output_names, {k: v.detach().numpy() for k, v in inputs.items() if k in input_names}
    )
    return input_names, ort_outputs


def export_t5(model_path, block_size, image_size, opset):
    # Load tokenizer and model
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, low_cpu_mem_usage=False)
    tokenizer = pipeline.tokenizer_3
    model = pipeline.text_encoder_3
    model.eval()

    wo_sfs = [61, 203, 398, 615, 845, 1190, 1402, 2242, 1875, 2393, 3845, 3213, 3922, 4429, 5020, 5623, 6439, 6206, 5165, 4593, 2802, 2618, 1891, 1419]
    assert len(wo_sfs) == 24
    with torch.no_grad():
        prev_sf = 1
        for i in range(len(model.encoder.block)):
            wosf = wo_sfs[i]
            model.encoder.block[i].layer[0].SelfAttention.o.weight *= 1 / wosf
            model.encoder.block[i].layer[0].scaling_factor *= prev_sf / wosf
            model.encoder.block[i].layer[1].DenseReluDense.wo.weight *= 1 / wosf
            prev_sf = wosf

    # Preprocess inputs
    #inputs = tokenizer("A sample prompt", return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
    inputs = tokenizer("A sample prompt", return_tensors="pt", padding="max_length")
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    if "attention_mask" in inputs:
        inputs.pop("attention_mask")
    enc_inputs = inputs.copy()

    # Encoder
    enc_pt_outputs = model(**enc_inputs)
    enc_output_names = list(enc_pt_outputs.keys())
    encoder_outputs = enc_pt_outputs.to_tuple()

    # Create dirs
    gen_models_path = f"onnx_files/text_encoder_3_{block_size}b_{image_size}i"
    Path(gen_models_path).mkdir(exist_ok=True)

    # Export and simplify ONNX model
    model_base_name = "model"
    fp32_model_name = export_onnx(
        model, enc_inputs, enc_output_names, gen_models_path, model_base_name, opset
    )
    fp32_model_name = simplify_onnx(gen_models_path, fp32_model_name)

    # Run ONNXRT inference
    input_names, ort_outputs = run_model_on_ort(
        f"{gen_models_path}/{fp32_model_name}.onnx", enc_inputs, enc_output_names
    )
    print("(Encoder) PyTorch vs. ONNXRT (MAD):")
    for oname, orto in zip(enc_output_names, ort_outputs):
        pto = enc_pt_outputs[oname]
        print(oname, np.abs(pto.detach().numpy() - orto).max())

    # Fix onnx for FP16
    fp16_model_name = fix_onnx_fp16(
        enc_inputs, enc_output_names, ort_outputs, gen_models_path, fp32_model_name
    )


if __name__ == "__main__":
    export_t5()
