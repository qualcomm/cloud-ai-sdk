##############################################################################
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023, Qualcomm Technologies, Inc. All Rights Reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
##############################################################################


import os
import subprocess
import argparse
from csv import DictReader
import torch
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import pandas as pd
from glob import glob
import time
import threading

import torchvision
from transformers import ResNetForImageClassification, ViTForImageClassification
from torchvision.models import list_models, get_model
# import timm

# torchvision_models = list_models(module=torchvision.models)
torchvision_models = ['resnet18','resnet34','resnet50','resnet101','resnet152','wide_resnet50_2','wide_resnet101_2','alexnet','resnext50_32x4d','resnext101_32x8d','resnext101_64x4d','efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','efficientnet_v2_l','efficientnet_v2_m','efficientnet_v2_s','vit_b_16','vit_b_32','inception_v3','densenet121','densenet161','densenet169','densenet201','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','mnasnet0_5','mnasnet0_75','mnasnet1_0','mnasnet1_3','shufflenet_v2_x0_5','shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0','squeezenet1_0','squeezenet1_1','mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small']
# torchvision_models = list_models()
# timm_pretrained_models = timm.list_models(pretrained=True) # This list is huge. Ignoring for now
timm_pretrained_models = []
aws_list = ['resnet-50', 'resnet-152', 'vit-base-patch16-224']
all_models  = list(set(torchvision_models+timm_pretrained_models+aws_list))
all_models.sort()


# computes the average or percentile for a pandas.Series object
def get_metric(series, method):
    if method == 'mean' or method == 'avg':
        return series.mean()
    elif method.endswith('pct'):
        prctile = int(method.replace('pct', ''))/100
        return series.quantile(prctile)
    return None

# computes the latency from the profiling latency text files, using the latency_method specified
def get_latency(latency_logs, latency_method):
    df = pd.concat([pd.read_csv(filename, skiprows=4)
                    for filename in latency_logs])
    col = df.columns[-3] # Execution Total Time in microseconds
    latency_ms = get_metric(df[col], latency_method)/1000.0
    return latency_ms    


# looks the model up in the lut.csv file and fetches its row entries
def look_model_up(model_name, objective, precision, lut_path):
    with open(lut_path, mode='r', encoding='utf-8-sig') as file:
        reader = DictReader(file)
        for row in reader:
            if (row['MODEL_NAME'] == model_name) and (row['OBJECTIVE'] == objective) and (row['PRECISION'] == precision):
                return row
        for row in reader:
            if (row['MODEL_NAME'] == model_name):
                return row


# executes the command and writes it down in the command.txt. The first time, mode is 'w', then 'a' (append)
def execute(cmd_elements, write_to_file, mode):
    cmd_str = ' '.join(str(x) for x in cmd_elements)
    os.system(cmd_str)
    with open(write_to_file, mode) as file:
        file.write(cmd_str + "\n\n")


# checks device status and cores available
def check_device(DEVICE_ID, CORES, INSTANCES):

    QAIC_UTIL = subprocess.run(
        ["/opt/qti-aic/tools/qaic-util", "-d", str(DEVICE_ID), "-q"],
        capture_output=True,
        text=True,
    ).stdout

    try:
        NSP_TOTAL = int(QAIC_UTIL.split("Nsp Total:")[1].split()[0])
    except:
        NSP_TOTAL = 14
    try:
        NSP_FREE = int(QAIC_UTIL.split("Nsp Free:")[1].split()[0])
    except:
        NSP_FREE = 14
    try:
        STATUS = QAIC_UTIL.split("Status:")[1].split()[0]
    except:
        STATUS = 'Ready'

    if (NSP_FREE < NSP_TOTAL or STATUS != 'Ready'):
        raise TypeError(
            'The device is not ready. Please try, sudo sh -c "echo 1 > /sys/bus/mhi/devices/mhi0/soc_reset", or restart.')

    if (NSP_TOTAL < CORES*INSTANCES):
        raise TypeError(f"Please specify valid inputs for --cores and --instance. Make sure CORES*INSTANCES is less or equal than {NSP_TOTAL} (= # NSP cores of the installed AIC100).")
        
    try:
        TEMPERATURE = float(QAIC_UTIL.split("(degree C):")[1].split()[0])
    except:
        TEMPERATURE = 100.

    return TEMPERATURE



# generates a sample random input
def generate_random_data(model_path, BS, IS, INPUT_FOLDER):
    if os.path.isfile(model_path):
        model = onnx.load(model_path)
        print(f"ONNX model found in {model_path}", flush=True)
    else:
        raise FileNotFoundError(f"ONNX model {model_path} not found!")
        return
        
    ort_inputs = {}
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    dummy_input = torch.randn(BS, 3, IS, IS)
    image_file = f"{INPUT_FOLDER}input_img_{BS}x3x{IS}x{IS}.raw"
    dummy_input.numpy().astype(np.float32).tofile(image_file)
    data_files = [image_file]
    input_list_file = f'./list_{BS}x3x{IS}x{IS}.txt'
    with open(input_list_file, 'w') as file:
        file.write(','.join(data_files))
    print(f"The random input samples are saved at {INPUT_FOLDER} and are addressed by {input_list_file}", flush=True)
    input_name = model.graph.input[0].name
    ort_inputs[input_name]=dummy_input.numpy().astype(np.float32)
    return ort_inputs, data_files, input_list_file


# clips the fp32 values to the range of fp16
def fix_onnx_fp16(gen_models_path,  model_base_name):
    print("Fixing ONNX to FP16", flush=True)
    finfo = np.finfo(np.float16)
    fp16_max = finfo.max
    fp16_min = finfo.min
    model = onnx.load(f"{gen_models_path}/{model_base_name}.onnx")
    fp16_fix = False
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        nptensor = numpy_helper.to_array(tensor, gen_models_path)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            nptensor = np.clip(nptensor, fp16_min, fp16_max)
            new_tensor = numpy_helper.from_array(nptensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            fp16_fix = True
    onnx.load_external_data_for_model(model, gen_models_path)
    size_gb = model.ByteSize() / 1073741824
    if fp16_fix:
        # Save FP16 model
        print("Found constants out of FP16 range, clipped to FP16 range", flush=True)
        # model_base_name += "_fixed_for_fp16"
        if size_gb <= 2:
            onnx.save(model,
                      f=f"{gen_models_path}/{model_base_name}.onnx")
        else:
            onnx.save(model,
                      f=f"{gen_models_path}/{model_base_name}.onnx",
                      save_as_external_data=True,
                      all_tensors_to_one_file=True,
                      location=f"{model_base_name}.onnx.data",
                      convert_attribute=True)
        print(f"Saving modified onnx file at {gen_models_path}/{model_base_name}.onnx", flush=True)
    return model_base_name


# This checks if arg_in is not specified by user, looks that up in the lut.csv or switches to 'default'
def user_lut_default (arg_in, row, entry, default):
    if arg_in == None:
        try: 
            arg_out = default if (row[entry]=='' or row[entry]==None) else row[entry]
        except: 
            arg_out = default
    else:
        arg_out = arg_in
    return arg_out

# Running model on onnxruntime for output comparison
def run_model_on_ort(onnx_path, ort_inputs):
    session = onnxruntime.InferenceSession(onnx_path)
    input_names  = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    ort_outputs = session.run(None, {k: v for k, v in ort_inputs.items() if k in input_names})
    return output_names, ort_outputs
    
# The main function
def main(args):
    RUN_ONLY = args.run_only
    MODEL_NAME = args.model_name
    OBJECTIVE = 'best-throughput' if args.objective is None else args.objective

    try:
        row = look_model_up(MODEL_NAME, OBJECTIVE, 'fp16', 'lut_cv_classifiers.csv')
    except:
        row = None

    try:
        OBJECTIVE = row['OBJECTIVE']
    except:
        OBJECTIVE = 'best-throughput' if args.objective is None else args.objective

    if args.device is None:
        print(f"Device id is not specified, choosing device 0", flush=True)
        DEVICE_ID = 0
    else:
        DEVICE_ID = args.device

    # Similarly, for the other arguments
    if OBJECTIVE == 'best-throughput':
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',  '14'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '224'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',        '1'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',   '14'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '2')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',    '10')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')
    elif OBJECTIVE == 'best-latency':
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',   '1'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '224'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',       '14'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',    '1'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '1')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',     '1')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')
    else:
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',   '1'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '224'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',        '4'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',    '3'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '1')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',     '2')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')  
        
    TIME      = args.time
    OPSET     = args.opset

    # check device status and cores available
    check_device(DEVICE_ID, CORES, INSTANCES)

    path = os.path.join('./', MODEL_NAME)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    # create necessary folders for inputs, outputs, and compiled binaries
    INPUT_FOLDER = "./inputFiles/"
    OUTPUT_FOLDER = "./outputFiles/"
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    MOTIF = f"B{BS}-C{CORES}-A{INSTANCES}-OLS{OLS}"
    if MOS != '': MOTIF = MOTIF + f"-MOS{MOS}"
    MOTIF = MOTIF + f"-{OBJECTIVE}"
    MOTIF = MOTIF.replace(' ','')
    
    if not RUN_ONLY:
        print("\n\n*************************************************************************************", flush=True)
        print(f"Downloading {MODEL_NAME} (OPSET {OPSET})", flush=True)
        print("*************************************************************************************\n\n", flush=True)
        # Downloading/Generating the onnx model

        if MODEL_NAME in torchvision_models:
            model = get_model (MODEL_NAME, weights="DEFAULT")
        elif MODEL_NAME in timm_pretrained_models:
            model = create_model (MODEL_NAME, pretrained=True)
        elif MODEL_NAME == 'resnet-50':
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif MODEL_NAME == 'resnet-152':
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")        
        elif MODEL_NAME == 'vit-base-patch16-224':
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        if not os.path.exists('./model'):
            os.makedirs('./model')

        # Export the PyTorch model to ONNX
        dummy_input = torch.randn(BS, 3, IS, IS)
        torch.onnx.export(model,                    # PyTorch model
                          dummy_input,              # Input tensor
                          './model/model.onnx',     # Output file
                          export_params=True,       # Export the model parameters
                          opset_version=OPSET,      # ONNX opset version
                          do_constant_folding=True, # Fold constant values for optimization
                          input_names=['input'],    # Input tensor names
                          output_names=['output'],  # Output tensor names
                          dynamic_axes={'input': {0: 'batch_size'},  # Dynamic axes
                                        'output': {0: 'batch_size'}})

    # For now, assuming only one onnx is generated in the directory ./model
    for file in os.listdir("./model"):
        if file.endswith(".onnx"):
            model_base_name = file[:-5]

    # Fixing for fp16
    model_base_name = fix_onnx_fp16('./model', model_base_name)
    MODEL = f"./model/{model_base_name}.onnx"

    # Generate a sample input 
    ort_inputs, data_files, input_list_file = generate_random_data(MODEL, BS, IS, INPUT_FOLDER)
    
    if not RUN_ONLY:
        print("\n\n*************************************************************************************", flush=True)
        print(f"Compiling for BATCH_SIZE {BS} & IMAGE_SIZE {IS} for {CORES} AIC100_CORES", flush=True)
        print("*************************************************************************************\n\n", flush=True)

        # Compile for fp16
        execute(["rm", "-rf", f"compiled-bin-fp16-{MOTIF}"], f"commands-{MOTIF}.txt", 'w')
        os.makedirs(f"{OUTPUT_FOLDER}fp16-{MOTIF}", exist_ok=True)
        cmd_elements = ["/opt/qti-aic/exec/qaic-exec",
                        f"-m={MODEL}",
                        f"-onnx-define-symbol=batch_size,{BS}",
                        f"-aic-hw",
                        f"-aic-hw-version=2.0",
                        f"-aic-num-cores={CORES}",
                        f"-ols={OLS}",
                        f"-convert-to-fp16",
                        f"-compile-only",
                        f"-aic-binary-dir=./compiled-bin-fp16-{MOTIF}",
                        f"-stats-batchsize={BS}",
                        f"{EXTRA}"
                        ]
        if MOS != '':
            cmd_elements.extend([f"-mos={MOS}"])
        execute(cmd_elements, f"commands-{MOTIF}.txt", 'a')

    print("\n\n*************************************************************************************", flush=True)
    print(f"Running {INSTANCES} INSTANCES for {TIME} seconds in a loop with OBJECTIVE {OBJECTIVE}", flush=True)
    print("*************************************************************************************\n\n", flush=True)

    # Run for fp16
    run_output_dir = f"{OUTPUT_FOLDER}fp16-{MOTIF}"
    os.makedirs(run_output_dir, exist_ok=True)

    try:
        while check_device(DEVICE_ID, CORES, INSTANCES) > 65.:
            print (f"Waiting for device to cool down! Current temperature {check_device(DEVICE_ID, CORES, INSTANCES)}C ")
            time.sleep(10) 
    except:
        None
    
    cmd_elements = ["/opt/qti-aic/exec/qaic-runner",
                    "-t", f"./compiled-bin-fp16-{MOTIF}",
                    "-a", f"{INSTANCES}",
                    "--time", f"{TIME}",
                    "--aic-profiling-type", "latency", "--aic-profiling-num-samples", "999999",
                    "--aic-profiling-out-dir", run_output_dir,
                    "-write-output-dir", run_output_dir,
                    "-S", f"{SET_SIZE}",
                    "-d", f"{DEVICE_ID}"
                    ]
    for data_file in data_files:
        cmd_elements.extend(["-i", data_file])
    execute(cmd_elements, f"commands-{MOTIF}.txt", 'a')
    
    # # computes the device avg power during runtime
    # def measure_power(TIME):
        # global avg_power 
        # SAMPLE_RATE = 3
        # avg_power = 0.
        # count = 0
        # time.sleep(TIME/(SAMPLE_RATE+2))
        # for i in range (SAMPLE_RATE):
            # time.sleep(TIME/(SAMPLE_RATE+2))  
            # try:
                # with open('/sys/class/hwmon/hwmon3/power1_input', 'r') as power:
                    # avg_power = avg_power + float(power.read())/1e6
                    # # print (float(power.read())/1e6)
                    # count = count + 1
            # except:
                # avg_power = 75.
        # avg_power = avg_power/count
    
    # t1 = threading.Thread(target=execute,       args=(cmd_elements, f"commands-{MOTIF}.txt", 'a',))
    # t2 = threading.Thread(target=measure_power, args=(TIME,))
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    # print(avg_power)
    
    latency_method = '95pct'
    config_folders = glob(f"{OUTPUT_FOLDER}fp16-{MOTIF}")
    latency_logs = glob(f"{config_folders[0]}/*latency.txt")
    LATENCY = get_latency(latency_logs, latency_method)
    print(f"Latency ({latency_method}) = {LATENCY:.3f} ms")


    print("\n\n*************************************************************************************", flush=True)
    print(f"Comparing AIC100 fp16 inference with onnxruntime fp32 inference", flush=True)
    print("*************************************************************************************\n\n", flush=True)

    
    output_names, ort_outputs = run_model_on_ort(MODEL, ort_inputs)
    # print (np.asarray(ort_outputs).flatten())

    for output_name, ort_output in zip(output_names, ort_outputs):
        aico16 = np.fromfile(f"{run_output_dir}/{output_name}-activation-0-inf-0.bin", np.float32)
        # print(ort_output)
        # print(aico16) 
        ort_output_flat = np.asarray(ort_output).flatten()
        aico16_flat = np.asarray(aico16).flatten()    
        print ("The first few output values from onnxruntime (fp32) and aic100 (fp16):")        
        print(ort_output_flat[0:min(4, len(ort_output_flat))])
        print(aico16_flat[0:min(4, len(aico16_flat))])
        diff = ort_output_flat - aico16_flat
        argmax = diff.argmax()
        print (f"The maximum difference is {np.abs(diff).max()} for values {ort_output_flat[argmax]} from onnxruntime (fp32) and {aico16_flat[argmax]} from aic100 (fp16)") 


def check_positive(arg_in):
    try:
        if int(arg_in) <= 0:
            raise ValueError(f"Expected positive integer, received '{int(arg_in)}'")
    except ValueError:
        raise ValueError(f"Expected integer, received '{arg_in}'")
    return int(arg_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, Compile, and Run vision models on randomly generated inputs")
    parser.add_argument(
        "--model-name", "-m",
        choices = all_models,
        required=True,
        help="Model name to download.",
    )
    parser.add_argument(
        "--objective", "-o", type=str,
        # choices=["best-latency", "best-throughput", "balanced"],
        help="Running for best-latency, best-throughput, or balanced",
    )
    parser.add_argument(
        "--opset",  type=check_positive,
        default=13,
        help="ONNX opset. Default <13>",
    )
    parser.add_argument(
        "--batch-size", "-b", type=check_positive,
        help="Sample input batch size.",
    )
    parser.add_argument(
        "--image-size", "-s", type=check_positive,
        help="Sample input image width/height. Default <224>.",
    )
    parser.add_argument(
        "--cores", "-c", type=int,
        choices=range(1, 15),
        help="Number of AIC100 cores to compile the model for. Default <2> ",
    )
    parser.add_argument(
        "--instances", "-i", type=int,
        choices=range(1, 15),
        help="Number of model instances to run on AIC100. Default <7>",
    )
    parser.add_argument(
        "--ols", type=int,
        choices=range(1, 15),
        help="Overlap split factor. Default <1>",
    )
    parser.add_argument(
        "--mos", type=str,
        help="Maximum output channel split.",
    )
    parser.add_argument(
        "--set-size", type=int,
        choices=range(1, 11),
        help="Set size. Default <10>",
    )
    parser.add_argument(
        "--extra", type=str,
        help="Extra compilation arguments.",
    )
    parser.add_argument(
        "--time", type=check_positive,
        default=20,
        help="Duration (in seconds) for which to submit inferences. Default <20>",
    )
    parser.add_argument(
        "--device",  "-d", type=int,
        choices=range(0, 8),
        help="AIC100 device ID. Default <0>",
    )
    parser.add_argument('--run-only', '-r',
                        action='store_true',
                        help="Performs the inference only, without re-exporting and re-compiling the model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
