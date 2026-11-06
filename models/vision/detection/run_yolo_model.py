# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import subprocess
import argparse
from csv import DictReader
import torch
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import torchvision
import pandas as pd
from glob import glob
from ultralytics import YOLO
import json

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

# generates the yaml file as per requirement of the qaic_model_preparator.py
def gen_yaml(model_base_name, image_size):
    onnx_model ='./model/'+model_base_name+'.onnx'
    dynamic_info = []   
    model = onnx.load(onnx_model)
    for inp in model.graph.input:
        name = str(inp.name)
        shape = inp.type.tensor_type.shape.dim
        shape_params = [str(s.dim_param) for s in shape]
        dynamic_info.append([name, shape_params])    
    dynamic_info=str(dynamic_info).replace("'", '"')
    yaml_file=f'''

MODEL:
    INFO:
        DESCRIPTION: "YoloV{model_base_name[5]} Models from Ultralytics Repo."
        MODEL_TYPE: YOLOV{model_base_name[5]}
        MODEL_PATH: {onnx_model}
        INPUT_INFO: []
        EXPORT_TYPE: ONNX
        DYNAMIC_INFO:  [["{name}", ["batch_size", 3, {image_size}, {image_size}]]] 
        VALIDATE: False
        WORKSPACE: model_with_smart_nms
        VERBOSE: INFO #INFO, DEBUG, WARNING. TRACE

    PRE_POST_HANDLE:
        ANCHOR_BIN_FILE: None
        POST_PLUGIN: SMARTNMS
        PRE_PLUGIN: True
        NMS_PARAMS:
            MAX_OUTPUT_SIZE_PER_CLASS: None
            MAX_TOTAL_SIZE: None
            IOU_THRESHOLD: None
            SCORE_THRESHOLD: None
            CLIP_BOXES: None
            PAD_PER_CLASS: None

    '''

    ### Replace the PRE_POST_HANDLE in the above string with the following to use QDETECT instead of SMARTNMS. Refer to https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Inference-Workflow/Export-the-model/Prepare-the-model/ 
    # PRE_POST_HANDLE:
        # ANCHOR_BIN_FILE: None
        # POST_PLUGIN: QDETECT
        # PRE_PLUGIN: True
        # NMS_PARAMS:
            # MAX_OUTPUT_SIZE_PER_CLASS: 100
            # MAX_TOTAL_SIZE: 100
            # IOU_THRESHOLD: 0.65
            # SCORE_THRESHOLD: 0.25
            # CLIP_BOXES: False
            # PAD_PER_CLASS: False


    ### This conditional allows the code to run Yolov8 with model preparator tool
    if model_base_name[5] == '8':
        print("For Yolov8, the SmartNMS is not being enabled")
        model_type = 'UNKNOWN'
        yaml_file = yaml_file.replace(f'MODEL_TYPE: YOLOV{model_base_name[5]}', f'MODEL_TYPE: {model_type}')
        yaml_file = yaml_file.replace(f'POST_PLUGIN: SMARTNMS', f'POST_PLUGIN: NONE')
        print(yaml_file)

    with open(f'./model.yaml', 'w') as file: file.write(yaml_file)


# executes the command and writes it down in the command.txt. The first time, mode is 'w', then 'a' (append)
def execute(cmd_elements, write_to_file, mode):
    cmd_str = ' '.join(str(x) for x in cmd_elements)
    os.system(cmd_str)
    with open(write_to_file, mode) as file:
        file.write(cmd_str + "\n\n")


# checks device status and cores available
def check_device(DEVICE_ID, CORES, INSTANCES):

    QAIC_UTIL = subprocess.run(
        ["sudo", "/opt/qti-aic/tools/qaic-util", "-d", str(DEVICE_ID), "-q"],
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

    return


# generates a sample random input
def generate_random_data(model_path, BS, IS, INPUT_FOLDER):
    if os.path.isfile(model_path):
        model = onnx.load(model_path)
        print(f"ONNX model found in {model_path}", flush=True)
    else:
        raise FileNotFoundError(f"ONNX model {model_path} not found!")
        return
        
    ort_inputs = {}
    aic_batch_io = {"IO-files": [[]]}
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
    aic_batch_io["IO-files"][0].append({"path":f"input_img_{BS}x3x{IS}x{IS}.raw", "io-direction": "in", "elem-size": 4, "map-to": input_name, "dims": [BS, 3, IS, IS]})
    with open(f"{INPUT_FOLDER}aic_batch_io.json", "w") as f:
        json.dump(aic_batch_io, f, indent=1) 
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


def get_yolov5_model(yolo_name, image_size, opset, include_nms):
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    os.system(f'python yolov5/export.py --weights {yolo_name}.pt --include onnx --opset {opset}')
    os.makedirs('./model', exist_ok=True)
    os.system(f'mv -v {yolo_name}.onnx model/.')
    if include_nms: 
        os.system(f'rm -rf model_with_smart_nms')
        gen_yaml(yolo_name, image_size)
        os.system('echo y |  /opt/qti-aic/dev/python/qaic-env/bin/python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config model.yaml')          
    return 

def get_yolov7_model(yolo_name, image_size, include_nms):
    if not os.path.exists('yolov7'):
        os.system('git clone https://github.com/WongKinYiu/yolov7.git')
    os.system(f'wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{yolo_name}.pt')
    
    os.system(f'python yolov7/export.py --weights {yolo_name}.pt --img-size {image_size} {image_size}')    
    os.makedirs('./model', exist_ok=True)
    os.system(f'mv -v {yolo_name}.onnx model/.')        
    os.system(f'rm -rf {yolo_name}*')
    if include_nms: 
        os.system(f'rm -rf model_with_smart_nms')        
        gen_yaml(yolo_name, image_size) 
        os.system('echo y |  /opt/qti-aic/dev/python/qaic-env/bin/python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config model.yaml')
    return 

def get_yolov8_model(yolo_name, image_size, include_nms):
        # Load the model
    model = YOLO(f"{yolo_name}.pt")  # load an official or custom trained model

    # Export the model to ONNX format
    model.export(format="onnx")

    # Move the exported model to the 'model' directory
    os.makedirs('./model', exist_ok=True)
    os.system(f'mv -v {yolo_name}.onnx model/.')

    if include_nms:
        os.system(f'rm -rf model_with_smart_nms')
        gen_yaml(yolo_name, image_size)
        os.system('echo y |  /opt/qti-aic/dev/python/qaic-env/bin/python /opt/qti-aic/tools/qaic-pytools/qaic-model-preparator.py --config model.yaml')          
    return

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
        row = look_model_up(MODEL_NAME, OBJECTIVE, 'fp16', 'lut_yolo_models.csv')
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
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',   '1'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '640'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',       '14'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',    '1'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '2')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',    '10')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')
    elif OBJECTIVE == 'best-latency':
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',   '1'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '640'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',       '14'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',    '1'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '1')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',     '1')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')
    else:
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',   '1'))
        IS        = int(user_lut_default(args.image_size, row, 'IMAGE_SIZE', '640'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',        '4'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',    '3'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',          '1')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',           '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',     '2')
        EXTRA     =     user_lut_default(args.extra,      row, 'EXTRA',         '')  

    INCLUDE_NMS = args.include_nms


    TIME = args.time
    OPSET = args.opset

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
    if INCLUDE_NMS: MOTIF = MOTIF + "-NMS"
    MOTIF = MOTIF + f"-{OBJECTIVE}"
    MOTIF = MOTIF.replace(' ','')
    
    YOLO_VERSION = MODEL_NAME[5]

    if not RUN_ONLY:

        print("\n\n*************************************************************************************", flush=True)
        print(f"Downloading {MODEL_NAME} (OPSET {OPSET})", flush=True)
        print("*************************************************************************************\n\n", flush=True)
        if (YOLO_VERSION == '7'):
            get_yolov7_model(MODEL_NAME, IS, INCLUDE_NMS)
        elif (YOLO_VERSION == '5'):
            get_yolov5_model(MODEL_NAME, IS, OPSET, INCLUDE_NMS)
        elif (YOLO_VERSION == '8'):
            get_yolov8_model(MODEL_NAME, IS, INCLUDE_NMS)

    # For now, assuming only one onnx is generated in the directory ./model
    
    MODEL_FOLDER = './model_with_smart_nms' if INCLUDE_NMS else './model'
    
    for file in os.listdir(MODEL_FOLDER):
        if file.endswith(".onnx"):
            MODEL_BASE_NAME = file[:-5]

    # Fixing for fp16
    MODEL_BASE_NAME = fix_onnx_fp16(MODEL_FOLDER, MODEL_BASE_NAME)
    MODEL = f"{MODEL_FOLDER}/{MODEL_BASE_NAME}.onnx"

    execute ([""], f"commands-{MOTIF}.txt", 'w')
    
    # Generate a sample input
    ort_inputs, data_files, input_list_file = generate_random_data(MODEL, BS, IS, INPUT_FOLDER)

    if not RUN_ONLY:
        print("\n\n*************************************************************************************", flush=True)
        print(f"Compiling for BATCH_SIZE {BS} & IMAGE_SIZE {IS} for {CORES} AIC100_CORES", flush=True)
        print("*************************************************************************************\n\n", flush=True)

        # Compile for fp16
        execute(["rm", "-rf", f"compiled-bin-fp16-{MOTIF}"], f"commands-{MOTIF}.txt", 'a')
        os.makedirs(f"{OUTPUT_FOLDER}fp16-{MOTIF}", exist_ok=True)
        cmd_elements = ["/opt/qti-aic/exec/qaic-compile",
                        f"-m={MODEL}",
                        f"-onnx-define-symbol=batch_size,{BS}",
                        f"-aic-hw",
                        f"-aic-hw-version=ai100",
                        f"-aic-num-cores={CORES}",
                        f"-ols={OLS}",
                        f"-convert-to-fp16",
                         f"-aic-binary-dir=./compiled-bin-fp16-{MOTIF}",
                        f"-stats-batchsize={BS}",
                        f"{EXTRA}"
                        ]
        if MOS.replace(' ','') != '':
            cmd_elements.extend([f"-mos={MOS}"])
        execute(cmd_elements, f"commands-{MOTIF}.txt", 'a')

    # Triton export: generate repo
    if args.triton_export:
        import shutil
        from pathlib import Path

        repo = Path(args.triton_export) / "yolo"
        (repo / "1").mkdir(parents=True, exist_ok=True)

        # Copy compiled QPC
        qpc_path = f"./compiled-bin-fp16-{MOTIF}/programqpc.bin"
        if not os.path.exists(qpc_path):
            raise FileNotFoundError(f"QPC not found at {qpc_path}. Run compilation first.")

        shutil.copy(qpc_path, repo / "1" / "programqpc.bin")

        # Load ONNX to get input/output names for config
        onnx_model = onnx.load(MODEL)
        input_name = onnx_model.graph.input[0].name
        output_name = onnx_model.graph.output[0].name

        # Config with explicit input/output for QAIC backend (640x640 hardcoded)
        config = f'''name: "yolo"
backend: "qaic"
max_batch_size: 0

input {{
  name: "{input_name}"
  data_type: TYPE_FP32
  dims: [1, 3, 640, 640]
}}

output {{
  name: "{output_name}"
  data_type: TYPE_FP32
  dims: [1, 84, 8400]
}}
'''
        (repo / "config.pbtxt").write_text(config)

        print(f"Triton repo generated at {args.triton_export}")

    print("\n\n*************************************************************************************", flush=True)
    print(f"Running {INSTANCES} INSTANCES repeatedly for {TIME} seconds with OBJECTIVE {OBJECTIVE}", flush=True)
    print("*************************************************************************************\n\n", flush=True)

    # Run for fp16
    run_output_dir = f"{OUTPUT_FOLDER}fp16-{MOTIF}"
    os.makedirs(run_output_dir, exist_ok=True)
    cmd_elements = ["sudo", "/opt/qti-aic/exec/qaic-runner",
                    "-t", f"./compiled-bin-fp16-{MOTIF}",
                    "-a", f"{INSTANCES}",
                    "--time", f"{TIME}",
                    "--aic-profiling-type", "latency", "--aic-profiling-num-samples", "999999",
                    "--aic-profiling-out-dir", run_output_dir,
                    "-write-output-dir", run_output_dir,
                    "-S", f"{SET_SIZE}",
                    "-d", f"{DEVICE_ID}",
                    "--aic-batch-json-input", "./inputFiles/aic_batch_io.json"
                    ]
    execute(cmd_elements, f"commands-{MOTIF}.txt", 'a')

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
        description="Download, Compile, and Run YOLO models on randomly generated inputs for benchmarking")
    parser.add_argument(
        "--model-name", "-m",
        required=True,
        choices=["yolov5s", "yolov5m",
                 "yolov5l", "yolov5x", "yolov7", "yolov7-tiny",
                 "yolov7x", "yolov7-w6", "yolov7-e6",
                 "yolov7-d6", "yolov7-e6e",
                 "yolov8x","yolov8n","yolov8s","yolov8m","yolov8l"],
        help="Model name to download.",
    )
    parser.add_argument(
        "--objective", "-o", type=str,
        # choices=["best-latency", "best-throughput", "balanced"],
        help="Running for best-latency, best-throughput, or balanced",
    )
    parser.add_argument(
        "--opset",  type=check_positive,
        default=12,
        help="ONNX opset. Default <12>",
    )
    parser.add_argument(
        "--batch-size", "-b", type=check_positive,
        help="Sample input batch size.",
    )
    parser.add_argument(
        "--image-size", "-s", type=check_positive,
        help="Sample input image width/height. Default <640>.",
    )
    parser.add_argument(
        "--cores", "-c", type=int,
        choices=range(1, 17),
        help="Number of AIC100 cores to compile the model for. Default <2> ",
    )
    parser.add_argument(
        "--instances", "-i", type=int,
        choices=range(1, 17),
        help="Number of model instances to run on AIC100. Default <7>",
    )
    parser.add_argument(
        "--ols", type=int,
        choices=range(1, 15),
        help="Overlap split factor. Default <1>",
    )
    parser.add_argument(
        "--mos", type=str,
        help="Maximum output channel split. Default <1>",
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
                        help="Performs the inference only, without re-exporting and re-compiling the model"
    )    
    parser.add_argument('--include-nms',
                        action='store_true',
                        help="add the non maximum suppression to the object detectio model"
    )
    parser.add_argument('--triton-export', type=str,
                        help="Generate Triton model repository at PATH"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
