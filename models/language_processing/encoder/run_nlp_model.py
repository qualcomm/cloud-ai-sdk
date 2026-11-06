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
import pandas as pd
from glob import glob
from pathlib import Path
import json
import sys

MODEL_ROOT='models'

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
def look_model_up(model_name, task, objective, precision, lut_path):
    with open(lut_path, mode='r', encoding='utf-8-sig') as file:
        reader = DictReader(file)
        for row in reader:
            if (row['MODEL_NAME'] == model_name) and (row['TASK'] == task) and (row['OBJECTIVE'] == objective) and (row['PRECISION'] == precision):
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
    if (STATUS != 'Ready'):
        raise TypeError(
            'The device is not ready. Please try, sudo sh -c "echo 1 > /sys/bus/mhi/devices/mhi0/soc_reset", or restart.')
    elif (CORES * INSTANCES) > NSP_FREE:
        raise TypeError(
            'Expected {} free AI cores but only {} are available on device {}'.format(CORES * INSTANCES, NSP_FREE, DEVICE_ID))
    elif (NSP_TOTAL < CORES*INSTANCES):
        raise TypeError(f"Please specify valid inputs for --cores and --instances. Make sure CORES*INSTANCES is less or equal than {NSP_TOTAL} (= # NSP cores of the installed Cloud AI device).")
    return

# generates a sample random input
def generate_random_data(model_path, BS, SL, INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    if os.path.isfile(model_path):
        model = onnx.load(model_path)
        print(f"ONNX model found in {model_path}", flush=True)
        with open ('model/config.json') as json_file:
            config = json.load(json_file)
            vocab_size = config['vocab_size']
    else:
        raise FileNotFoundError(f"ONNX model {model_path} not found!")
        return
    data_files = []
    ort_inputs = {}
    aic_batch_io = {"IO-files": [[]]}
    if len(model.graph.input) >= 1:
        input_ids = torch.randint(0, vocab_size, (BS, SL))
        input_ids_file = f"{INPUT_FOLDER}input_ids_{BS}x{SL}.raw"
        input_ids.numpy().astype(np.int64).tofile(input_ids_file)
        data_files.append(input_ids_file)
        ort_inputs['input_ids']=input_ids.numpy().astype(np.int64)
        aic_batch_io["IO-files"][0].append({"path":f"input_ids_{BS}x{SL}.raw", "io-direction": "in", "elem-size": 8, "map-to": "input_ids", "dims": [BS, SL]})
    if len(model.graph.input) >= 2:
        attention_mask = torch.ones((BS, SL))
        attention_mask_file = f"{INPUT_FOLDER}attention_mask_{BS}x{SL}.raw"
        attention_mask.numpy().astype(np.int64).tofile(attention_mask_file)
        data_files.append(attention_mask_file)
        ort_inputs['attention_mask']=attention_mask.numpy().astype(np.int64)
        aic_batch_io["IO-files"][0].append({"path":f"attention_mask_{BS}x{SL}.raw", "io-direction": "in", "elem-size": 8, "map-to": "attention_mask", "dims": [BS, SL]})
    if len(model.graph.input) >= 3:
        token_type_ids = torch.ones((BS, SL))
        token_type_ids_file = f"{INPUT_FOLDER}token_type_ids_{BS}x{SL}.raw"
        token_type_ids.numpy().astype(np.int64).tofile(token_type_ids_file)
        data_files.append(token_type_ids_file)
        ort_inputs['token_type_ids']=token_type_ids.numpy().astype(np.int64)
        aic_batch_io["IO-files"][0].append({"path":f"token_type_ids_{BS}x{SL}.raw", "io-direction": "in", "elem-size": 8, "map-to": "token_type_ids", "dims": [BS, SL]})
    input_list_file = f'./list_{BS}x{SL}.txt'
    with open(input_list_file, 'w') as fid:
        fid.write(','.join(data_files))
    with open(f"{INPUT_FOLDER}aic_batch_io.json", "w") as f:
        json.dump(aic_batch_io, f, indent=1)
    print(f"The random input samples are saved at {INPUT_FOLDER} and are addressed by {input_list_file}", flush=True)
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

def inference_complete_task(inference_set, inf_id):
        status, inf_handle = inference_set.getCompletedId(inf_id)
        inference_set.putCompleted(inf_handle)

def cosim(vector_a, vector_b):
    # Calculate cosine similarity using numpy
    vector_a = np.squeeze(np.asarray(vector_a))
    vector_b = np.squeeze(np.asarray(vector_b))

    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    return dot_product / (magnitude_a * magnitude_b)

# The main function
def main(args):

    RUN_ONLY = args.run_only
    API_RUN = args.api_run
    MODEL_NAME = args.model_name
    TASK = args.task
    OBJECTIVE = 'best-throughput' if args.objective is None else args.objective
    try:
        row = look_model_up(MODEL_NAME, TASK, OBJECTIVE, 'fp16', 'lut_nlp_models.csv')
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

    # If TASK not specified by user, look that up in lut.csv or set to 'default'
    if TASK is None:
        try:
            TASK = row['TASK']
        except:
            TASK = 'default'

    # Similarly, for the other arguments
    if OBJECTIVE == 'best-throughput':
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',        '8'))
        SL        = int(user_lut_default(args.seq_length, row, 'SEQUENCE_LENGTH', '128'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',             '2'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',         '7'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',               '')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',               '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',          '4')
        EXTRA     =     user_lut_default(args.extra, row,      'EXTRA',  '')
    elif OBJECTIVE == 'best-latency':
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',        '1'))
        SL        = int(user_lut_default(args.seq_length, row, 'SEQUENCE_LENGTH', '128'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',            '12'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',         '1'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',               '')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',               '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',          '1')
        EXTRA     =     user_lut_default(args.extra, row,      'EXTRA',  '')
    else:
        BS        = int(user_lut_default(args.batch_size, row, 'BATCH_SIZE',        '2'))
        SL        = int(user_lut_default(args.seq_length, row, 'SEQUENCE_LENGTH', '128'))
        CORES     = int(user_lut_default(args.cores,      row, 'CORES',             '6'))
        INSTANCES = int(user_lut_default(args.instances,  row, 'INSTANCES',         '2'))
        OLS       =     user_lut_default(args.ols,        row, 'OLS',               '')
        MOS       =     user_lut_default(args.mos,        row, 'MOS',               '')
        SET_SIZE  =     user_lut_default(args.set_size,   row, 'SET_SIZE',          '1')
        EXTRA     =     user_lut_default(args.extra, row,      'EXTRA',  '')

    TIME = args.time
    OPSET = args.opset

    # check device status and cores available
    check_device(DEVICE_ID, CORES, INSTANCES)

    path = os.path.join(MODEL_ROOT, MODEL_NAME)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    # create necessary folders for inputs, outputs, and compiled binaries
    INPUT_FOLDER = "./inputFiles/"
    OUTPUT_FOLDER = "./outputFiles/"
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    MOTIF = f"B{BS}-C{CORES}-A{INSTANCES}"
    if OLS != '': MOTIF = MOTIF + f"-OLS{OLS}"
    if MOS != '': MOTIF = MOTIF + f"-MOS{MOS}"
    MOTIF = MOTIF + f"-{OBJECTIVE}"
    MOTIF = MOTIF.replace(' ','')

    print('Model {}'.format(MODEL_NAME))

    onnx_files = glob(os.path.join('./model', '*.onnx'))

    if not RUN_ONLY and not onnx_files:
        print("\n\n*************************************************************************************", flush=True)
        print(f"Downloading {MODEL_NAME} (OPSET {OPSET}) for TASK {TASK} from HuggingFace", flush=True)
        print("*************************************************************************************\n\n", flush=True)

        # Generating the onnx model using Optimum - an extension of Transformers.
        execute(["optimum-cli", "export", "onnx", "--model", f"{MODEL_NAME}", "./model", "--cache_dir", "./cache", "--task", f"{TASK}", "--opset", f"{OPSET}", "--trust-remote-code"], f"commands-{MOTIF}.txt", 'w')

    # For now, assuming only one onnx is generated in the directory ./model
    onnx_files = glob(os.path.join('./model', '*.onnx'))
    assert onnx_files, 'Error generating ONNX'
    model_base_name = Path(onnx_files[0]).stem

    # Fixing for fp16
    model_base_name = fix_onnx_fp16('./model', model_base_name)
    MODEL = f"./model/{model_base_name}.onnx"

    # Generate a sample input - check if it has 3 inputs
    ort_inputs, data_files, input_list_file = generate_random_data(
        MODEL, BS, SL, INPUT_FOLDER)

    if not RUN_ONLY:
        print("\n\n*************************************************************************************", flush=True)
        print(f"Compiling for BATCH_SIZE {BS} & SEQUENCE_LENGTH {SL} for {CORES} AI_CORES", flush=True)
        print("*************************************************************************************\n\n", flush=True)

        # Compile for fp16
        execute(["rm", "-rf", f"compiled-bin-fp16-{MOTIF}"], f"commands-{MOTIF}.txt", 'a')
        os.makedirs(f"{OUTPUT_FOLDER}fp16-{MOTIF}", exist_ok=True)
        cmd_elements = ["/opt/qti-aic/exec/qaic-compile",
                        f"-m={MODEL}",
                        f"-onnx-define-symbol=batch_size,{BS}",
                        f"-onnx-define-symbol=sequence_length,{SL}",
                        f"-aic-hw",
                        f"-aic-hw-version=ai100",
                        f"-aic-num-cores={CORES}",
                        f"-convert-to-fp16",
                        f"-aic-binary-dir=./compiled-bin-fp16-{MOTIF}",
                        f"-stats-batchsize={BS}",
                        f"{EXTRA}"
                        ]
        if MOS != '':
            cmd_elements.extend([f"-mos={MOS}"])
        if OLS != '':
            cmd_elements.extend([f"-ols={OLS}"])
        execute(cmd_elements, f"commands-{MOTIF}.txt", 'a')

    print("\n\n*************************************************************************************", flush=True)
    print(f"Running {INSTANCES} INSTANCES repeatedly for {TIME} seconds with OBJECTIVE {OBJECTIVE}", flush=True)
    print("*************************************************************************************\n\n", flush=True)

    # Run for fp16
    if(API_RUN):
        import qaic
        print("\n\n-----------------------------------------", flush=True)
        print(f"Running qaic session for benchmark mode ", flush=True)
        print("------------------------------------------\n\n", flush=True)

        sess = qaic.Session(model_path=f"./compiled-bin-fp16-{MOTIF}/programqpc.bin", num_activations=int(INSTANCES), set_size=int(SET_SIZE),dev_id=int(DEVICE_ID))
        results=sess.run_benchmark(inf_time=int(TIME))
        print(f'bench_mark results:{results}')


        print("\n\n-----------------------------------------", flush=True)
        print(f"Running qaic session with worker threads and save output ", flush=True)
        print("------------------------------------------\n\n", flush=True)

        import concurrent.futures
        print(sess.model_input_shape_dict)
        print(sess.model_output_shape_dict)
        input_dict={}
        for (key,dim_info),data_file in zip(sess.model_input_shape_dict.items(),data_files):
            print(f' read from file {data_file} and it dim info is {dim_info} and key is {key} ')
            input_dict[key]=np.fromfile(data_file,dtype=np.int64).reshape(dim_info[0])

        def infer(input_data):
            output_dict = sess.run(input_data)
            return output_dict

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(infer, input_dict) for i  in range(10)]

        run_output_dir = f"{OUTPUT_FOLDER}fp16-{MOTIF}"
        os.makedirs(run_output_dir, exist_ok=True)
        index = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            output_dict = future.result()
            for output_items in sess.model_output_shape_dict:
                output_dict[output_items].tofile(f"./{run_output_dir}/{output_items}_{index}.bin")
                index +=1
        #Release all resources acquired by the session
        sess.reset()
        print(f"Finish qaic session with worker threads !!!!  ", flush=True)


        print("\n\n-----------------------------------------", flush=True)
        print(f"Running qaicrt session ", flush=True)
        print("------------------------------------------\n\n", flush=True)

        #Running inference with qaicrt inference set
        import sys
        import time
        import threading

        sys.path.append("/opt/qti-aic/dev/lib/x86_64/")
        import qaicrt
        dev_list = qaicrt.QIDList()
        dev_list.append(int(DEVICE_ID)) # Default to use the device 0
        context = qaicrt.Context(dev_list)

        qpc = qaicrt.Qpc(f"./compiled-bin-fp16-{MOTIF}/")
        buf_mappings = qpc.getBufferMappings()
        inferenceVector = qaicrt.InferenceVector(buf_mappings)

        #Set input data
        for buf_mapping, data_file in zip(buf_mappings, data_files):
            if buf_mapping.ioType==qaicrt.BufferIoTypeEnum.BUFFER_IO_TYPE_INPUT:
                img=np.fromfile(data_file,dtype=np.int64)
                buf_bytes=img.tobytes()
                inferenceVector.getVector()[buf_mapping.index]=qaicrt.QBuffer(buf_bytes)

        set_size = int(SET_SIZE)
        inference_set = qaicrt.InferenceSet(context, qpc,dev_list[0], set_size, int(INSTANCES))

        iterations=2000
        start_time = time.time()
        thread_list = []

        for inf_cnt in range(iterations):
            #User can update the inference vectore here.
            ##
            ##    inferenceVector.getVector()[buf_mapping.index]=qaicrt.QBuffer(buf_bytes)
            ##
            inference_set.submit(inferenceVector, inf_cnt)
            inf_complete_thread = threading.Thread(target=inference_complete_task, args=(inference_set, inf_cnt))
            thread_list.append(inf_complete_thread)
            inf_complete_thread.start()

        for thread in thread_list:
            thread.join()

        inference_set.waitForCompletion()

        end_time = time.time()

        inf_time= end_time - start_time
        qps = int(iterations * int(BS) / inf_time)
        print(f'Inference performance is  Inf/Sec : {qps}')

        print(f"Finish qaicrt session !!!!  ", flush=True)

    else:

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
        print(f"Comparing Cloud AI fp16 inference with onnxruntime fp32 inference", flush=True)
        print("*************************************************************************************\n\n", flush=True)

        output_names, ort_outputs = run_model_on_ort(MODEL, ort_inputs)
        # print (np.asarray(ort_outputs).flatten())

        for output_name, ort_output in zip(output_names, ort_outputs):
            aico16 = np.fromfile(f"{run_output_dir}/{output_name}-activation-0-inf-0.bin", np.float32)
            # print(ort_output)
            # print(aico16)
            ort_output_flat = np.asarray(ort_output).flatten()
            aico16_flat = np.asarray(aico16).flatten()
            print ("The first few output values from onnxruntime (fp32) and Cloud AI (fp16):")
            print(ort_output_flat[0:min(4, len(ort_output_flat))])
            print(aico16_flat[0:min(4, len(aico16_flat))])
            print('Cosine similarity for onnxruntime (fp32) and Cloud AI (fp16) outputs: {}'.format(cosim(ort_output_flat, aico16_flat)))


def check_positive(arg_in):
    try:
        if int(arg_in) <= 0:
            raise ValueError(f"Expected positive integer, received '{int(arg_in)}'")
    except ValueError:
        raise ValueError(f"Expected integer, received '{arg_in}'")
    return int(arg_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, Compile, and Run encoder-type NLP models on randomly generated inputs")
    parser.add_argument(
        "--model-name", "-m",
        required=True,
        help="Model name to download from Hugging Face. Try bert-base-cased for instance.",
    )
    parser.add_argument(
        "--task", "-t", type=str,
        choices=["default", "fill-mask", "question-answering", "text-classification",
                 "token-classification", "feature-extraction", "sentence-similarity"],
        help="Model task for encoder-type NLP models",
    )
    parser.add_argument(
        "--objective", "-o", type=str,
        choices=["best-latency", "best-throughput", "balanced"],
        help="Running for best-latency, best-throughput, or balanced",
    )
    parser.add_argument(
        "--opset",  type=check_positive,
        default=13,
        help="ONNX opset. Default <13>",
    )
    parser.add_argument(
        "--batch-size", "-b", type=check_positive,
        help="Sample input batch size. Default <1>.",
    )
    parser.add_argument(
        "--seq-length", "-s", type=check_positive,
        help="Sample input sequence length. Default <128>.",
    )
    parser.add_argument(
        "--cores", "-c", type=int,
        choices=range(1, 17),
        help="Number of AI cores to compile the model for. Default <2> ",
    )
    parser.add_argument(
        "--instances", "-i", type=int,
        choices=range(1, 17),
        help="Number of model instances to run on Cloud AI device. Default <7>",
    )
    parser.add_argument(
        "--ols", type=int,
        choices=range(1, 17),
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
        choices=range(0, 64),
        help="Cloud AI device ID. Default <0>",
    )
    parser.add_argument('--run-only', '-r',
                        action='store_true',
                        help="Performs the inference only, without re-exporting and re-compiling the model")

    parser.add_argument('--api-run', '-a',
                        action='store_true',
                        help="Performs api for inference. By default ,it is using qaic-runner to run infernece")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
