# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pprint import pprint
import os
import argparse
import subprocess
import json
import pandas as pd
import re
import time
import psutil
from pathlib import Path
from tqdm import tqdm
import glob
import hashlib
import requests
from urllib.parse import urlparse
import tarfile

import multiprocessing
from multiprocessing.managers import BaseManager

from datetime import datetime
from safe_tar import safe_extract_tar
from QEfficient import QEFFAutoModelForCausalLM
from huggingface_hub import login
from tempfile import NamedTemporaryFile
from transformers import AutoTokenizer

from QEfficient.utils.hash_utils import hash_dict_params
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils import load_hf_tokenizer

class QBenchDeviceList():
    def __init__(self):
        self.device_info = None
        self.num_cores = 16

    def scan(self, device_filter=None):
        if device_filter:
            device_filter = sorted(device_filter)

        print('device_filter {}'.format(device_filter))

        # skuType in qaicrt not accurate, so use qaic-util instead
        # Get all devices
        tmp_file = NamedTemporaryFile(delete=False)
        tmp_file_name = tmp_file.name
        tmp_file.close()

        command = '/opt/qti-aic/tools/qaic-util -q -j {}'.format(tmp_file_name)

        device_info = {'info': []}
        qaic_util = None

        result = subprocess.run(command.split(), stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            with open(tmp_file_name, 'r') as f:
                qaic_util = json.load(f)

            devices_available = []
            for info in qaic_util['device_info']:
                device_id = info['qid']
                if not device_filter or (device_filter and device_id in device_filter):
                    devices_available.append(device_id)
                    attr_list = ['qid', 'dev_status', 'sku_type', 'nsp_total', 'nsp_free']
                    attr_info = {}
                    for attr in attr_list:
                        attr_info[attr] = info[attr]
                    device_info['info'].append(attr_info)

            device_info['device_list'] = devices_available

        os.unlink(tmp_file_name)

        if not qaic_util:
            print('Could not retrieve device information')
            return None

        self.num_cores = device_info['info'][0]['nsp_total']
        self.device_info = device_info
        return device_info

    def device_group_to_string(self, num_devices):
        device_group = ''

        for device in range(num_devices):
            device_group += '{},'.format(self.device_info['device_list'][device])
        device_group = device_group[0:-1]

        return device_group

    def device_group_to_list(self, num_devices):
        device_group = []

        for device in range(num_devices):
            device_group.append(self.device_info['device_list'][device])

        return device_group

class QBenchModelPreparer:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.model_base_name = snapshot.split("/")[-1]

        self.model = QEFFAutoModelForCausalLM.from_pretrained(f"{snapshot}", continuous_batching=True)

        if 'XDG_CACHE_HOME' in os.environ:
            self.onnx_root = Path(os.environ['XDG_CACHE_HOME']) / 'qaic_bench'
        else:
            self.onnx_root = Path('~/.cache/qaic_bench').expanduser()

        self.replicate_kv_heads_script = Path(__file__).resolve().parent / 'replicate_kv_heads.py'
        if not os.path.exists(self.replicate_kv_heads_script):
            self.replicate_kv_heads_script = None

        self.onnx_export = None
        self.qpc_path = None

    @property
    def model_name(self) -> str:
        return self.snapshot.replace('/', '_')

    def export(self, full_batch_size=None, replicate_kv_heads=False):
        self.onnx_export = None

        if replicate_kv_heads and self.replicate_kv_heads_script:
            kv_heads = 2
            new_kv_heads = self.model.model.config.num_key_value_heads * kv_heads

            model_hash = hash_dict_params(self.model.hash_params)

            export_dir = self.onnx_root / '{}-{}'.format(self.model.model_name, model_hash)

            onnx_path = f"{self.model_base_name}-{new_kv_heads}kvheads-*"

            search_path = export_dir / onnx_path / '*.onnx'.format(self.model_name.replace('/', '_'))
            onnx_export = glob.glob(str(search_path))

            if not onnx_export:
                print('Replicating KV heads')

                if not os.path.exists(export_dir):
                    os.makedirs(export_dir, exist_ok=True)

                cmd = ['python3', self.replicate_kv_heads_script, '--model_name', self.snapshot, '--prompt', 'Hello, world!', '--repeat', '{}'.format(kv_heads)]
                if full_batch_size:
                    cmd.append('--full_batch_size')
                    cmd.append('{}'.format(full_batch_size))

                print(cmd)

                try:
                    result = subprocess.run(cmd, cwd=export_dir, capture_output=True, text=True, check=True)
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e.stderr}")

                onnx_export = glob.glob(str(search_path))

            self.onnx_export = onnx_export[0]
        else:
            self.onnx_export = self.model.export()

        print('ONNX export {}'.format(self.onnx_export))
        return self.onnx_export

    def compile(self, num_devices=1, num_cores=16, batch_size=1, context_len=4096):
        qpc_path = self.model.compile(
                onnx_path=self.onnx_export,
                mxint8_kv_cache=True,
                num_cores=num_cores,
                num_devices=num_devices,
                prefill_seq_len=128,
                ctx_len=context_len,
                aic_enable_depth_first=True,
                mxfp6_matmul=True,
                allow_mxint8_mdp_io=True,
                full_batch_size=batch_size
            )

        self.qpc_path = qpc_path

        return qpc_path

class QBenchVLLM:
    def __init__(self, vllm_root, devices):
        self.vllm_root = vllm_root
        self.devices = devices

        self.vllm_env = os.environ.copy()
        self.vllm_env['VLLM_QAIC_MAX_CPU_THREADS'] = '8'
        self.vllm_env['VLLM_TARGET_DEVICE'] = 'qaic'

    def get_metrics(self):
        return ['vLLM E2E (tok/s)', 'vLLM Normalized E2E (tok/s)']

    def benchmark_throughput(self, model, qpc, num_devices=1, batch_size=1, input_len=2048, output_len=2048):
        stats = {'vLLM E2E (tok/s)': 0, 'vLLM Normalized E2E (tok/s)': 0}

        if not os.path.isfile(os.path.join(qpc, 'programqpc.bin')):
            print('QPC not found, skipping')
            return stats

        cwd = os.getcwd()

        device_group = self.devices.device_group_to_string(num_devices)

        prompts = batch_size

        arg_data = {
            '--input-len': '{}'.format(input_len),
            '--output-len': '{}'.format(output_len),
            '--max-num-seqs': '{}'.format(batch_size),
            '--max-seq-len-to-capture': '128',
            '--kv-cache-dtype': 'mxint8',
            '--max-model-len': '{}'.format(input_len + output_len),
            '--num-prompts': '{}'.format(prompts),
            '--quantization': 'mxfp6',
            '--model': model,
            '--device-group': device_group,
            '--temperature': '0.0',
            '--seed': '20',
        }

        command = ['python3', 'benchmarks/benchmark_throughput.py']

        for arg in arg_data.keys():
            command.append(arg)
            command.append(arg_data[arg])

        print(' '.join(command))

        self.vllm_env['VLLM_QAIC_QPC_PATH'] = qpc
        result = subprocess.run(command, env=self.vllm_env, cwd=self.vllm_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(result.stdout)
            output_token_rate = 0.0

            for line in result.stdout.splitlines():
                line_new = line.strip()
                if "Processed prompts:" in line_new:
                    match = re.match(r'^Processed prompts.+output\: ([\d\.]+) toks.+', line_new)
                    if match:
                        output_token_rate = float(match.group(1))

            stats['vLLM E2E (tok/s)'] = output_token_rate
            stats['vLLM Normalized E2E (tok/s)'] = round(output_token_rate / batch_size, 2)
        else:
            print(result.stdout)

        return stats

class QBenchQEff:
    def __init__(self, devices):
        self.devices = devices

    def get_metrics(self):
        return ['QEff TTFT (seconds)', 'QEff E2E (tok/s)', 'QEff Normalized E2E (tok/s)']

    def benchmark_throughput_2(self, model, qpc, num_devices=1, batch_size=1, input_len=2048, output_len=2048):
        stats = {}
        metrics = self.get_metrics()
        for name in metrics:
            stats[name] = 0

        if not os.path.isfile(os.path.join(qpc, 'programqpc.bin')):
            print('QPC not found, skipping')
            return stats

        tokenizer = load_hf_tokenizer(
                pretrained_model_name_or_path=(model)
            )

        device_group = self.devices.device_group_to_list(num_devices)

        exec_info = cloud_ai_100_exec_kv(
            tokenizer=tokenizer,
            qpc_path=qpc,
            device_id=device_group,
            prompt='What is the capital of France?',
            generation_len=output_len,
            stream=False
        )

        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model)
        #print(exec_info)

        return stats

    def benchmark_throughput(self, model, qpc, num_devices=1, batch_size=1, input_len=2048, output_len=2048):
        stats = {}
        metrics = self.get_metrics()
        for name in metrics:
            stats[name] = 0

        if not os.path.isfile(os.path.join(qpc, 'programqpc.bin')):
            print('QPC not found, skipping')
            return stats

        random_prompt = RandomPrompt(model, max_len=input_len)
        #print(vars(random_prompt))

        device_group = self.devices.device_group_to_string(num_devices)

        arg_data = {
            '--model-name': model,
            '--qpc_path': qpc,
            '--device-group': device_group,
            '--prompt': random_prompt.prompt,
            '--generation-len': '{}'.format(output_len),
        }

        command = ['python3', '-m', 'QEfficient.cloud.execute']
        command_print = ['python3', '-m', 'QEfficient.cloud.execute']

        for arg in arg_data.keys():
            command.append(arg)
            command_print.append(arg)

            command.append(arg_data[arg])
            if arg == '--prompt':
                command_print.append(arg_data[arg][:8] + '...')
            else:
                command_print.append(arg_data[arg])

        print(' '.join(command_print))

        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            #print(result.stdout)
            ttft = 0.0
            output_token_rate = 0.0

            for line in result.stdout.splitlines()[-6:]:
                line_new = line.strip()
                #print(line_new)

                match = re.match(r'^Average Prefill time a.k.a TTFT is\= ([\d\.]+) sec', line_new)
                if match:
                    ttft = float(match.group(1))

                match = re.match(r'^Decode is\= ([\d\.]+) tokens\/sec', line_new)
                if match:
                    output_token_rate = float(match.group(1))

            stats['QEff TTFT (seconds)'] = ttft
            stats['QEff E2E (tok/s)'] = output_token_rate
            stats['QEff Normalized E2E (tok/s)'] = round(output_token_rate / batch_size, 2)
        else:
            print('Return code {}'.format(result.returncode))
            print(result.stdout)

        return stats

def qpc_compile(config):
    # Prepare QPC model binary
    preparer = config['preparer']
    context_len = config['prompt_len'] + config['generation_len']
    qpc = preparer.compile(config['devices'],
                           config['cores'],
                           config['batch_size'],
                           context_len)
    return str(qpc)

class RandomPrompt():
    def __init__(self, model, max_len=2048):
        tokenizer = AutoTokenizer.from_pretrained(model)
        # Random prompt generated from vLLM RandomDataset
        with open('random_prompt.txt', 'r', encoding='utf-8') as f:
            prompt = f.read()

        prompt_ids = tokenizer(prompt).input_ids
        prompt_len = len(prompt_ids)

        # Leave room for prompt tokens
        max_len -= tokenizer.num_special_tokens_to_add()

        if prompt_len < max_len:
            # lazy expand
            expand_by = int((max_len - prompt_len) / prompt_len) + 1
            prompt_ids *= expand_by
            prompt_len = len(prompt_ids)

        if prompt_len > max_len:
            prompt_ids = prompt_ids[:max_len]
            prompt_len = max_len
            prompt = tokenizer.decode(prompt_ids)

        self.prompt = prompt
        self.prompt_len = prompt_len

def qpc_download(url):
    if 'XDG_CACHE_HOME' in os.environ:
        cache_root = Path(os.environ['XDG_CACHE_HOME']) / 'qaic_bench'
    else:
        cache_root = Path('~/.cache/qaic_bench').expanduser()

    dir_hash = hashlib.sha256()
    dir_hash.update(url.encode('utf-8'))
    dir_hash = dir_hash.hexdigest()[:16]

    parsed_url = urlparse(url)

    qpc_root = cache_root / 'qpc_{}'.format(dir_hash)

    if not os.path.exists(qpc_root):
        os.makedirs(qpc_root)

    destination = qpc_root / Path(parsed_url.path).name

    if not os.path.exists(destination):
        try:
            print('Downloading {} to {}'.format(url, destination))
            # Send a GET request to the URL
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status() # Check if the download was successful

            qpc_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 # 1 KB chunks

            with tqdm(
                desc=Path(parsed_url.path).name,
                total=qpc_size if qpc_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                # Open a local file in binary write mode and save the content in chunks
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            if qpc_size != 0 and progress_bar.n != qpc_size:
                raise RuntimeError("Incomplete download")

            print("Download complete")
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return None

    tar_root = destination.with_suffix('')

    if not os.path.exists(tar_root) and not os.path.exists(qpc_root / 'extracted'):
        try:
            if not os.path.exists(tar_root):
                os.makedirs(tar_root)
            print('Extracting {}'.format(destination))
            with tarfile.open(destination, "r:*") as tar:
                safe_extract_tar(tar, tar_root)
            print('Files extracted')
            with open(qpc_root / 'extracted', 'w') as f:
                pass
        except tarfile.TarError as e:
            print(f"An error occurred during extraction: {e}")
            return None

    qpc_dir = glob.glob(os.path.join(str(tar_root), '**', 'programqpc.bin'), recursive=True)
    if qpc_dir:
        return str(Path(qpc_dir[0]).parent)

    return None

def qbench_benchmark(devices, config, run_vllm=False, compile_only=False):

    device_list = None

    if devices:
        device_list = list(map(int, devices.split(',')))

    qaic_devices = QBenchDeviceList()

    qaic_devices.scan(device_list)
    pprint(qaic_devices.device_info)

    with open(config, 'r') as fp:
        bench_configs = json.load(fp)

    qeff_bench = QBenchQEff(qaic_devices)

    if run_vllm:
        vllm_bench = QBenchVLLM(bench_configs['vllm_root'], qaic_devices)

    if compile_only:
        print('Generating QPCs only, skipping benchmarks')

    BaseManager.register('QBenchModelPreparer', QBenchModelPreparer)

    for model in bench_configs['models']:
        with BaseManager() as manager:
            preparer = None
            preparer_kv = None

            # Generate all QPCs in one pass

            qpc_list = []

            for config in model['configs']:
                if not 'cores' in config:
                    config['cores'] = qaic_devices.num_cores

                if 'qpc' in config:
                    if config['qpc'].startswith(tuple(['http://', 'https://'])):
                        config['qpc'] = qpc_download(config['qpc'])
                        print('Updated QPC: {}'.format(config['qpc']))
                else:
                    if config['devices'] == 16:
                        if not preparer_kv:
                            preparer_kv = manager.QBenchModelPreparer(model['model'])
                            preparer_kv.export(replicate_kv_heads=True, full_batch_size=config['batch_size'])

                        config['preparer'] = preparer_kv
                    else:
                        if not preparer:
                            preparer = manager.QBenchModelPreparer(model['model'])
                            preparer.export()

                        config['preparer'] = preparer

                    qpc_list.append(config)

            if qpc_list:
                print('qpc_list {}'.format(qpc_list))
                with multiprocessing.Pool(processes=6) as pool:
                    try:
                        qpc_results = list(tqdm(pool.imap(qpc_compile, qpc_list), total=len(qpc_list), desc='Generating QPCs'))
                    except KeyboardInterrupt:
                        pool.terminate()

                        # Clean up qaic-exec processes
                        for child in psutil.Process().children(recursive=True):
                            if child.name() == 'qaic-exec':
                                print('Stopping qaic-exec pid {}'.format(child.pid))
                                child.terminate()

                        exit(1)
                    else:
                        pool.close()
                    pool.join()

                for idx, config in enumerate(qpc_list):
                    config['qpc'] = qpc_results[idx]
                    print('Updated qpc {} {}'.format(model['name'], config))


        df = pd.DataFrame() # Empty dataframe

        if not compile_only:
            all_results = {
                'Model': [],
                'Devices': [],
                'Cores': [],
                'PL': [],
                'GL': [],
                'BS': [],
            }

            stats = qeff_bench.get_metrics()
            for name in stats:
                all_results[name] = []

            if run_vllm:
                stats = vllm_bench.get_metrics()
                for name in stats:
                    all_results[name] = []

            for config in model['configs']:
                print('Benchmarking {} {}'.format(model['name'], config))

                all_results['Model'].append(model['name'])
                all_results['Devices'].append(config['devices'])
                all_results['Cores'].append(config['cores'])
                all_results['PL'].append(config['prompt_len'])
                all_results['GL'].append(config['generation_len'])
                all_results['BS'].append(config['batch_size'])

                stats = qeff_bench.benchmark_throughput(
                    model['model'],
                    config['qpc'],
                    config['devices'],
                    config['batch_size'],
                    config['prompt_len'],
                    config['generation_len'])
                print(stats)

                for name, measurement in stats.items():
                    all_results[name].append(measurement)

                if run_vllm:
                    stats = vllm_bench.benchmark_throughput(
                        model['model'],
                        config['qpc'],
                        config['devices'],
                        config['batch_size'],
                        config['prompt_len'],
                        config['generation_len'])
                    print(stats)

                    for name, measurement in stats.items():
                        all_results[name].append(measurement)

            df = pd.DataFrame.from_dict(all_results)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file with model configurations')
    parser.add_argument('--devices', help='List of comma separated device IDs to use for inferencing')
    parser.add_argument('--compile-only', action='store_true', help='Generate QPCs and skip benchmarking')
    parser.add_argument('--vllm', action='store_true', help='Run vLLM benchmarks')
    parser.add_argument('--hf_token', help='Hugging Face access token')
    args = parser.parse_args()

    if args.hf_token:
        login(args.hf_token)

    print('Benchmarking started at {}'.format(datetime.now()))

    results = qbench_benchmark(args.devices, args.config, args.vllm, args.compile_only)

    print('Benchmarking ended at {}'.format(datetime.now()))

    if not results.empty:
        print(results)
        results.to_csv('results_{}.csv'.format(int(time.time())), index=False)

if __name__=="__main__":
    main()
