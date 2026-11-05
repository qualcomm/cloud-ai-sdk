## Description

Download the YOLOv5, YOLOv7 and YOLOv8 models, prepare for the Qualcomm AIC100, compile for high-throughput, minimum-latency, or balanced throughput with fp16 precision, run the model on a generated random sample, and obtain the benchmarking results and output values.

## Source of the models

The models are downloaded from (https://github.com/ultralytics/yolov5). This script has been tested for the following requested models:
* YOLOv5s
* YOLOv5m
* YOLOv5l
* YOLOv5x
* YOLOv7-e6e
* YOLOv8m

## Virtual environment

For a quick environment setup:

```commandline
python3.12 -m venv det_env
source det_env/bin/activate
pip3 install -r requirements.txt

```
## Syntax

Copy the run_yolo_model.py and the lut_yolo_models.csv to a working directory. Pick a MODEL_NAME from the list above, and type:

```commandline

usage: run_yolo_model.py [-h] --model-name {yolov5s,yolov5m,yolov5l,yolov5x,yolov7-e6e,yolov8m}
             [--objective {best-latency,best-throughput,balanced}] 
	     [--opset OPSET] 
	     [--batch-size BATCH_SIZE]
             [--image-size IMAGE_SIZE] 
	     [--cores {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
             [--instances {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
	     [--ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
	     [--mos {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
             [--set-size {1,2,3,4,5,6,7,8,9,10}] 
	     [--extra EXTRA] 
	     [--time TIME] 
	     [--device {0,1,2,3,4,5,6,7}] 
	     [--run-only]



Download, Compile, and Run YOLO models on randomly generated inputs.


optional arguments:
  -h, --help            show this help message and exit
  --model-name, -m {yolov5s,yolov5m,yolov5l,yolov5x,yolov7-e6e,yolov8m}
                        Model name to download.
  --objective, -o {best-latency,best-throughput,balanced}
                        Running for best-latency, best-throughput, or balanced
  --opset OPSET         ONNX opset. Default <12>
  --batch-size, -b BATCH_SIZE
                        Sample input batch size. Default <1>.
  --image-size, -s IMAGE_SIZE
                        Sample input image width/height. Default <640>.
  --cores, -c {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of AIC100 cores to compile the model for. Default <2>
  --instances, -i {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of model instances to run on AIC100. Default <7>
  --ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Overlap split factor. Default <1>
  --mos {1,2,3,4,5,6,7,8,9,10,11,12,13,14}             
                        Maximum output channel split. Default <None>
  --set-size {1,2,3,4,5,6,7,8,9,10}
                        Set size. Default <10>
  --extra EXTRA         Extra compilation arguments.
  --time TIME           Duration (in seconds) for which to submit inferences. Default <20>
  --device, -d {0,1,2,3,4,5,6,7}
                        AIC100 device ID. Default <0>
  --run-only, -r        Performs the inference only, without re-exporting and re-compiling the model
  --include-nms         Run the model preparator tool to optimize the graph, and to add the Post Processing to supported models. Details on model preparator tool here- https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Inference-Workflow/Export-the-model/Prepare-the-model/ 


```
Examples:
```commandline
python run_yolo_model.py -m yolov5s -o best-throughput
```
```commandline
python run_yolo_model.py -m yolov5m -o balanced
```
```commandline
python run_yolo_model.py -m yolov5x -o best-throughput
```

The hardware configuration will be either associated to the corresponding row in the lut_yolo_models.csv or to default values if not specified by the user. If the MODEL_NAME is not included in the lut_yolo_models.csv, default values will be used.

After download, compilation, and run is complete, the working directory of the selected model looks as follows. 
# Working directory structure
```
|── model                   # Contains the onnx file of the picked model 
|   └── model.onnx          # The onnx file of the picked model
|── inputFiles              # Contains the (randomly generated) input files of the compiled model
│   └── input_img*.raw      # Randomly generated input files of the compiled model
|── outputFiles             # Contains the corresponding output to input, as well as the hardware profiling for latency
│   └── fp16*               
│       └── output-act*.bin # Corresponding output to the randomly generated input_img*.raw
│       └── aic-profil*.bin # The hardware profiling for round trip latency between host and device for each inference
├── compiled-bin*           # Compilation path
│   └── programqpc.bin      # For the selected objective, the model.onnx is compiled into programqpc.bin 
├── list*.txt               # A list that contains paths to the inputs. Can be used as input to qaic-runner
├── commands*.txt           # Includes necessary compilation and running scripts to reproduce the results manually.

```
To manually reproduce the results, navigate to the working directory, select the qaic compile/run commands from the command*.txt and run them in the terminal.

## Docker Compose Setup

### Build the Docker image

```commandline
docker compose build
```

### Export YOLO model and generate Triton repo

```commandline
# Default: yolov8m
docker compose run --rm yolo-export

# Or specify model
YOLO_MODEL=yolov5m docker compose run --rm yolo-export
```

This generates a Triton model repository at `models/vision/detection/${YOLO_MODEL}/triton-model-repo/`.

### Start Triton server

```commandline
# Default port 8000
docker compose up yolo-triton

# Or customize ports
TRITON_HTTP_PORT=8080 docker compose up yolo-triton
```

### Run inference client

```commandline
# From host machine
pip3 install numpy Pillow
python3 triton_client.py --endpoint localhost:8000
```

### Options

- `--endpoint`: Triton server URL (default: localhost:8000)
- `--image`: Image path or URL (default: https://ultralytics.com/images/zidane.jpg)
- `--model`: Model name in Triton (default: yolo)
- `--output`: Output image path with detections (default: output.jpg)

### Environment variables

- `YOLO_MODEL`: Model subfolder (default: yolov8m)
- `TRITON_HTTP_PORT`: HTTP port (default: 8000)
- `TRITON_GRPC_PORT`: gRPC port (default: 8001)
- `TRITON_METRICS_PORT`: Metrics port (default: 8002)
- `TRITON_LOG_VERBOSE`: Enable verbose logging (default: 0) 
