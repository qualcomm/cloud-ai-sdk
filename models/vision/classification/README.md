## Description
---

This document uses a script (run_cv_classifier.py) to download a computer-vision model from Huggingface or Torchvision, prepares it for the Qualcomm AIC100, compiles it for a specific hardware configuration (best-throughput or best-latency) with fp16 precision, runs the model on a generated random sample, and obtains the benchmarking results (Inf/Sec and Latency) and output values.

## Source of the models
---
This script has been tested for the following models. These models are downloaded either from Torchvision (e.g. resnet-50) or HuggingFace (e.g. resnet50).

* alexnet
* densenet121
* densenet161
* densenet169
* densenet201
* efficientnet_b0
* efficientnet_b1
* efficientnet_b2
* efficientnet_b3
* efficientnet_b4
* efficientnet_b5
* efficientnet_b6
* efficientnet_b7
* efficientnet_v2_l
* efficientnet_v2_m
* efficientnet_v2_s
* inception_v3
* mnasnet0_5
* mnasnet0_75
* mnasnet1_0
* mnasnet1_3
* mobilenet_v2
* mobilenet_v3_large
* mobilenet_v3_small
* resnet101
* resnet152
* resnet-152
* resnet18
* resnet34
* resnet50
* resnet-50
* resnext101_32x8d
* resnext101_64x4d
* resnext50_32x4d
* shufflenet_v2_x0_5
* shufflenet_v2_x1_0
* shufflenet_v2_x1_5
* shufflenet_v2_x2_0
* squeezenet1_0
* squeezenet1_1
* vgg11
* vgg11_bn
* vgg13
* vgg13_bn
* vgg16
* vgg16_bn
* vgg19
* vgg19_bn
* vit_b_16
* vit_b_32
* vit-base-patch16-224
* wide_resnet101_2
* wide_resnet50_2

## Virtual environment
---
For a quick environment setup:

```commandline
python3.12 -m venv cv_env
source cv_env/bin/activate
```

## Framework and version
---
```commandline
pip3 install -r requirements.txt
```
## Syntax
---
Pick a MODEl_NAME from the list above. At the working directory where two files run_cv_classifier.py and the lut_cv_classifiers.csv exist, use the following command:

```commandline
usage: run_cv_classifier.py [-h] --model-name {alexnet, densenet121, densenet161, densenet169, densenet201, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, inception_v3, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, resnet101, resnet152, resnet-152, resnet18, resnet34, resnet50, resnet-50, resnext101_32x8d, resnext101_64x4d, resnext50_32x4d, shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vit_b_16, vit_b_32, vit-base-patch16-224, wide_resnet101_2, wide_resnet50_2}
                            [--objective {best-latency,best-throughput,balanced}] 
			    [--opset OPSET] 
			    [--batch-size BATCH_SIZE]
                            [--image-size IMAGE_SIZE] 
			    [--cores {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
                            [--instances {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
			    [--ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
			    [--mos MOS]
                            [--set-size {1,2,3,4,5,6,7,8,9,10}] 
			    [--extra EXTRA] 
			    [--time TIME] 
			    [--device {0,1,2,3,4,5,6,7}] 
			    [--run-only]

Download, Compile, and Run vision models on randomly generated inputs

optional arguments:
  -h, --help            show this help message and exit
  --model-name, -m {alexnet, densenet121, densenet161, densenet169, densenet201, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, inception_v3, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, resnet101, resnet152, resnet-152, resnet18, resnet34, resnet50, resnet-50, resnext101_32x8d, resnext101_64x4d, resnext50_32x4d, shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vit_b_16, vit_b_32, vit-base-patch16-224, wide_resnet101_2, wide_resnet50_2}
                        Model name to download.
  --objective, -o OBJECTIVE
                        Running for best-latency, best-throughput, or balanced
  --opset OPSET         ONNX opset. Default <13>
  --batch-size, -b BATCH_SIZE
                        Sample input batch size.
  --image-size, -s IMAGE_SIZE
                        Sample input image width/height. Default <224>.
  --cores, -c {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of AIC100 cores to compile the model for. Default <2>
  --instances, -i {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of model instances to run on AIC100. Default <7>
  --ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Overlap split factor. Default <1>
  --mos MOS             Maximum output channel split.
  --set-size {1,2,3,4,5,6,7,8,9,10}
                        Set size. Default <10>
  --extra EXTRA         Extra compilation arguments.
  --time TIME           Duration (in seconds) for which to submit inferences. Default <20>
  --device, -d {0,1,2,3,4,5,6,7}
                        AIC100 device ID. Default <0>
  --run-only, -r        Performs the inference only, without re-exporting and re-compiling the model


```
Examples:
```commandline
python run_cv_classifier.py -m resnet-50
```
```commandline
python run_cv_classifier.py -m resnet-152 -o best-throughput
```
```commandline
python run_cv_classifier.py -m resnet-50 -o best-latency
```
```commandline
python run_cv_classifier.py -m vit-base-patch16-224 -o best-latency
```

The hardware configuration will be either associated to the corresponding row in the lut_cv_classifiers.csv or to defualt values if not specified by the user. If the MODEL_NAME is not included in the lut_cv_classifiers.csv, default values will be used.

After download, compile, and run is complete, the working directory of the selected model is as follows. 
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
├── list*.txt               # A list that contains path to the inputs. Can be used as input to qaic-runner
├── commands*.txt           # Includes necessary compilation and running scripts to reproduce the results manually.

```
To manually resproduce the results, navigate to the working directory, select the qaic compile/run commands from the command*.txt and run them in the terminal. 
