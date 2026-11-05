## Description
---

Download the DETR-ResNet50 model, prepare for the Qualcomm AIC100, compile the model, run the model on a generated random sample along with input image, and obtain the output.


## Source of the model
---

This model is an implementation of DETR-ResNet50 found at (https://github.com/facebookresearch/detr).


## Environment setup
---
Run these commands from the DETR tutorial directory. Use the SDK 1.23.2.0
container or a host environment with the matching Cloud AI SDK installed.

```bash
cd tutorials/Computer-Vision/DETR
python3.12 -m venv cv_workflow_env
source cv_workflow_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebook also requires access to the SDK tools and a Cloud AI device. It
uses the SDK-shipped `qaic-pytools` model preparator and
`/opt/qti-aic/exec/qaic-compile`.

## Launch the notebook
---
With `cv_workflow_env` active, start Jupyter from the same directory:

```commandline
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root
```

Open the URL printed by Jupyter. When running remotely or in a container,
forward or expose port `8888` to the machine where the browser is running.

