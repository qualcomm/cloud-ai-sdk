## Prerequisites

- Python 3.12
- A Cloud AI 100 environment or container with the Platform and Apps SDKs installed
- A Hugging Face account and access token. The default example is gated.
- Persistent writable storage for model downloads, exported ONNX, and QPC files.

Set these variables in the shell before starting Jupyter. The paths must be
available from the notebook's Python process:

```bash
export HF_TOKEN="<your Hugging Face token>"
export HF_HOME="/path/to/persistent/hf_home"
export QEFF_HOME="/path/to/persistent/qeff_models"
mkdir -p "$HF_HOME" "$QEFF_HOME"
```

`HF_HOME` stores Hugging Face models and tokenizers. `QEFF_HOME` stores
QEfficient exports, ONNX files, and QPC artifacts.

The notebook reads these variables before importing QEfficient. If they are
unset, it uses `./hf_home` and `./qeff_models` relative to the notebook's
working directory.

## Installation steps

### Create python virtual environment and activate it
```
python3.12 -m venv qeff_env
source qeff_env/bin/activate
pip install --upgrade pip
```

### Install the efficient transformers library
```
pip install git+https://github.com/qualcomm/efficient-transformers@release/v1.23.0
```

### After installation of efficient transformers library, install jupyter notebook
```
pip install notebook
```

Verify the installed packages:

```bash
python -c 'import QEfficient, peft; print(QEfficient.__version__)'
```

### Launch Notebook
```
jupyter notebook --no-browser --allow-root
```

Jupyter prints a browser URL in the terminal, including an access token. Copy
that complete URL into a web browser on the host running the notebook. With
the default port, it will look like:

```
http://localhost:8888/tree?token=<token>
```

Open `Efficient transformers library for Cloud AI 100.ipynb` from the Jupyter
file browser.
