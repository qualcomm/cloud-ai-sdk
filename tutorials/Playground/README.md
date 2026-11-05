# Cloud AI Playground Notebook setup

## Set Up the Python Environment

```bash
# Create and activate the virtual environment.
python3.12 -m venv imagine_env
source imagine_env/bin/activate
python -m pip install --upgrade pip

# Install the Qualcomm Imagine Python library and other dependencies.
python -m pip install -r requirements.txt
```

## Configure Playground Access

Export the Playground credentials in the terminal before starting Jupyter. The
notebooks inherit these values and do not store API keys.

```bash
export IMAGINE_API_KEY="<your Playground API key>"
export IMAGINE_API_ENDPOINT="https://aisuite.cirrascale.com/apis/v2"
```

## Launch Notebook

With `imagine_env` still active, start Jupyter:

```bash
python -m jupyter notebook \
  --no-browser \
  --allow-root
```

Jupyter listens on its default local port, `8888`, and prints a URL containing
an access token. Open that complete URL in a browser; it will look like:

```text
http://localhost:8888/tree?token=<token>
```

For a remote host, create an SSH tunnel first:

```bash
ssh -L 8888:localhost:8888 <user>@<host>
```

Then open the printed Jupyter URL through `http://localhost:8888`.



