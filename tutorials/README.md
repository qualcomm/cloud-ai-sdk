Tutorials are Jupyter notebools designed to walk the developer through the Cloud AI inference workflow. The tutorials are split into 2 categories - CV and NLP. Overall, the inference workflow for CV and NLP models are very similar and have been presented for convenience. 

`Model-Onboarding` - This is one of the beginner notebooks. This goes through exporting and preparing the model, compiling the model using a CLI tool and executing inference using CLI tool / Python APIs. 

`Performance-Tuning` - This is another beginner notebook that walks the developer through the key parameters to optimize for best performance (latency and throughput) on Cloud AI platforms. Going through this notebook and 'Performance Tuning' section in the Quick start guide will equip developers with a intuitive understanding of how to use the key parameters to meet inference application KPIs (AI compute resource usage, throughput and latency).   

`Profiler` - This is a intermediate-level notebook that describes system and device level inference profiling capabilities. Developers can use the tools and techniques described in this tutorial to measure application/device level latency and identify system/device bottlenecks. 


### Pre-requisites
1. Clone this repo
2. Create python3.12 venv and activate it.
   `python3.12 -m venv jn_env` <br>
   `source jn_env/bin/activate` <br>
3. Install qaic
   `pip install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl`
4. Install Jupyter notebook
   `pip install notebook`
   `pip install urllib3==1.26.6`
5. Run the notebook
   `jupyter notebook --allow-root --ip 0.0.0.0 --no-browser`.<br>
   You should see `http://<hostname>:8888/tree?token=<token#>`.<br>
   On the local machine, type `http://<hostname>:8888/tree?token=<token#>` on a browser to run the tutorial notebooks. 
