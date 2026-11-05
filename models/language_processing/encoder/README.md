## Description
---
This document runs a script (run_nlp_model.py) to benchmark encoder-type NLP model from Hugging Face.  The script downloads the model, prepares it for the Qualcomm Cloud AI accelerator, compiles it for a specific hardware configuration (best-throughput or best-latency) with fp16 precision, runs the model on a generated random sample, and obtains the benchmarking results and output values.

## Efficient Transformers
---
Support for embedding models is now available in the Qualcomm efficient-transformers library.  Refer to [link](https://qualcomm.github.io/efficient-transformers/source/validate.html#embedding-models) for more details.

## Source of the models
---
The models are downloaded from (https://huggingface.co). This script has been tested with the following models:

* albert-base-v2
* allenai/scibert_scivocab_uncased
* avichr/heBERT_sentiment_analysis
* BAAI/bge-reranker-large
* BAAI/bge-base-en-v1.5
* BAAI/bge-reranker-v2-m3
* BAAI/bge-large-en-v1.5
* BAAI/bge-small-en-v1.5
* beomi/kcbert-base
* bert-base-cased
* bert-base-chinese
* bert-base-german-cased
* bert-base-multilingual-cased
* bert-base-multilingual-uncased
* bert-base-uncased
* bert-large-uncased
* bert-large-uncased-whole-word-masking-finetuned-squad
* bhadresh-savani/distilbert-base-uncased-emotion
* camembert-base
* cardiffnlp/twitter-roberta-base-sentiment
* cardiffnlp/twitter-roberta-base-sentiment-latest
* ckiplab/albert-tiny-chinese
* classla/bcms-bertic-ner
* cl-tohoku/bert-base-japanese
* cl-tohoku/bert-base-japanese-char
* cl-tohoku/bert-base-japanese-whole-word-masking
* cross-encoder/ms-marco-MiniLM-L-12-v2
* dbmdz/bert-base-italian-xxl-cased
* dbmdz/electra-base-german-europeana-cased-discriminator
* DeepPavlov/rubert-base-cased-conversational
* DeepPavlov/rubert-base-cased-sentence
* deepset/roberta-base-squad2
* deepset/xlm-roberta-large-squad2
* distilbert-base-cased
* distilbert-base-cased-distilled-squad
* distilbert-base-german-cased
* distilbert-base-multilingual-cased
* distilbert-base-uncased
* distilbert-base-uncased-distilled-squad
* distilbert-base-uncased-finetuned-sst-2-english
* distilroberta-base
* dslim/bert-base-NER
* dslim/bert-base-NER-uncased
* emilyalsentzer/Bio_ClinicalBERT
* finiteautomata/beto-sentiment-analysis
* google/bert_uncased_L-2_H-128_A-2
* google/electra-base-discriminator
* google/electra-small-discriminator
* hfl/chinese-bert-wwm-ext
* hfl/chinese-electra-180g-base-discriminator
* hfl/chinese-roberta-wwm-ext
* intfloat/e5-large
* intfloat/e5-large-v2
* jinaai/jina-embeddings-v2-base-code
* kit-nlp/bert-base-japanese-sentiment-cyberbullying
* klue/bert-base
* m3hrdadfi/bert-fa-base-uncased-wikinli
* Maltehb/danish-bert-botxo
* microsoft/codebert-base
* microsoft/deberta-v3-base
* microsoft/deberta-v3-small
* monologg/bert-base-cased-goemotions-original
* mrm8488/bert-spanish-cased-finetuned-ner
* nlptown/bert-base-multilingual-uncased-sentiment
* prajjwal1/bert-mini
* prajjwal1/bert-tiny
* ProsusAI/finbert
* roberta-base
* roberta-large
* roberta-large-mnli
* Rostlab/prot_bert
* sentence-transformers/all-distilroberta-v1
* sentence-transformers/all-MiniLM-L6-v2
* sentence-transformers/all-mpnet-base-v2
* sentence-transformers/bert-base-nli-mean-tokens
* sentence-transformers/distilbert-base-nli-stsb-mean-tokens
* sentence-transformers/gtr-t5-large
* sentence-transformers/msmarco-distilbert-base-tas-b
* sentence-transformers/paraphrase-MiniLM-L6-v2
* sentence-transformers/paraphrase-mpnet-base-v2
* sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
* sentence-transformers/paraphrase-xlm-r-multilingual-v1
* tals/albert-xlarge-vitaminc-mnli
* vinai/bertweet-base
* vinai/phobert-base
* xlm-roberta-base
* xlm-roberta-large
* xlm-roberta-large-finetuned-conll03-english
* yiyanghkust/finbert-tone

## Virtual environment
---
For a quick environment setup:

```commandline
python3.12 -m venv nlp_env
source nlp_env/bin/activate
pip3 install -r requirements.txt
```

## Hugging Face authentication

Some models on Hugging Face require an access token.  Refer to https://huggingface.co/docs/hub/en/security-tokens for more information.

```
export HF_TOKEN=<your_auth_token>
```

## Syntax
---
Pick a MODEl_NAME from the list above. At the working directory where two files run_nlp_model.py and the lut_nlp_models.csv exist, use the following command:


```commandline
usage: run_nlp_model.py [-h] --model-name MODEL_NAME
                        [--task {default,fill-mask,question-answering,text-classification,token-classification,feature-extraction,sentence-similarity}]
                        [--objective {best-latency,best-throughput,balanced}] 
			[--opset OPSET] 
			[--batch-size BATCH_SIZE]
                        [--seq-length SEQ_LENGTH] 
			[--cores {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
                        [--instances {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
			[--ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}] 
			[--mos MOS]
                        [--set-size {1,2,3,4,5,6,7,8,9,10}] 
			[--extra EXTRA] 
			[--time TIME] 
			[--device {0,1,2,3,4,5,6,7}] 
                        [--api-run]
			[--run-only]

Download, Compile, and Run encoder-type NLP models on randomly generated inputs

optional arguments:
  -h, --help            show this help message and exit
  --model-name, -m MODEL_NAME
                        Model name to download from Hugging Face. Try bert-base-cased for instance.
  --task, -t {default,fill-mask,question-answering,text-classification,token-classification,feature-extraction,sentence-similarity}
                        Model task for encoder-type NLP models
  --objective, -o {best-latency,best-throughput,balanced}
                        Running for best-latency, best-throughput, or balanced
  --opset OPSET         ONNX opset. Default <13>
  --batch-size, -b BATCH_SIZE
                        Sample input batch size. Default <1>.
  --seq-length, -s SEQ_LENGTH
                        Sample input sequence length. Default <128>.
  --cores {1,2,3,4,5,6,7,8,9,10,11,12,13,14}, -c {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of AI cores to compile the model for. Default <2>
  --instances, -i {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Number of model instances to run on Cloud AI device. Default <7>
  --ols {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        Overlap split factor. Default <1>
  --mos MOS             Maximum output channel split. Default <1>
  --set-size {1,2,3,4,5,6,7,8,9,10}
                        Set size. Default <10>
  --extra EXTRA         Extra compilation arguments.
  --time TIME           Duration (in seconds) for which to submit inferences. Default <20>
  --device, -d {0,1,2,3,4,5,6,7}
                        Cloud AI device ID. Default <0>
  --api-run, -a         Performs the inference using qaic session (high-level) and qaicrt(low-level) Python APIs. If this flag is not specified, qaic-runner CLI is used. 
  --run-only, -r        Performs the inference only, without re-exporting and re-compiling the model

```
Examples:
Use qaic session and qaicrt Python APIs 
```commandline
python run_nlp_model.py --model-name albert-base-v2 -t question-answering --objective best-throughput --api-run
```

Use qaic-runner CLI
```commandline
python run_nlp_model.py -m Rostlab/prot_bert
```
```commandline
python run_nlp_model.py -m bert-base-cased -t question-answering -o best-throughput
```
```commandline
python run_nlp_model.py --model-name bert-base-uncased --objective best-latency
```

The TASK and hardware configuration will be either associated to the corresponding row in the lut_nlp_models.csv or to default values if not specified by the user. If the MODEL_NAME is not included in the lut_nlp_models.csv, pick a corresponding task, or switch to default.

After download, compile, and run is complete, the working directory of the selected model is as follows. 
# Working directory structure
```
|── model                   # Contains the onnx file of the picked model 
|   └── model.onnx          # The onnx file of the picked model
|── inputFiles              # Contains the (randomly generated) input files of the compiled model
│   └── input_ids*.raw      # Randomly generated input files for the compiled model
│   └── attention_mask*.raw 
│   └── token_type_ids*.raw 
|── outputFiles             # Contains the corresponding output to input, as well as the hardware profiling for latency
│   └── fp16*               
│       └── output-act*.bin # Corresponding output to the randomly generated input_img*.raw
│       └── aic-profil*.bin # The hardware profiling for round trip latency between host and device for each inference
├── compiled-bin*           # Compilation path
│   └── programqpc.bin      # For the selected objective, the model.onnx is compiled into programqpc.bin 
├── list*.txt               # A list that contains path to the inputs. Can be used as input to qaic-runner
├── commands*.txt           # Includes necessary compilation and running scripts to reproduce the results manually.

```
To manually reproduce the results, navigate to the working directory, select the qaic compile/run commands from the command*.txt and run them in the terminal. 


## Run an OpenAI-compatible REST endpoint
---

Setup:

```
source ./nlp_env/bin/activate
```

Build the embedding model:

```
python run_nlp_model.py -m BAAI/bge-large-en-v1.5 --task feature-extraction --objective best-throughput
```

Start endpoint:

```
python server.py --qpc_path ./models/BAAI/bge-large-en-v1.5/compiled-bin-fp16-B1-C4-A3-OLS2-MOS1-best-throughput
```

Test the endpoint:

```
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["your-text-here"],
    "model": "bge-large-en-v1.5",
    "encoding_format": "float"
  }'
```

