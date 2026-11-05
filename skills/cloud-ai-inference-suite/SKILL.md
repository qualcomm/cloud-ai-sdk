---
name: cloud-ai-inference-suite
description: Qualcomm AI Inference Suite end-to-end guide. Use when user asks to connect to an inference suite endpoint, use the Imagine API, build GenAI apps with python-imagine-sdk, or call chat/embeddings/reranker/image/transcribe/translate endpoints backed by Cloud AI 100.
---

# Skill: Qualcomm AI Inference Suite – End-to-End Guide

## Overview

The **Qualcomm AI Inference Suite** is an inference-as-a-service platform backed by Qualcomm Cloud AI 100 hardware. It exposes a REST API (the "Imagine API") and a Python SDK (`python-imagine-sdk`) for building GenAI applications — chatbots, RAG pipelines, multimodal apps, agentic workflows.

This skill covers:
- Installation and environment setup
- Client construction and configuration
- All inference endpoints: chat, completion, embeddings, reranker, image generation, transcription, translation
- Model discovery and management
- Health checks and usage reporting
- Framework integrations (LangChain, CrewAI, AutoGen)
- Error handling and retry behavior
- Direct REST API reference

**SDK docs:** https://pypi.org/project/python-imagine-sdk/
**LangChain integration:** https://github.com/qualcomm/langchain-qualcomm-inference-suite
**Cloud AI 100 hardware:** https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100
**Support:** cloudai@qti.qualcomm.com

---

## 1. Installation

### 1.1 Basic (Imagine client only)

```bash
pip install python-imagine-sdk
```

Requires Python 3.9+.

### 1.2 With framework support

| Command | Installs |
|---|---|
| `pip install python-imagine-sdk` | Core `ImagineClient` only |
| `pip install python-imagine-sdk[langchain]` | Core + LangChain integration |
| `pip install python-imagine-sdk[crewai]` | Core + CrewAI integration |
| `pip install python-imagine-sdk[autogen]` | Core + AutoGen (Microsoft) integration |

### 1.3 Environment variables

| Variable | Required | Description |
|---|---|---|
| `IMAGINE_API_KEY` | Yes | Bearer token for API authentication |
| `IMAGINE_API_ENDPOINT` | Yes | Base URL of the inference suite endpoint, e.g. `https://my-server/api/v2` |
| `IMAGINE_API_PROXY` | No | HTTP/HTTPS proxy URL |
| `IMAGINE_DEBUG` | No | Set to `true` to enable debug logging |

Both `IMAGINE_API_KEY` and `IMAGINE_API_ENDPOINT` **must** be set — either as environment variables or passed as constructor arguments. Missing either raises `ValueError`.

---

## 2. Client Construction

Qualcomm Inference Suite provides two clients with identical methods: synchronous and asynchronous.

### 2.1 Synchronous client

```python
from imagine import ImagineClient

client = ImagineClient(
    endpoint="https://my-server/api/v2",   # or set IMAGINE_API_ENDPOINT
    api_key="my-api-key",                  # or set IMAGINE_API_KEY
    max_retries=3,                         # default: 3
    timeout=60,                            # seconds, default: 60
    verify=False,                          # SSL verification, default: False
    proxy=None,                            # optional HTTP proxy
    debug=False,                           # verbose logging
)
```

### 2.2 Asynchronous client

```python
from imagine import ImagineAsyncClient
import asyncio

async def main():
    client = ImagineAsyncClient(
        endpoint="https://my-server/api/v2",
        api_key="my-api-key",
    )
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="Llama-3.1-8B",
    )
    print(response.first_content)
    await client.close()  # async client has an explicit close()

asyncio.run(main())
```

Use `ImagineClient` for standard scripts and notebooks. Use `ImagineAsyncClient` when integrating into `asyncio`-based services.

---

## 3. Model Management

### 3.1 List available models

```python
# All models on the server
all_models = client.get_available_models()

# Filter by type
llm_models = client.get_available_models(model_type=ModelType.LLM)

# Grouped by type
by_type = client.get_available_models_by_type()
# Returns: {ModelType.LLM: ["Llama-3.1-8B", ...], ModelType.EMBEDDING: [...], ...}
```

### 3.2 ModelType enum

| Enum Value | String | Default Model |
|---|---|---|
| `ModelType.LLM` | `'llm'` | `Llama-3.1-8B` |
| `ModelType.EMBEDDING` | `'embedding'` | `BAAI/bge-large-en-v1.5` |
| `ModelType.RERANKER` | `'reranker'` | `BAAI/bge-reranker-base` |
| `ModelType.TEXT_TO_IMAGE` | `'text_to_image'` | `sdxl-turbo` |
| `ModelType.TRANSCRIBE` | `'transcribe'` | `whisper-tiny` |
| `ModelType.TRANSLATE` | `'translate'` | `Helsinki-NLP/opus-mt-en-es` |

```python
from imagine.types.models import ModelType
```

---

## 4. LLM Chat

### 4.1 Single-turn chat (non-streaming)

```python
from imagine import ChatMessage, ImagineClient

client = ImagineClient()

response = client.chat(
    messages=[ChatMessage(role="user", content="What is the best Spanish cheese?")],
    model="Llama-3.1-8B",
    temperature=0.7,
    max_tokens=512,
)

print(response.first_content)        # assistant reply text
print(response.usage.prompt_tokens)  # token usage
```

`ChatMessage` fields:
- `role: str` — `"user"`, `"assistant"`, `"system"`, or `"tool"`
- `content: str | list[dict] | None` — message text or multimodal content blocks
- `tool_calls: list | None` — for tool call responses
- `tool_call_id: str | None` — for tool result messages
- `name: str | None` — optional name

Dicts are also accepted: `{"role": "user", "content": "Hello"}`.

### 4.2 Multi-turn conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 4+4?"},
]
response = client.chat(messages=messages, model="Llama-3.1-8B")
```

### 4.3 Streaming chat

```python
for chunk in client.chat_stream(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="Llama-3.1-8B",
    max_tokens=1024,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 4.4 Tool use (function calling)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
response = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    model="Llama-3.1-8B",
    tools=tools,
)
```

### 4.5 LLM sampling parameters

All sampling parameters are optional (`None` = server default):

| Parameter | Type | Description |
|---|---|---|
| `max_tokens` | `int` | Max output tokens |
| `temperature` | `float` | Sampling temperature (0.0 = greedy) |
| `top_k` | `int` | Top-K sampling |
| `top_p` | `float` | Nucleus sampling threshold |
| `frequency_penalty` | `float` | Reduce repetition of frequent tokens |
| `presence_penalty` | `float` | Reduce repetition of any token seen so far |
| `repetition_penalty` | `float` | Repetition penalty (multiplicative, >1 = penalize) |
| `stop` | `list[str]` | Stop sequences |
| `max_seconds` | `int` | Timeout for inference in seconds |
| `ignore_eos` | `bool` | Continue past EOS token |
| `skip_special_tokens` | `bool` | Skip special tokens in output |
| `stop_token_ids` | `list[list[int]]` | Stop at specific token ID sequences |

---

## 5. Text Completion

For non-chat (raw prompt) inference:

```python
response = client.completion(
    prompt="The capital of France is",
    model="Llama-3.1-8B",
    max_tokens=50,
    temperature=0.0,
)
print(response.choices[0].text)

# Streaming
for chunk in client.completion_stream(prompt="Once upon a time", model="Llama-3.1-8B"):
    print(chunk.choices[0].text, end="", flush=True)
```

`CompletionResponse` fields: `id`, `object`, `created`, `model`, `choices`, `usage`, `generation_time`.

---

## 6. Embeddings

```python
# Single string
response = client.embeddings(
    text="What is the meaning of life?",
    model="BAAI/bge-large-en-v1.5",
)
vector = response.data[0].embedding  # list[float]

# Batch (list of strings)
response = client.embeddings(text=["doc1", "doc2", "doc3"])
vectors = [obj.embedding for obj in response.data]
```

`EmbeddingResponse` fields: `id`, `object`, `data` (list of `EmbeddingObject`), `model`, `usage`.

---

## 7. Re-Ranking

Re-rank a list of candidate documents against a query:

```python
response = client.reranker(
    query="What is machine learning?",
    documents=["ML is...", "Python is...", "Deep learning is..."],
    model="BAAI/bge-reranker-base",
    top_n=2,              # return only top 2
    return_documents=True, # include document text in response
)
for result in response.data:
    print(result.relevance_score, result.document.text)
```

`ReRankerResponse` fields: `id`, `data` (list of `ReRankerObject`), `failure`, `model`, `object`, `usage`.

---

## 8. Image Generation (Text-to-Image)

```python
response = client.images_generate(
    prompt="A photorealistic cat sitting on a spaceship",
    model="sdxl-turbo",
    negative_prompt="blurry",   # default: "blurry"
    seed=27,                    # default: 27
    n=1,                        # number of images
    num_inference_steps=20,     # default: 20
    size="512x512",             # default: "512x512"
    guidance_scale=6.5,         # default: 6.5
    response_format="b64_json", # "b64_json" or "url"
)
# Base64 image
import base64
img_data = base64.b64decode(response.data[0].b64_json)

# Streaming generation (progressive image updates)
for partial in client.images_generate_stream(prompt="A sunset over mountains"):
    pass  # partial.data[0].b64_json updated each frame
```

---

## 9. Audio Transcription

Transcribe audio files to text:

```python
# From file path (string)
response = client.transcribe(
    input_file="/path/to/audio.wav",
    model="whisper-tiny",
)
print(response.text)
print(response.ts)  # optional timestamp data

# From file object
with open("audio.mp3", "rb") as f:
    response = client.transcribe(input_file=f, model="whisper-tiny")
```

`TranscribeResponse` fields: `id`, `text`, `generation_time`, `ts` (optional timestamps).

---

## 10. Translation

Machine translation (default: English → Spanish):

```python
response = client.translate(
    prompt="The weather is nice today.",
    model="Helsinki-NLP/opus-mt-en-es",
    max_tokens=256,
)
print(response.choices[0].text)
```

`TranslateResponse` shares fields with `CompletionResponse`: `id`, `object`, `created`, `model`, `choices`, `usage`, `generation_time`.

---

## 11. Health Checks and Usage

### 11.1 Health check

```python
# Lightweight ping
pong = client.ping()
print(pong.status, pong.message)  # "ok", "pong"

# Full health (databases + models)
health = client.health_check()
print(health.postgres, health.redis, health.models)
```

### 11.2 Usage reporting

```python
import datetime as dt

response = client.usage(
    aggregation_duration="1h",              # optional: aggregate by hour
    since=dt.datetime(2025, 1, 1),          # optional start
    until=dt.datetime(2025, 12, 31),        # optional end
    model="Llama-3.1-8B",                   # optional model filter
)
# response.usage → list of UsageRecord
# response.overall → list of UsageRecordAggregated
```

### 11.3 History (per-session)

The client tracks the last N requests in-process memory:

```python
history = client.get_chat_history(max_items=5)
client.get_completion_history(max_items=5)
client.get_embedding_history(max_items=5)
client.get_reranker_history(max_items=5)
```

---

## 12. Framework Integrations

### 12.1 LangChain

```bash
pip install python-imagine-sdk[langchain]
# or
pip install langchain-qualcomm-inference-suite
```

```python
from imagine.langchain import ChatQIS, QISEmbeddings, QISLLM
# or
from langchain_qualcomm_inference_suite import ChatQIS, QISEmbeddings, QISLLM

# Chat model
llm = ChatQIS(model="Llama-3.1-8B")
llm.invoke("Explain transformers in one sentence.")

# Embeddings
embed = QISEmbeddings(model="BAAI/bge-large-en-v1.5")
embed.embed_query("What is RAG?")

# Raw LLM
raw = QISLLM(model="Llama-3.2-1B", temperature=0.0)
raw.invoke("The capital of Japan is")
```

`ChatQIS` / `QISLLM` init params (all optional beyond endpoint/api_key):
- `model`, `temperature`, `max_tokens`, `top_k`, `top_p`
- `frequency_penalty`, `presence_penalty`, `repetition_penalty`
- `stop`, `max_seconds`, `streaming`

GitHub: https://github.com/qualcomm/langchain-qualcomm-inference-suite

### 12.2 CrewAI

```bash
pip install python-imagine-sdk[crewai]
```

Use `ImagineClient` methods directly inside CrewAI tools, or use the LangChain wrappers above which CrewAI supports natively.

### 12.3 AutoGen (Microsoft)

```bash
pip install python-imagine-sdk[autogen]
```

---

## 13. Error Handling

| Exception | Trigger |
|---|---|
| `ValueError` | Missing `IMAGINE_API_KEY` or `IMAGINE_API_ENDPOINT` |
| `ImagineConnectionException` | Network failure / server unreachable |
| `ImagineAPIException` | HTTP 4xx error (except 429) |
| `ImagineAPIStatusException` | HTTP 500/502/503/504 — automatically retried with exponential backoff |
| `ImagineAPITooManyRequestsException` | HTTP 429 — rate limit exceeded |
| `ImagineException` | All other errors |

```python
from imagine import ImagineClient
from imagine.exceptions import (
    ImagineAPITooManyRequestsException,
    ImagineConnectionException,
    ImagineAPIException,
)

try:
    response = client.chat(messages=[{"role": "user", "content": "Hi"}])
except ImagineAPITooManyRequestsException:
    print("Rate limited — back off and retry")
except ImagineConnectionException as e:
    print(f"Cannot reach server: {e}")
except ImagineAPIException as e:
    print(f"API error {e.status_code}: {e.message}")
```

Retry behavior: `ImagineAPIStatusException` (5xx) is retried automatically up to `max_retries` times with exponential backoff (`2^attempt` seconds). Set `max_retries=0` to disable retries.

---

## 14. REST API Reference

The Imagine API is a standard REST API. All requests require:
- `Authorization: Bearer <API_KEY>` header
- Base URL: `IMAGINE_API_ENDPOINT` (e.g., `https://my-server/api/v2`)

| Method | Path | Description |
|---|---|---|
| GET | `ping` | Lightweight liveness check |
| GET | `health` | Full health (postgres, redis, models) |
| GET | `models` | List all available models (grouped by type) |
| GET | `usage` | Token usage reporting (params: `aggregation_duration`, `since`, `until`, `model`) |
| POST | `chat/completions` | Chat (OpenAI-compatible messages format) |
| POST | `completions` | Raw text completion |
| POST | `embeddings` | Text-to-vector embeddings |
| POST | `reranker` | Document re-ranking |
| POST | `images/generations` | Text-to-image generation |
| POST | `translate` | Machine translation |
| GET | `reranker` | Re-ranker history (param: `max_items`) |
| GET | `embeddings` | Embedding history (param: `max_items`) |
| POST | `transcribe` | Audio transcription (multipart/form-data) |

**Direct curl example:**

```bash
curl -s https://my-server/api/v2/chat/completions \
  -H "Authorization: Bearer $IMAGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B",
    "stream": false,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**SSE streaming:**

```bash
curl -s https://my-server/api/v2/chat/completions \
  -H "Authorization: Bearer $IMAGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"model": "Llama-3.1-8B", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}' \
  | grep "^data:" | grep -v "[DONE]" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line[6:])
    delta = d[\"choices\"][0][\"delta\"].get(\"content\", \"\")
    print(delta, end=\"\", flush=True)
"
```

---

## 15. Quick Reference

```python
from imagine import ImagineClient, ChatMessage
from imagine.types.models import ModelType

client = ImagineClient()  # reads IMAGINE_API_KEY and IMAGINE_API_ENDPOINT

# LLM chat
r = client.chat([{"role": "user", "content": "Hi"}], model="Llama-3.1-8B")
print(r.first_content)

# Embeddings
r = client.embeddings("embed this", model="BAAI/bge-large-en-v1.5")
vec = r.data[0].embedding

# Re-rank
r = client.reranker("query", ["doc1", "doc2"], top_n=1)

# Image
r = client.images_generate("a red robot", model="sdxl-turbo")

# Audio
r = client.transcribe("/path/to/audio.wav")

# Health
client.ping()
client.health_check()

# Models
client.get_available_models(ModelType.LLM)
```

---

## 16. OpenAI SDK Compatibility

Qualcomm Inference Suite exposes OpenAI-compatible endpoints (`chat/completions`, `completions`, `embeddings`) at the same base URL. You can use the standard `openai` Python package directly:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="my-api-key",                    # maps to IMAGINE_API_KEY
    base_url="https://my-server/api/v2",     # maps to IMAGINE_API_ENDPOINT
)

# Chat
response = client.chat.completions.create(
    model="Llama-3.1-8B",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)

# Streaming chat
stream = client.chat.completions.create(
    model="Llama-3.1-8B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

# Embeddings
emb = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input="What is machine learning?",
)
vector = emb.data[0].embedding
```

**When to use `openai` vs `python-imagine-sdk`:**
- Use `openai` when integrating with existing tooling that expects an OpenAI client (e.g., LangChain's `ChatOpenAI`, Open WebUI, LiteLLM, frameworks that accept `openai.OpenAI`).
- Use `python-imagine-sdk` for Qualcomm Inference Suite-specific features: reranker, image generation, audio transcription, translation, usage reporting, or in-process history.

---

## Related Skills

- **Cloud AI 100 – LLM Onboarding** (`cloud-ai-llm-onboarding` skill) — Use to deploy a new LLM onto the Cloud AI 100 hardware backing this Qualcomm Inference Suite endpoint, using QEfficient.
- **Cloud AI 100 – Model Conversion & Compilation** (`cloud-ai-model-conversion` skill) — Use to export and compile vision/CNN models to run on the Qualcomm Inference Suite hardware.
- **Cloud AI 100 – Environment Setup** (`cloud-ai-env-setup` skill) — Use to set up the Cloud AI 100 Docker/SDK environment on the server running Qualcomm Inference Suite.
- **Claude API** (`claude-api` skill) — For building apps with the Anthropic Claude API; architectural reference for designing GenAI apps on top of Qualcomm Inference Suite.
