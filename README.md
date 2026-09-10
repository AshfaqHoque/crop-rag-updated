# Crop RAG Chatbot

A retrieval-augmented chatbot for crop-advisory questions in Bangladesh. The
project exposes a FastAPI endpoint, a Streamlit chat UI, a LangGraph workflow,
and Chroma-backed knowledge indexes for crop and company information.

The application accepts Bangla and English requests. It can answer questions
about crops, varieties, cultivation, fertilizer, pests, diseases, harvesting,
and the indexed company knowledge base. Answers are generated from retrieved
context, but should still be checked before important farming or business
decisions are made.

## Features

- FastAPI API with typed request and response models.
- Streamlit chat interface with multiple local conversations.
- LangGraph routing for crop questions, company questions, chitchat, and
  meaningless input.
- Dense multilingual retrieval with Chroma and `bge-m3` embeddings.
- External reranking and context compression for crop questions.
- Follow-up query rewriting using LangGraph checkpointed session state.
- Separate crop and company knowledge collections.
- JSONL ingestion commands for pre-built chunks and source data.

## Architecture

```text
POST /api/v1/chat
        |
        v
  rewrite_query
        |
       route
    /    |       |          \
   /     |        |           \
crop  company  chitchat  meaningless
  |       |        |           |
extract  retrieve  direct      direct
  |       |        |           |
retrieve  generate  generate    generate
  |
rerank -> compress -> generate
```

The compiled graph starts with query rewriting and then routes by intent:

- `crop_query`: extract crop information, retrieve from the crop collection,
  call the external reranker, compress context, and generate an answer.
- `company_query`: retrieve from `company_knowledge_base` and generate an
  answer without the crop reranker/compressor path.
- `chitchat`: generate a conversational response without retrieval.
- `meaningless`: generate a clarification response without retrieval.

The graph uses `session_id` as its LangGraph `thread_id`. The default
`MemorySaver` checkpoint is process-local; use a shared checkpoint backend
before running multiple API replicas.

## Requirements

- Python 3.11 or newer.
- Chroma, either embedded locally or reachable as an HTTP service.
- An Ollama server for embeddings using `bge-m3`.
- A chat provider: vLLM, Ollama, or Groq.
- An HTTP reranker for crop questions at `RERANKER_URL`.

The default Python settings use an OpenAI-compatible vLLM server at
`http://localhost:8091/v1`. Docker Compose overrides this and uses Ollama.

## Installation

From the repository root:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The Streamlit UI uses `requests`. If it is not already present in the local
environment, install it with:

```bash
python -m pip install requests
```

## Configuration

Create a `.env` file in the repository root when changing defaults. Important
settings and their current defaults are:

```dotenv
APP_ENV=dev
CHAT_PROVIDER=vllm
VLLM_CHAT_MODEL=gemma4:12b
VLLM_BASE_URL=http://localhost:8091/v1
VLLM_API_KEY=not-needed
OLLAMA_CHAT_MODEL=gemma4:12b
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=bge-m3:latest
GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=
CHROMA_HOST=
CHROMA_PORT=8000
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=crop_knowledge_base
CHROMA_COMPANY_COLLECTION=company_knowledge_base
CROP_REGISTRY_PATH=./data/crops.json
RETRIEVAL_TOP_K=20
COMPANY_RETRIEVAL_TOP_K=3
RERANK_TOP_K=6
RERANKER_URL=http://localhost:8090/rerank
HISTORY_MAX_TURNS=1
CONTEXT_MAX_CHARS_PER_CHUNK=3000
LLM_TEMPERATURE=0
CHECKPOINT_BACKEND=memory
```

Allowed `CHAT_PROVIDER` values are `vllm`, `ollama`, and `groq`. Groq requires
`GROQ_API_KEY`; embeddings still use Ollama. LangSmith tracing is enabled by
default in the settings and can be controlled with `LANGSMITH_TRACING` and
the related LangSmith settings.

The checkpoint implementation currently supports Redis when configured and
otherwise falls back to in-memory state. The `postgres` and `sqlite` values
listed in the configuration comments are not currently implemented.

## Run Locally

Start the required model services first. For Ollama, for example:

```bash
ollama pull bge-m3
ollama pull gemma4:12b
```

Start the API from the repository root:

```bash
uvicorn app.main:app --reload --app-dir src
```

The API runs at `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Example health request:

```bash
curl http://localhost:8000/health
```

Start the optional web UI in a second terminal:

```bash
streamlit run ui/streamlit_app.py
```

The UI currently sends requests to the hardcoded API URL
`http://localhost:8000/api/v1/chat`.

## Docker Compose

```bash
docker compose up --build
```

Compose starts:

- `app` on host port `8000`;
- `ollama` on host port `11434`;
- `ollama-init`, which downloads `gemma3:4b` and `bge-m3`;
- `chroma` on host port `8001`.

The Compose app uses Ollama, `gemma3:4b`, Chroma at `chroma:8000`, and the
reranker URL `http://host.docker.internal:8090/rerank` unless overridden with
`RERANKER_URL`. Compose does not start the Streamlit UI or a reranker service.

The host `data` directory is mounted into the app container. Named volumes
preserve Ollama models, Chroma data, and the Hugging Face cache.

## Build the Knowledge Indexes

### Crop knowledge

Build fact-unit chunks from `data/crops.json`, write `data/chunks.jsonl`, and
load the crop collection:

```bash
python -m app.ingestion.build_index
```

Reset the crop collection first or inspect all options with:

```bash
python -m app.ingestion.build_index --reset
python -m app.ingestion.build_index --help
```

To load an existing JSONL file directly:

```bash
crop-rag-ingest --input data/chunks.jsonl
```

Each non-empty JSONL record must contain `chunk_id` and `text`. Metadata is
optional and is sanitized before being written to Chroma. Existing IDs are
upserted.

### Company knowledge

Build the separate company collection from the Markdown source:

```bash
python -m app.ingestion.load_company_data
```

The default input is `data/aunkur_company_info.md`; the command writes
`data/company_chunks.jsonl` and loads `company_knowledge_base`. Use `--help`
for input, reset, and other command options.

For Compose, run ingestion inside the app container:

```bash
docker compose exec app python -m app.ingestion.loader \
  --input /app/data/chunks.jsonl
```

## API Usage

### Request

`POST /api/v1/chat` requires all three fields:

```json
{
  "session_id": "farmer-123",
  "message": "What is the seed rate for Boro Paddy?",
  "language_type": "english"
}
```

`language_type` must be `english` or `bangla`. `session_id` must be 1-200
characters and `message` must be 1-2,000 characters after trimming.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"What is the seed rate for Boro Paddy?","language_type":"english"}'
```

Use the same `session_id` for follow-up questions so the rewrite/checkpoint
logic can use the previous conversation context:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"How many times should it be irrigated?","language_type":"english"}'
```

### Response

```json
{
  "session_id": "farmer-123",
  "answer": "...",
  "language": "english",
  "rewritten_query": "...",
  "retrieval_mode": "dense_filtered",
  "sources": [
    {
      "chunk_id": "5_seed",
      "crop_name": "Boro Paddy",
      "section": "seed",
      "distance": 0.21
    }
  ],
  "messages": []
}
```

`distance` is a retrieval ranking value, not calibrated confidence. Chitchat
and meaningless requests may return no sources. The API also exposes
`GET /health` with the active chat model, embedding model, and environment.

## Tests and Linting

The tests are primarily unit and mocked integration tests and do not require
live model servers:

```bash
pytest -q
ruff check .
```

Live Ollama, vLLM, Groq, Chroma, Docker, and reranker connectivity should be
validated separately in the target deployment environment.

## Repository Layout

```text
src/app/main.py                         FastAPI application and health route
src/app/api/v1/endpoints/chat.py        Chat endpoint
src/app/schemas/chat.py                 API contracts
src/app/core/config.py                  Environment-backed settings
src/app/services/chat_service.py        API-to-graph service layer
src/app/services/pipeline/graph.py     LangGraph assembly and routing
src/app/services/pipeline/nodes/        Query, retrieval, and generation nodes
src/app/services/retrieval/             Chroma and retrieval helpers
src/app/ingestion/                      Crop and company indexing commands
data/crops.json                          Crop registry/source data
data/aunkur_company_info.md             Company knowledge source
data/chunks.jsonl                        Crop chunks
data/company_chunks.jsonl                Company chunks
ui/streamlit_app.py                     Streamlit client
tests/                                   Unit and integration tests
```

## Current Limitations

- The default in-memory checkpoint is not shared across processes or replicas.
- Crop questions require a reachable external reranker.
- Crop extraction currently computes matches but returns an empty crop list in
  the live node, so crop metadata filtering is not currently effective.
- BM25 and reciprocal-rank fusion helpers exist but are dormant; active crop
  retrieval is dense Chroma retrieval followed by reranking.
- The API does not add enforced citations, calibrated confidence, or a
  separate answer-verification pass.
- Romanized crop-name matching is not guaranteed by the deterministic matcher.
- The repository does not currently declare a license.

## Project Status

The core API, graph routing, ingestion workflows, retrieval components, and
UI are implemented. Production deployment still requires selecting and
operating compatible model services, a reranker, durable checkpoint storage,
and a deployment-specific validation process.