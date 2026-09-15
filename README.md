# Crop RAG Chatbot

This repository contains a crop-advisory chat application for Bangla and English queries. It exposes a FastAPI API, a Streamlit interface, a LangGraph orchestration pipeline, and Chroma-backed retrieval over crop and company knowledge.

The application answers questions about crop production, pests, diseases, fertilizer, irrigation, climate, and company information using grounded retrieval and LLM generation. The generated answer should still be reviewed before making important agronomic or business decisions.

## What the project does

- Serves a chat API at `POST /api/v1/chat` and a streaming endpoint at `POST /api/v1/chat/stream`
- Uses a LangGraph pipeline to rewrite follow-up questions, route intent, retrieve documents, rerank, compress, and generate answers
- Supports `crop_query`, `company_query`, `chitchat`, and `meaningless` intents
- Stores crop and company knowledge in separate Chroma collections
- Reuses the same session ID as the LangGraph thread ID for multi-turn context
- Runs a Streamlit UI for local testing and demos
- Loads crop/company data from JSON, Markdown, and JSONL ingestion stages

## Architecture

```text
HTTP client
   |
   v
FastAPI app (/api/v1/chat)
   |
   v
LangGraph workflow
   |
   +--> rewrite_query
   |
   +--> route
          |
          +--> crop_query -> extract_crop -> retrieve -> compress_chunk -> generate
          +--> company_query -> retrieve_company -> generate_company
          +--> chitchat -> generate_chitchat
          +--> meaningless -> generate_meaningless
```

The graph is assembled in `src/app/services/pipeline/graph.py` and uses `session_id` as the LangGraph thread key. The default checkpoint is an in-memory `MemorySaver`, with Redis support available when configured.

## Runtime stack

- Python 3.11+
- FastAPI for the HTTP API
- Streamlit for the local web UI
- LangGraph for orchestration
- LangChain components for model wrappers, structured output, embeddings, and document compression
- Chroma for vector search
- Ollama for local embeddings and optional chat generation
- vLLM or Groq as alternative LLM providers
- External reranker service for crop Q&A ranking

## Project structure

```text
src/app/main.py                     FastAPI application and health endpoint
src/app/api/v1/router.py           API router registration
src/app/api/v1/endpoints/chat.py   Chat and streaming endpoints
src/app/core/config.py             Environment-backed settings
src/app/services/chat_service.py   API-to-graph orchestration layer
src/app/services/pipeline/         LangGraph pipeline and retrieval nodes
src/app/services/retrieval/        Chroma and hybrid retrieval helpers
src/app/ingestion/                 Crop and company indexing scripts
data/                             Local data and vector store storage
ui/streamlit_app.py                Streamlit front-end
tests/                            Unit and integration tests
```

## Local setup

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

If you want to run the Streamlit app in isolation, install the UI dependency set from the project package metadata or the generated requirements file.

## Configuration

Create a `.env` file in the repository root to override the defaults. The project reads settings from `src/app/core/config.py`.

```dotenv
APP_ENV=dev
CHAT_PROVIDER=ollama
OLLAMA_CHAT_MODEL=gemma4:31b-cloud
OLLAMA_BASE_URL=http://localhost:11434
BANGFISH_CONVERTER_MODEL=gemma4:12b
VLLM_CHAT_MODEL=gemma4:12b
VLLM_BASE_URL=http://localhost:8091/v1
VLLM_API_KEY=not-needed
GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=
EMBED_MODEL=bge-m3:latest
CHROMA_HOST=
CHROMA_PORT=8000
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=crop_knowledge_base
CHROMA_COMPANY_COLLECTION=company_knowledge_base
CROP_REGISTRY_PATH=./data/crops.json
GRAPHQL_ENDPOINT=https://aunkur-backend-311104304042.us-central1.run.app/graphql
RETRIEVAL_TOP_K=20
COMPANY_RETRIEVAL_TOP_K=3
RERANK_TOP_K=6
RERANKER_URL=http://localhost:8090/rerank
HISTORY_MAX_TURNS=1
CONTEXT_MAX_CHARS_PER_CHUNK=3000
LLM_TEMPERATURE=0
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=crop-rag-chatbot
CHECKPOINT_BACKEND=memory
REDIS_URL=redis://localhost:6379/0
```

Accepted values for `CHAT_PROVIDER` are `ollama`, `groq`, and `vllm`. The default project setup is `ollama`, with embeddings served through Ollama and an optional external reranker used by cropping queries.

## Run the API locally

Start the backing services first, especially Ollama and Chroma if they are not already running.

```bash
uvicorn app.main:app --reload --app-dir src
```

The API listens on `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Example health check:

```bash
curl http://localhost:8000/health
```

## Run the Streamlit UI

In a second terminal:

```bash
streamlit run ui/streamlit_app.py
```

The UI posts to `http://localhost:8000/api/v1/chat/stream` by default.

## API contract

### Request

```json
{
  "session_id": "farmer-123",
  "message": "What is the seed rate for Boro Paddy?",
  "language_type": "english"
}
```

`language_type` must be either `english` or `bangla`.

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

## Build and ingest data

### Crop registry

Refresh the local crop registry from the upstream GraphQL endpoint:

```bash
python -m app.ingestion.fetch_crops
python -m app.ingestion.fetch_crops --updated-within-days 7
```

Build the crop chunks and load the crop Chroma collection:

```bash
python -m app.ingestion.build_index
```

To reset the crop collection before rebuild:

```bash
python -m app.ingestion.build_index --reset
```

You can also ingest an existing JSONL dump directly:

```bash
crop-rag-ingest --input data/chunks.jsonl
```

### Company knowledge

Build the separate company collection from the Markdown file:

```bash
python -m app.ingestion.load_company_data
```

The default source is `data/aunkur_company_info.md`, which produces `data/company_chunks.jsonl` and loads the company Chroma collection.

## Validation

Run the project test suite from the repository root:

```bash
pytest -q
```

Optional linting:

```bash
ruff check .
```

## Notes

- The default checkpoint is process-local memory and is not shared across multiple API instances.
- Crop retrieval depends on a reachable reranker service at `RERANKER_URL`.
- This project is designed for a local or self-hosted deployment and should be validated against the target model and vector store environment before production use.
- The app currently has no license metadata declared in the repository.