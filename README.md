# Crop RAG Chatbot

A history-aware crop-advisory question-answering API for farmers in Bangladesh.
The service accepts Bangla, English, and Banglish-style input, retrieves relevant
crop knowledge from Chroma, and generates a concise answer grounded in that
retrieved context.

This README documents the code that is currently connected and executable. Some
retrieval and orchestration capabilities are implemented in the repository but
are intentionally marked as deferred where they are not part of the live graph.

## 1. Problem Definition

Farmers need practical answers about seed rates, varieties, irrigation,
fertilizer, cultivation, pests, diseases, and harvesting. A general-purpose
language model can produce fluent answers, but fluency is not enough for
agricultural advice:

- Crop names and technical terms may be written in Bangla, English, or Latin
  characters representing Bangla pronunciation (Banglish).
- Follow-up questions such as "এতে কতবার সেচ দিতে হয়?" do not repeat the crop
  name and require conversation context.
- A model may invent rates, doses, dates, or treatment instructions when the
  relevant source information is missing.
- Passing a large crop record to a small model adds unrelated varieties and
  pest entries, making the correct fact harder to identify.

The project solves this with a retrieval-augmented generation (RAG) pipeline.
Crop data is cleaned, divided into small answerable fact units, embedded, and
stored in Chroma. At query time, the service normalizes the language, resolves
follow-up subjects, identifies crops from a canonical registry, retrieves
filtered context, and asks the chat model to answer only from that context.

## 2. High-Level Architecture

```text
Client
  |
  v
FastAPI: POST /api/v1/chat
  |
  v
ChatService
  |-- load in-process session history
  |-- serialize requests per session
  v
LangGraph
  normalize_language
        |
  rewrite_query
        |
  extract_crop
        |
  dense Chroma retrieval
        |
  grounded answer generation
        |
  persist the conversation turn
  |
  v
ChatResponse with answer, rewritten query, retrieval mode, and sources
```

### Current live graph

The checked-in graph is currently a linear workflow:

```text
START
  -> normalize_language
  -> rewrite_query
  -> extract_crop
  -> retrieve
  -> rerank
  -> filter_relevant
  -> generate
  -> END
```

`understand_query` is registered but is not connected to the compiled graph. The
conditional routing code around it is commented out, so the current API does not
perform an intent-based small-talk bypass. Reranking is active end to end.

## 3. End-to-End Request Workflow

### 3.1 Request validation

`ChatRequest` validates and trims both fields:

- `session_id`: required, nonblank, 1-200 characters.
- `message`: required, nonblank, 1-2,000 characters.

Invalid requests receive FastAPI/Pydantic validation errors before the pipeline
runs.

### 3.2 Language normalization

Bangla-script text is passed through unchanged. Latin-script input is treated as
Banglish and processed in two stages:

1. `avro.parse()` transliterates the Banglish text into Bangla script.
2. An Ollama spelling-correction model corrects spelling only.

The spelling-correction prompt explicitly prevents translation, paraphrasing,
answering the question, or changing the meaning. This design was chosen because
asking an LLM to directly convert Banglish was causing frequent hallucinations
and unintended rewrites. Avro provides a deterministic first pass, while the LLM
is limited to cleanup.

Current limitation: crop extraction matches canonical English and Bangla names,
not Romanized crop aliases. A Banglish sentence can be normalized successfully,
but Romanized crop-name handling is not guaranteed by the deterministic matcher.

### 3.3 History-aware query rewriting

The service keeps recent messages for each session. When a new question omits
its subject, the rewrite node uses the previous user question to create a
standalone query. For example:

```text
Previous: বোরো ধানের কোন জাতগুলো ভালো?
Current:  এর বীজ হার কত?
Rewritten: বোরো ধানের বীজ হার কত?
```

If the current question already names its crop, variety, disease, or topic, it
is kept as a standalone question instead of being mixed with the previous one.

### 3.4 Deterministic crop extraction

Crop names are matched against `data/crops.json` and the pipeline registry.
Matching uses normalized whole tokens and optional `bnltk` Bangla stemming. This
avoids false positives such as matching a short crop name inside an unrelated
word. Crop and section values are then available for Chroma metadata filtering.

### 3.5 Dense retrieval

The active retrieval path uses the `bge-m3` embedding model through Ollama and
Chroma. Results are filtered by detected crop and section where applicable. The
current response normally reports:

```json
{"retrieval_mode": "dense_filtered"}
```

### 3.6 Grounded generation

The generation prompt tells the selected chat model to:

- answer in natural Bangla or clear English;
- answer directly and concisely;
- use the supplied context as the factual source;
- avoid inventing agricultural facts, doses, rates, dates, or varieties; and
- say when the context does not contain enough information.

Retrieved content is truncated per chunk before being sent to the model. The
response includes source metadata, but the current prompt does not add numbered
citations to the natural-language answer.

### 3.7 History persistence

The user message, answer, and rewritten query are appended to an in-process
conversation store. Requests for the same session are serialized with an
`asyncio.Lock`, preventing follow-ups from reading stale history concurrently.
The store is not durable and is not shared between application replicas.

## 4. Knowledge Ingestion Workflow

The project supports both pre-built JSONL chunks and a crop-export-to-chunks
workflow.

### 4.1 Pre-built JSONL loader

```bash
crop-rag-ingest --input data/chunks.jsonl
```

Equivalent module command:

```bash
python -m app.ingestion.loader --input data/chunks.jsonl
```

Each non-empty line must be a JSON object containing `chunk_id` and `text`:

```json
{
  "chunk_id": "5_seed",
  "text": "ফসল (Crop): বোরো ধান (Boro Paddy)\n...",
  "metadata": {
    "crop_id": "5",
    "crop_name": "Boro Paddy",
    "crop_bangla_name": "বোরো ধান",
    "section": "seed"
  }
}
```

The loader reads UTF-8 JSONL, skips malformed or incomplete lines, sanitizes
metadata into Chroma-compatible scalar values, and upserts documents in batches.
Using the same `chunk_id` updates the existing document.

### 4.2 Crop export to chunks and Chroma

`app.ingestion.build_index` prepares source crop records, cleans HTML, creates
fact-unit chunks, writes inspectable JSONL, and loads the chunks into Chroma.
The command supports an optional reset and batch size:

```bash
python -m app.ingestion.build_index
python -m app.ingestion.build_index --reset
python -m app.ingestion.build_index --batch-size 64
```

The exact source and output arguments are defined by that module's CLI help:

```bash
python -m app.ingestion.build_index --help
```

### 4.3 Chunking design

The project deliberately does not use generic fixed-size text windows. One
crop record contains many independent topics and varieties, so combining them
would force a small model to choose the right value from a noisy document.

The chunker instead creates independently answerable fact units:

- one crop overview or crop-level section per chunk;
- one chunk per variety;
- one chunk per pest or disease entry, including its relevant chemicals;
- one chunk per herbicide entry where source data provides it; and
- summary chunks for supported child collections.

Every chunk starts with Bangla and English crop identity plus a section label.
Metadata includes fields such as `crop_id`, `crop_name`, `section`,
`variety_name`, or `disease_name`, allowing semantic retrieval and hard
metadata filtering to work together. HTML source fields are cleaned before
they become chunk text.

## 5. Technology Choices

| Technology | Role | Why it is used |
| --- | --- | --- |
| FastAPI | HTTP API and dependency injection | Typed, asynchronous API surface with automatic OpenAPI documentation |
| Uvicorn | ASGI server | Runs the FastAPI application locally or in a container |
| Pydantic / `pydantic-settings` | Request, response, and environment models | Validates contracts and centralizes configuration |
| LangGraph | Pipeline orchestration | Represents the request workflow as testable stateful nodes |
| LangChain Core | Messages and model abstractions | Provides common interfaces for chat messages and structured output |
| Ollama | Local chat and embedding provider | Keeps chat and `bge-m3` embeddings available through a local service |
| Groq | Optional chat provider | Allows hosted chat inference without changing the pipeline |
| Chroma | Vector database | Supports persistent local storage or a separate HTTP service |
| `bge-m3` | Embedding model | Produces multilingual semantic vectors for Bangla and English crop text |
| Avro | Banglish transliteration | Provides a deterministic first pass before LLM spelling correction |
| `bnltk` | Optional Bangla stemming | Helps match inflected Bangla crop names such as `গমের` to `গম` |
| `rank-bm25` | Lexical retrieval implementation | Provides a tested lexical fallback/fusion path for future activation |
| HTTPX | Reranker client | Sends retrieved documents to the external reranker service |
| BeautifulSoup | HTML cleanup | Converts HTML-heavy source fields into readable chunk text |
| Tenacity | Retry utilities | Supports resilient model/provider calls |
| Docker Compose | Local multi-service environment | Runs the API, Ollama, Ollama model initialization, and Chroma together |

## 6. Configuration

Settings are read from environment variables and `.env` through
`src/app/core/config.py`. Important values include:

```dotenv
CHAT_PROVIDER=ollama
OLLAMA_CHAT_MODEL=gpt-oss:20b-cloud
BANGLISH_CONVERTER_MODEL=gemma4:31b-cloud
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=bge-m3
CHROMA_HOST=
CHROMA_PORT=8000
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=crop_knowledge_base
CROP_REGISTRY_PATH=./data/crops.json
RETRIEVAL_TOP_K=5
HISTORY_MAX_TURNS=1
CONTEXT_MAX_CHARS_PER_CHUNK=3000
```

Environment variable names are case-insensitive through Pydantic settings. The
canonical Python setting is `banglish_converter_model`, exposed as
`BANGLISH_CONVERTER_MODEL` in the environment.

For Groq chat instead of Ollama chat:

```dotenv
CHAT_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_CHAT_MODEL=openai/gpt-oss-20b
```

Embeddings still use Ollama even when Groq is selected. The checked-in settings
default to Ollama. If this repository includes an `.env.example`, review its
values before copying them because provider/model defaults may differ from the
Python defaults.

## 7. Local Setup

Python 3.11 or newer is required.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Start Ollama and download the embedding model plus the selected chat model:

```bash
ollama pull bge-m3
ollama pull gpt-oss:20b-cloud
```

Run the API from the repository root:

```bash
uvicorn app.main:app --reload --app-dir src
```

The API is available at `http://localhost:8000`. FastAPI documentation is at
`http://localhost:8000/docs`, and the health endpoint is:

```bash
curl http://localhost:8000/health
```

Example health response:

```json
{
  "status": "ok",
  "chat_model": "gpt-oss:20b-cloud",
  "embed_model": "bge-m3",
  "env": "dev"
}
```

Before chatting, ingest the knowledge collection if it is not already present.
For a local embedded Chroma instance, the default persistence directory is
`./data/chroma`.

## 8. Docker Compose

```bash
docker compose up --build
```

Compose starts:

- `app`: API at `http://localhost:8000`;
- `ollama`: model server at `http://localhost:11434`;
- `ollama-init`: one-shot service that pulls `gemma3:4b` and `bge-m3`;
- `chroma`: Chroma at `http://localhost:8001` from the host.

Inside the Compose network, the app connects to Chroma as `chroma:8000`. The
host port is `8001` because Compose maps `8001:8000`. Named volumes preserve
Ollama models, Chroma data, and the Hugging Face cache.

To use Groq with Compose, provide `CHAT_PROVIDER=groq` and `GROQ_API_KEY` in
the environment before starting the services. The Compose app still needs
Ollama for embeddings and for the Banglish converter.

Ingest a mounted file inside the app container:

```bash
docker compose exec app python -m app.ingestion.loader \
  --input /app/data/chunks.jsonl
```

## 9. API Usage

### Bangla question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"বোরো ধানের বীজ হার কত?"}'
```

### English question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"What is the seed rate for Boro Paddy?"}'
```

### Banglish question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-456","message":"Boro dhan er bijer har koto?"}'
```

### Follow-up question

Reuse the same `session_id` so the rewrite node can resolve the omitted crop:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"এতে কতবার সেচ দিতে হয়?"}'
```

The response shape is:

```json
{
  "session_id": "farmer-123",
  "answer": "...",
  "language": "bn",
  "rewritten_query": "বোরো ধানের বীজ হার কত?",
  "retrieval_mode": "dense_filtered",
  "sources": [
    {
      "chunk_id": "5_seed",
      "crop_name": "Boro Paddy",
      "section": "seed",
      "score": 0.21
    }
  ]
}
```

Scores are retrieval distances/scores supplied by the active backend; they are
useful for inspection but should not be treated as calibrated confidence.

## 10. Project File Guide

### Root files

- `pyproject.toml`: canonical package metadata, dependencies, CLI entry point,
  pytest settings, and Ruff configuration.
- `requirements.txt`: secondary dependency artifact; `pyproject.toml` is the
  source of truth for installation.
- `Dockerfile`: Python application image and Uvicorn command.
- `docker-compose.yml`: API, Ollama, model initialization, and Chroma services.
- `BUILD_SUMMARY.md`: historical build notes, not the authoritative description
  of the current runtime graph.
- `data/crops.json`: canonical crop registry used for crop matching.
- `data/chunks.jsonl`: checked-in chunk data for ingestion.
- `data/chroma/`: local persistent Chroma files when embedded storage is used.

### Application bootstrap and core

- `src/app/main.py`: creates the FastAPI app, health endpoint, exception handler,
  and API router.
- `src/app/core/config.py`: cached Pydantic settings and model/provider values.
- `src/app/core/exceptions.py`: application error types and status handling.
- `src/app/core/logging.py`: logging configuration and logger factory.

### API and schemas

- `src/app/api/v1/router.py`: versioned API router registration.
- `src/app/api/v1/endpoints/chat.py`: `POST /api/v1/chat` endpoint.
- `src/app/schemas/chat.py`: request, response, and source-chunk models.
- `src/app/schemas/extraction.py`: structured query-rewrite/extraction models.
- `src/app/schemas/chunk_schema.py`: ingestion chunk contract.

### Ingestion

- `src/app/ingestion/loader.py`: validates JSONL and upserts documents to Chroma.
- `src/app/ingestion/build_index.py`: command-line index-building workflow.
- `src/app/ingestion/prepare_data.py`: crop-record loading and fact-unit chunking.
- `src/app/ingestion/html_cleaner.py`: removes markup and normalizes source text.

### Services and pipeline

- `src/app/services/chat_service.py`: joins API requests, session history, graph
  execution, and response conversion.
- `src/app/services/pipeline/graph.py`: builds and caches the LangGraph workflow.
- `src/app/services/pipeline/state.py`: typed state passed between nodes.
- `src/app/services/pipeline/registry.py`: canonical crop and section metadata.
- `normalize_language.py`: Bangla detection, Avro transliteration, and spelling
  correction.
- `rewrite_query.py`: follow-up subject resolution using recent history.
- `understand_query.py`: intent/language/section understanding module; currently
  registered but disconnected from the live graph.
- `extract_crop.py`: deterministic crop-name extraction and optional stemming.
- `retrieve.py`: calls the active semantic retriever.
- `rerank.py`: calls the external reranker service and orders chunks by relevance.
- `generate.py`: grounded final-answer generation.

### Models, retrieval, and memory

- `src/app/services/llm/client.py`: Ollama/Groq chat calls and structured output.
- `src/app/services/llm/embeddings.py`: Ollama embedding adapter.
- `src/app/services/retrieval/vector_store.py`: Chroma client and similarity search.
- `src/app/services/retrieval/hybrid.py`: dense retrieval plus dormant BM25/RRF
  helper implementation.
- `src/app/memory/store.py`: in-process conversation history implementation.

### Tests

The `tests/` directory contains unit tests for nodes, services, ingestion,
retrieval, reranking, memory, and API/graph integration. Tests generally use
mocks and doubles, so they do not prove that live Ollama, Groq, Chroma, Avro,
or the external reranker service is correctly configured.

## 11. Testing and Linting

Install the development extras, then run:

```bash
pytest -q
ruff check .
```

The test suite is intended to run without downloading or calling live models.
Integration with real model servers and containers should be tested separately
in the deployment environment.

## 12. Design Decisions and Current Status

### Hallucination mitigation

The project reduces hallucination risk through several layers:

1. Structured Pydantic models constrain model-produced extraction output.
2. Crop names come from a canonical registry rather than free-form model output.
3. Metadata filters limit retrieval to the relevant crop and section.
4. Fact-unit chunks reduce unrelated varieties and disease records in context.
5. Each chunk identifies its crop and section in its own text.
6. Context is truncated per chunk to control prompt size and noise.
7. The generation prompt requires grounded answers and disclosure of missing data.

This is grounding, not a formal guarantee. The current code does not enforce
citations, calibrated confidence thresholds, answer verification, or a separate
fact-checking pass.

### BM25 and RRF

BM25 tokenization, filtered-corpus caching, lexical search, and reciprocal-rank
fusion are implemented in `services/retrieval/hybrid.py`. They were tested as a
possible addition, but the current dense retrieval results did not justify the
extra complexity for the present dataset and workflow. The active method returns
dense Chroma results before the BM25/RRF branch, so BM25 is currently dormant.

This can be revisited if exact keyword matching, rare crop terms, or larger
corpora make lexical retrieval valuable.

### Reranking

The rerank node posts the query and retrieved document text to `RERANKER_URL`,
maps the returned indexes and relevance scores back to the chunks, sorts them,
and keeps `RERANK_TOP_K` chunks. It is active immediately after retrieval.

### Production considerations

- Replace the in-memory conversation store with Redis, PostgreSQL, or another
  durable shared store before running multiple replicas.
- If BM25 is enabled later, consider a versioned prebuilt lexical index for
  larger datasets instead of rebuilding filtered corpora in each process.
- Keep the external reranker service reachable from every API replica.
- Pin Docker image digests and validate compatible Ollama, Chroma, and model
  versions before production deployment.
- Treat retrieval scores as ranking signals, not answer confidence.

### Feature status

| Capability | Status |
| --- | --- |
| FastAPI chat endpoint | Active |
| Bangla input handling | Active |
| Avro plus LLM Banglish normalization | Active, with Romanized crop-name limitations |
| History-aware follow-up rewriting | Active |
| Canonical crop extraction | Active |
| Dense `bge-m3` Chroma retrieval | Active |
| Grounded generation | Active |
| In-process session history | Active, not horizontally scalable |
| Intent-based routing and small-talk bypass | Implemented but disconnected |
| BM25 and reciprocal-rank fusion | Implemented but dormant |
| External cross-encoder reranking | Active |
| Citation insertion and answer verification | Not implemented |

## License and project maturity

The repository does not currently define a license in its root metadata. Review
and add the appropriate license before distributing the project. The codebase
has useful unit and integration coverage, but live-provider, container, model
download, and production-scale persistence behavior still require deployment
validation.
