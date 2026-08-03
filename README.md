# Crop RAG Chatbot — Steps 1–9

A history-aware crop advisory RAG API for Bangla and English, built with FastAPI,
LangGraph, Ollama, optional Groq, Chroma, BM25, and `BAAI/bge-reranker-v2-m3`.

## Implemented pipeline

1. Project scaffold, settings, logging, health check
2. Selectable Ollama/Groq chat clients, Ollama `bge-m3` embeddings, Chroma, ingestion CLI
3. Combined language/intent/crop/section extraction
4. History-aware standalone-query rewrite for subject resolution
5. Metadata-filtered retrieval:
   - crop detected: dense `bge-m3` retrieval through Chroma
   - no crop detected: dense + BM25, fused with reciprocal rank fusion
   - crop and section filters are applied before both retrieval paths
6. `BAAI/bge-reranker-v2-m3` cross-encoder reranking and grounded generation
7. End-to-end LangGraph wired to `POST /api/v1/chat`
8. Unit tests for each node/service plus full graph and API integration tests
9. Dockerfile and Docker Compose for app + Ollama + Chroma

## Pipeline flow

```text
request
  -> load session history
  -> rewrite_query
  -> understand_query
  -> [crop query?]
       yes -> hybrid retrieve -> rerank -> generate
       no  -> generate directly
  -> persist user/assistant turn
  -> response
```

The current history store is deliberately isolated behind an interface but is in-process memory. Replace it with Redis/PostgreSQL before running multiple app replicas.

## Local setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev,reranker]"
cp .env.example .env
```

Ollama is the default chat provider. Configure `.env` like this:

```dotenv
CHAT_PROVIDER=ollama
OLLAMA_CHAT_MODEL=gemma3:4b
```

To use Groq instead for chat calls, change only these values:

```dotenv
CHAT_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_CHAT_MODEL=openai/gpt-oss-120b
```

Ollama remains available for local chat and is always used for embeddings:

```bash
ollama pull gemma3:4b
ollama pull bge-m3
```

Run the API:

```bash
uvicorn app.main:app --reload --app-dir src
```

Health check:

```bash
curl http://localhost:8000/health
```

## Ingest crop chunks

```bash
crop-rag-ingest --input path/to/chunks.jsonl
# Equivalent: python -m app.ingestion.loader --input path/to/chunks.jsonl
```

Expected JSONL shape:

```json
{"chunk_id":"5_seed","text":"...","metadata":{"crop_id":"5","crop_name":"Boro Paddy","section":"seed"}}
```

`crop_name` and `section` metadata must exactly match `data/crops.json` and `services/pipeline/registry.py`.

## Chat request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"বোরো ধানের বীজ হার কত?"}'
```

A follow-up can use the same session:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"farmer-123","message":"এতে কতবার সেচ দিতে হয়?"}'
```

Response fields include the answer, detected language, standalone rewritten query, retrieval mode, and source chunks.

## Docker Compose

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Ollama: `http://localhost:11434`
- Chroma: `http://localhost:8001`

Set `CHAT_PROVIDER=groq` and `GROQ_API_KEY` before starting Compose when Groq is
desired. Otherwise, Compose uses Ollama. The one-shot `ollama-init` service pulls
`gemma3:4b` and `bge-m3`; the reranker is downloaded when first loaded.

Ingest inside Docker:

```bash
docker compose exec app python -m app.ingestion.loader --input /app/data/your_chunks.jsonl
```

## Tests and lint

The test suite mocks all live model calls; running Groq, Ollama, and the reranker
model are not required.

```bash
pytest -q
ruff check .
```

## Important production notes

- Keep `RERANKER_FAIL_OPEN=true` if availability is more important than reranker strictness. Set it to `false` when a missing reranker must fail the request.
- `RERANKER_USE_FP16=true` is suitable on supported GPU deployments; keep it false for CPU.
- BM25 caches each metadata-filtered corpus in-process for `BM25_CACHE_TTL_SECONDS`. For a very large corpus, move this to a versioned, prebuilt lexical index and refresh it after ingestion.
- Use Redis or a durable conversation store for horizontal scaling.
- Pin Docker image digests in your deployment environment after validating compatible Ollama and Chroma versions.
