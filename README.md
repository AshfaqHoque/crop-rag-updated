# AUNKUR AI Chatbot

This repository contains a crop-advisory chat application for Bangla and English queries. It exposes a FastAPI API, a Streamlit interface, a LangGraph orchestration pipeline, and Chroma-backed retrieval over crop and company knowledge.

The application answers questions about crop production, pests, diseases, fertilizer, irrigation, climate, and company information using grounded retrieval and LLM generation. The generated answer should still be reviewed before making important agronomic or business decisions.

## What the project does

- Serves a chat API at `POST /api/v1/chat` and a streaming endpoint at `POST /api/v1/chat/stream`
- Uses a LangGraph pipeline to rewrite follow-up questions, route intent, retrieve documents, rerank, compress, and generate answers
- Supports `crop_query`, `soil_test_query`, `company_query`, `chitchat`, and `meaningless` intents
- Stores crop and company knowledge in separate Chroma collections
- Stores Porokh and general soil-test FAQs in the `soil_test_knowledge_base` collection
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
          +--> soil_test_query -> retrieve_soil_test -> generate_soil_test
          +--> company_query -> retrieve_company -> generate_company
          +--> chitchat -> generate_chitchat
          +--> meaningless -> generate_meaningless
```

The graph is assembled in `src/app/services/pipeline/graph.py` and uses `session_id` as the LangGraph thread key. The default checkpoint is an in-memory `MemorySaver`, with Redis support available when configured.

## How it works

The project uses local files for source data and the VM for model and vector services.

### Ingestion flow

1. An ingestion command reads source files from the local `data/` folder.
  - Crop data comes from `data/crops.json`.
  - Company data comes from `data/aunkur_company_info.md`.
  - Soil-test data comes from the FAQ CSV files.
2. The files are split into small text chunks with metadata such as crop, section, and source.
3. The local app sends each chunk to the VM's Ollama service through the SSH tunnel.
4. Ollama creates an embedding vector for each chunk.
5. The app sends the text, metadata, and vectors to Chroma running on the VM.
6. Chroma stores the vectors in its VM disk directory and makes them available for search.

The JSONL files in `data/` are inspectable chunk dumps. They are useful for checking the generated chunks, but Chroma is the database used for retrieval.

### Question-answering flow

1. A user sends a question through the Streamlit UI or the FastAPI API.
2. The pipeline rewrites follow-up questions and identifies the request type.
3. The router classifies soil-testing questions separately from crop questions. Soil-test questions search the dedicated `soil_test_knowledge_base` Chroma collection; crop and company questions search their matching collections.
4. The most relevant results are filtered, reranked, and compressed into context.
5. The selected context is sent to the configured chat model on the VM.
6. The API returns the answer together with the rewritten question and source information.

### Connection flow

The SSH tunnel makes VM services look like local services:

```text
Local app -> localhost:8001 -> SSH tunnel -> VM Chroma:8000
Local app -> localhost:11434 -> SSH tunnel -> VM Ollama:11434
Local app -> localhost:8091 -> SSH tunnel -> VM vLLM:8091
Local app -> localhost:8090 -> SSH tunnel -> VM reranker:8090
```

With `CHROMA_HOST=127.0.0.1` and `CHROMA_PORT=8000`, the app uses remote Chroma. It does not read vectors from local `data/chroma`; that directory is used only when `CHROMA_HOST` is empty. The source documents and ingestion scripts still remain local.

When the app runs in Docker Compose, it uses `host.docker.internal` to reach the same forwarded host ports. The Compose stack starts the app, UI, and Redis, but Chroma, Ollama, vLLM, and the reranker remain on the VM.

### Collections

The data is stored in separate Chroma collections:

- `crop_knowledge_base` for crop information
- `company_knowledge_base` for company information
- `soil_test_knowledge_base` for soil-test and Porokh FAQs

Running an ingestion command again updates existing chunks with the same IDs. Use `--reset` when you want to delete the selected collection and rebuild it from the current local source files.

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
CHAT_PROVIDER=vllm
OLLAMA_CHAT_MODEL=gemma4:31b-cloud
OLLAMA_BASE_URL=http://localhost:11434
VLLM_CHAT_MODEL=gemma4:12b
VLLM_BASE_URL=http://localhost:8091/v1
VLLM_API_KEY=not-needed
GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=
EMBED_MODEL=bge-m3:latest
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=crop_knowledge_base
CHROMA_COMPANY_COLLECTION=company_knowledge_base
CHROMA_SOIL_TEST_COLLECTION=soil_test_knowledge_base
SOIL_TEST_RETRIEVAL_TOP_K=3
CROP_REGISTRY_PATH=./data/crops.json
GRAPHQL_ENDPOINT=https://aunkur-backend-311104304042.us-central1.run.app/graphql
RETRIEVAL_TOP_K=20
COMPANY_RETRIEVAL_TOP_K=3
RERANK_TOP_K=6
RERANKER_URL=http://localhost:8090/rerank
CONTEXT_MAX_CHARS_PER_CHUNK=3000
LLM_TEMPERATURE=0
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=aunkur-chat-api
CHECKPOINT_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

Accepted values for `CHAT_PROVIDER` are `ollama`, `groq`, and `vllm`. The default project setup is `vllm`, with embeddings served through Ollama and an optional external reranker used by cropping queries.

## Run the API locally

Start the backing services first, especially Ollama and Chroma if they are not already running.

```bash
uvicorn app.main:app --reload --app-dir src --port 8052 
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

## Vector store commands

The default local configuration uses embedded Chroma and persists its data in
`data/chroma`. Make sure the embedding model is available before ingesting:

```bash
ollama serve
ollama pull bge-m3
```

The Docker stack expects the model services to be available through the SSH
forwarded host ports shown below. Start the tunnel before starting Compose:

```bash
ssh -N -L 8091:127.0.0.1:8091 -L 8090:127.0.0.1:8090 -L 11434:127.0.0.1:11434 -L 8000:127.0.0.1:8000 ashfaq@34.24.59.50
```
ssh -i ~/.ssh/id_rsa ashfaq@34.24.59.50
Port `8091` provides vLLM (`gemma4:12b`), port `8090` provides the reranker,
port `11434` provides Ollama embeddings, and port `8001` provides VM Chroma.
The full Docker stack can then be started with `docker compose up -d`.

The stack also starts the Streamlit UI. Open the API at
`http://localhost:8000/docs` and the UI at `http://localhost:8501`.

The Docker configuration uses only the host-forwarded vLLM, reranker, Ollama
embedding, and Chroma services. Compose does not start a local Chroma or Ollama
container, so it cannot reset the VM collections during startup.

The ingestion commands reset individual collections only when `--reset` is
provided:

```bash
docker compose exec app python -m app.ingestion.build_index --reset
docker compose exec app python -m app.ingestion.load_company_data --reset
docker compose exec app python -m app.ingestion.load_soil_test_data --reset
```

These commands delete collections in the VM because Docker is configured to use
the forwarded VM Chroma endpoint.

Run the following commands from the repository root.

### Ingestion collections (reset + rebuild)

Run the three ingestion commands together to reset and reload all collections:

```bash
python -m app.ingestion.build_index --fetch --reset
python -m app.ingestion.load_company_data --reset
python -m app.ingestion.load_soil_test_data --reset
```

For crops, `--fetch` calls the GraphQL crop service and keeps the returned crop
rows in memory. It does not write `data/crops.json`. The same command then
creates chunks, generates embeddings through the VM Ollama service, and stores
the vectors in the VM Chroma collection. Without `--fetch`, `build_index` uses
the existing local `data/crops.json` file.

### Crop collection

Fetch crops from the upstream GraphQL endpoint and save the response locally:

```bash
python -m app.ingestion.fetch_crops
python -m app.ingestion.fetch_crops --updated-within-days 7
```

Build chunks from `data/crops.json` and load the `crop_knowledge_base` Chroma
collection:

```bash
python -m app.ingestion.build_index
```

Fetch the latest crops and rebuild the collection directly without saving a
local JSON registry:

```bash
python -m app.ingestion.build_index --fetch --reset
```

Reset the crop collection before rebuilding it:

```bash
python -m app.ingestion.build_index --reset
```

Load an existing crop JSONL dump without rebuilding chunks:

```bash
python -m app.ingestion.loader --input data/chunks.jsonl
crop-rag-ingest --input data/chunks.jsonl
```

Inspect chunk sizes before ingesting:

```bash
python -m app.ingestion.analyze_chunks --input data/chunks.jsonl
python -m app.ingestion.analyze_chunks --input data/chunks.jsonl --top 20
```

### Company collection

Build the company collection from `data/aunkur_company_info.md`:

```bash
python -m app.ingestion.load_company_data
```

Reset the company collection before loading it:

```bash
python -m app.ingestion.load_company_data --reset
```

Use another Markdown file or several files:

```bash
python -m app.ingestion.load_company_data --input data/aunkur_company_info.md
python -m app.ingestion.load_company_data --input data/company_one.md data/company_two.md
```

The command writes the inspectable chunk dump to `data/company_chunks.jsonl`
and loads the `company_knowledge_base` collection.

### Soil-test collection

Load the default soil-test FAQ CSV into the `soil_test_knowledge_base`
collection:

```bash
python -m app.ingestion.load_soil_test_data
```

Load multiple FAQ files into the same collection, resetting it first:

```bash
python -m app.ingestion.load_soil_test_data \
  --input data/porokh_faq.csv data/soil_test_faq.csv \
  --reset
```

The command writes the inspectable chunk dump to
`data/soil_test_chunks.jsonl`.

The two CSV sources cover different parts of the soil-testing experience:

- `data/porokh_faq.csv` contains Porokh device and service FAQs.
- `data/soil_test_faq.csv` contains general soil sampling, testing, and report FAQs.

Both sources are indexed into `soil_test_knowledge_base`. The same collection is
used for questions about soil sampling, test frequency, sample depth and amount,
wet or fertilized soil, pH/EC and nutrient interpretation, soil-test-based
fertilizer recommendations, and the Porokh service. Soil-test retrieval uses up
to `SOIL_TEST_RETRIEVAL_TOP_K` results (default `3`) before answer generation.

Example questions:

```text
মাটি পরীক্ষা কেন করব?
How deep should I collect a soil sample?
How long does a Porokh soil test report take?
What nutrients does the Porokh device test?
```

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

ssh -N -L 8091:127.0.0.1:8091 -L 8090:127.0.0.1:8090 -L 11434:127.0.0.1:11434 -L 8001:127.0.0.1:8000 imtiazhossain@35.243.250.134


## Docker Image push to GCloud Artifact Registry

gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=upheld-setting-423215-p7 \
  .

## Artifact Registry run deploy

gcloud run deploy chat-bot-api \
--image=us-central1-docker.pkg.dev/upheld-setting-423215-p7/development/chat-bot-api:latest \
--port=8000 \
--region=us-central1 \
--allow-unauthenticated \
--platform managed

## Azure Deploy

az containerapp up \
  --name aunkur-chat-api \
  --resource-group aunkur-ai-rg \
  --environment aunkur-ai-env \
  --source . \
  --ingress external \
  --target-port 8000

## Update 
az containerapp update \
  --name aunkur-chat-api \
  --resource-group aunkur-ai-rg \
  --source .


sudo systemctl restart app-8080.service \
journalctl -u app-8080.service -f
