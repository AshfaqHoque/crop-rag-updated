# Build Summary — Steps 4–9

## Delivered

- History-aware query rewriting for follow-up subject resolution
- Combined rewrite-first LangGraph pipeline
- Metadata-filtered dense retrieval with no-crop BM25 fallback and reciprocal-rank fusion
- HTTP reranking through the external reranker service
- Grounded generation with numbered source references
- Session-aware chat service and `POST /api/v1/chat`
- Node, service, graph, and API tests
- Dockerfile and Docker Compose for app, Ollama, model initialization, and Chroma
- Local and remote Chroma support
- Ingestion console command: `crop-rag-ingest`

## Verification performed

- 42 tests passed using isolated test doubles for external model/database packages
- All source and test files compiled successfully
- `pyproject.toml` and `docker-compose.yml` parsed successfully
- Python files satisfy the configured 100-character line limit

## Environment limitations during verification

The execution environment could not download the real dependencies because its package mirror returned HTTP 503 responses, and it did not provide a Docker daemon. Therefore, live Ollama, Chroma, the external reranker, full dependency installation, and `docker compose up` were not executed here. The project is configured for those services and includes the commands required to run them in a normal development or deployment environment.
