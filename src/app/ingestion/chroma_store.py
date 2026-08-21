"""
Persistent local vector store using Chroma. We compute embeddings
ourselves (via OllamaEmbedder) and pass them in explicitly -- we do NOT
use Chroma's built-in embedding_function, so there is never ambiguity
about which model produced which vector.
"""
import chromadb

from app.core.config import get_settings
from app.schemas.chunk_schema import Chunk

settings = get_settings()

class ChromaStore:
    def __init__(self, persist_dir: str = str(settings.chroma_persist_dir), collection_name: str = settings.chroma_collection):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 200):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_emb = embeddings[i : i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[_clean_metadata(c.metadata) for c in batch],
                embeddings=batch_emb,
            )

    def query(self, embedding: list[float], n_results: int = 10, where: dict | None = None):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()


def _clean_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- strip Nones."""
    return {k: v for k, v in meta.items() if v is not None}