"""
Test: MultiQueryRetriever (default prompt) + Reranker
"""

import logging
import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.llm.client import get_chat_llm

# ── Show the decomposed queries ────────────────────────────────────
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# fallback logger name in some versions
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

settings = get_settings()

# ── Embeddings + Vectorstore ───────────────────────────────────────
embeddings = OllamaEmbeddings(model=settings.embed_model)

vectorstore = Chroma(
    collection_name=settings.chroma_collection,
    embedding_function=embeddings,
    persist_directory=settings.chroma_persist_dir,
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

llm = get_chat_llm(temperature=0.0)

# ── Default prompt (no custom prompt) ──────────────────────────────
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,          # also keep the original query
)

# ── Test Query ─────────────────────────────────────────────────────
query = "বোরো ধানে চারা থেকে চারার দূরত্ব কত?"
# query = "bhutta chara ki ki jaat chash korle bhalo folon pawa jay"

print(f"Original Query: {query}\n")

# ── 1. Multi-query retrieval ───────────────────────────────────────
docs: list[Document] = multi_retriever.invoke(query)

print(f"\nRetrieved {len(docs)} unique documents after multi-query\n")

# ── 2. Rerank ──────────────────────────────────────────────────────
documents_text = [doc.page_content for doc in docs]

payload = {
    "query": query,
    "documents": documents_text,
}

response = httpx.post(
    settings.reranker_url,
    json=payload,
    timeout=30.0,
)
response.raise_for_status()
data = response.json()

reranked = []
for result in data["results"]:
    index = result["index"]
    score = result["relevance_score"]
    doc = docs[index]
    reranked.append((doc, score))

reranked.sort(key=lambda x: x[1], reverse=True)

# ── 3. Print top results ───────────────────────────────────────────
print("Top 10 after reranking:\n")
for doc, score in reranked[:10]:
    print("=" * 80)
    print(f"Score   : {score:.4f}")
    print(f"Chunk   : {doc.metadata.get('chunk_id')}")
    print(f"Crop    : {doc.metadata.get('crop_name') or doc.metadata.get('crop')}")
    print(doc.page_content[:350])
    print()