import httpx
import numpy as np
from rank_bm25 import BM25Okapi
from app.services.retrieval.vector_store import get_vector_store
import re
import unicodedata

_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[a-z0-9]+")
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200c\u200d\u200b\ufeff"))

def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "").translate(_ZERO_WIDTH).translate(_BN_DIGITS).lower()
    return _TOKEN_RE.findall(text)


vectorstore = get_vector_store()

query = "হীরা ২ কী ধরনের ধানের জাত?"

# ---------------- Semantic Search ----------------
semantic_results = vectorstore.similarity_search_with_score(query, k=10)

semantic_docs = [doc for doc, _ in semantic_results]


# ---------------- BM25 ----------------
raw = vectorstore.get(include=["documents", "metadatas"])
corpus_docs = list(zip(raw["ids"], raw["documents"], raw["metadatas"]))

bm25 = BM25Okapi([tokenize(text or "") for _, text, _ in corpus_docs])
scores = bm25.get_scores(tokenize(query))
top = np.argsort(scores)[::-1][:10]

bm25_docs = []

for i in top:
    cid, text, meta = corpus_docs[i]

    # Recreate a Document-like object using the same structure
    from langchain_core.documents import Document

    bm25_docs.append(
        Document(
            page_content=text or "",
            metadata=meta or {},
        )
    )


# ---------------- Combine Semantic + BM25 ----------------
# Remove duplicates using chunk_id when available
combined_docs = []
seen = set()

for doc in semantic_docs + bm25_docs:
    chunk_id = doc.metadata.get("chunk_id")

    # Fall back to page content if chunk_id is unavailable
    unique_key = chunk_id or doc.page_content

    if unique_key not in seen:
        seen.add(unique_key)
        combined_docs.append(doc)


# Should normally be <= 20
combined_docs = combined_docs[:20]

documents = [doc.page_content for doc in combined_docs]


# ---------------- Reranker ----------------
payload = {
    "query": query,
    "documents": documents,
}

response = httpx.post(
    "http://localhost:8090/rerank",
    json=payload,
    timeout=30.0,
)

response.raise_for_status()
data = response.json()

reranked = []

for result in data["results"]:
    index = result["index"]
    score = result["relevance_score"]

    doc = combined_docs[index]

    reranked.append(
        (doc, score)
    )


reranked.sort(
    key=lambda x: x[1],
    reverse=True,
)


# ---------------- Reranker TOP 15 ----------------
for doc, score in reranked[:15]:
    print("=" * 80)
    print(f"Score: {score:.4f}")
    print(f"Chunk: {doc.metadata.get('chunk_id')}")
    print(doc.page_content[:300])