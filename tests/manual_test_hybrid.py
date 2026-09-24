import httpx

from app.services.retrieval.vector_store import get_vector_store

vectorstore = get_vector_store()

query = "লবণাক্ত জমির জন্য ব্রি ধান ৪৭ ও আলোড়নের মধ্যে কী পার্থক্য আছে?"

semantic_results = vectorstore.similarity_search_with_score(query, k=20)

semantic_docs = [doc for doc, _ in semantic_results]

documents = [doc.page_content for doc in semantic_docs]

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

    doc = semantic_docs[index]

    reranked.append(
        (doc, score)
    )
    
reranked.sort(
    key=lambda x: x[1],
    reverse=True,
)


for doc, score in reranked[:10]:
    print("=" * 80)
    print(f"Score: {score:.4f}")
    print(f"Chunk: {doc.metadata.get('chunk_id')}")
    print(doc.page_content[:300])
