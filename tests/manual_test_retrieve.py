import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings

settings = get_settings()

embedding_model = settings.embed_model
collection_name = settings.chroma_collection
persist_directory = settings.chroma_persist_dir

embeddings = OllamaEmbeddings(model=embedding_model)

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

query = "বোরো ধানে চারা থেকে চারার দূরত্ব কত?"

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
