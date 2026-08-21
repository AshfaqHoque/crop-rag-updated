import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

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

# def tokenize(text: str):
#     return re.findall(r"\w+", text.lower())
# data = vectorstore.get()
# docs = [Document(page_content=text, metadata=meta) for text, meta in zip(data["documents"], data["metadatas"])]
# tokenize_docs =[tokenize(doc.page_content) for doc in docs]
# bm25 = BM25Okapi(tokenize_docs)

query = "বোরো ধানের কী কী জাত আছে?"

semantic_results = vectorstore.similarity_search_with_score(query, k=20)

semantic_docs = [doc for doc, _ in semantic_results]

pairs = [
    (query, doc.page_content)
    for doc in semantic_docs
]

scores = reranker.predict(pairs)

reranked = sorted(
    zip(semantic_docs, scores),
    key=lambda x: x[1],
    reverse=True,
)

print("Reranked Results")
for doc, score in reranked[:10]:
    print("=" * 80)
    print(f"Score: {score:.4f}")
    print(f"Chunk: {doc.metadata['chunk_id']}")
    print(doc.page_content[:100])

# print("bm255555555")
# bm25_results =  bm25.get_scores(tokenize(query))
# results = sorted(
#     zip(docs, bm25_results),
#     key=lambda x: x[1],
#     reverse=True,
# )
# best_score = max(bm25_results)
# threshold = best_score * 0.75
# for doc, score in results[:10]:
#     if score < threshold:
#         continue

#     print("=" * 80)
#     print(f"Score    : {score:.4f}")
#     print(f"Chunk ID : {doc.metadata['chunk_id']}")
#     print(doc.page_content[:100])