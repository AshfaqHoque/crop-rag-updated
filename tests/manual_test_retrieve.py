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

query = "ব্রি ধান৮৭"
result = vectorstore.similarity_search_with_score(query, k=10)
for doc, score in result:
    print("-----score---- ", score)
    print(doc.metadata.get("chunk_id"))
    print(doc.page_content[:100])
    
    
