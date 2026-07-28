from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3")

vectorstore = Chroma(embedding_function=embeddings, persist_directory="./data/chroma")

query = "What is the seed rate of boro paddy?"

docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, start=1):
    print(f"\n=== Result {i} ===")
    print(doc.page_content)
    print("Metadata:", doc.metadata)