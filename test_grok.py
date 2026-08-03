import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("XAI_API_KEY"):
    raise RuntimeError("XAI_API_KEY was not loaded")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY was not loaded")

# from langchain_xai import ChatXAI

# llm = ChatXAI(model="grok-4", temperature=0)
# response = llm.invoke("what is the capital of France?")
# print(response.content)

from langchain_groq import ChatGroq

llm2 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)
response = llm2.invoke("what is the capital of France?")
print(response.content)
