import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

SYSTEM_PROMPT = """
Convert the user's Banglish or Bangla-English mixed sentence into natural
Bangla written in Bengali Unicode.

Rules:
- Preserve the original meaning exactly.
- Convert Romanized Bengali into proper Bangla.
- Convert English words into Bangla when naturally possible.
- Keep technical terms such as pH, NPK, TSP, numbers and units unchanged.
- Do not answer the question.
- Do not explain anything.
- Return only the converted Bangla sentence.
""".strip()


def convert_to_bangla(text: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
    )

    return response.content.strip()


if __name__ == "__main__":
    while True:
        text = input("> ").strip()

        if text.lower() in {"exit", "quit"}:
            break

        if text:
            print(convert_to_bangla(text))