import unicodedata

import avro
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

settings = get_settings()

if settings.chat_provider == "vllm":
    _banglish_llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        model=settings.vllm_chat_model,
        api_key=settings.vllm_api_key,
        temperature=0,
        top_p=0.9,
        extra_body={
            "top_k": 30,
        },
    )
elif settings.chat_provider == "groq":
    _banglish_llm = ChatGroq(
        api_key=settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
        model=settings.groq_chat_model,
        temperature=0.0,
        max_retries=0,
    )
else:
    _banglish_llm = ChatOllama(
        model=settings.banglish_converter_model,
        temperature=0.0,
        reasoning=False,
    )

def is_bangla_text(text: str) -> bool:
    has_alpha = False

    for char in text:
        if not char.isalpha():
            continue

        has_alpha = True

        if not ("\u0980" <= char <= "\u09FF"):
            return False

    return has_alpha

_SYSTEM_PROMPT = """
You are an expert Bengali spelling corrector for agricultural texts.
Fix the spelling mistakes in the provided Bengali text. Use the Original Banglish as a phonetic reference.

Rules:
- Preserve the original word order, sentence structure, and core meaning exactly.
- Fix misspelled words into standard agricultural Bengali spellings.
- Maintain technical terms and English abbreviations (e.g., pH, EC, NPK, TSP, GPS) as they are.
- Do NOT rewrite, answer, or add any extra text. 
- Return ONLY the final corrected Bengali sentence.
"""

def normalize_language(state: PipelineState) -> PipelineState:
    raw_query = state["raw_query"].strip()

    # if is_bangla_text(raw_query):
    #     logger.info("language_normalization language=bn")
    #     return {
    #         **state,
    #         "language": "bn",
    #         "normalized_query": raw_query
    #     }

    # avro_output = avro.parse(raw_query)

    # user_prompt = (
    #     f"Original Banglish:\n{raw_query}\n\n"
    #     f"Bengali text to spell-check:\n{avro_output}"
    # )

    # response = _banglish_llm.invoke([
    #         SystemMessage(content=_SYSTEM_PROMPT),
    #         HumanMessage(content=user_prompt),
    #     ])

    # normalized_query = str(response.content).strip()

    # logger.info("language_normalization language=banglish original=%r normalized=%r", raw_query, normalized_query)  # noqa: E501
    return {
        **state,
        "language": "bn",
        "normalized_query": raw_query,
    }