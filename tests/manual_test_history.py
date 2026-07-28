# import json
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate

# # System prompt rewritten for a 7B model using positive constraints and clear logic steps
# SYSTEM_PROMPT = """You are an agricultural query rewriter. Your only job is to resolve pronouns/references in the Current Question using the Previous Question.

# Follow these strict logical steps:
# 1. Identify the main agricultural subject (crop, disease, pest, fertilizer) in the Previous Question.
# 2. Check the Current Question. Does it already mention a specific crop, disease, pest, or fertilizer?
#    - If YES, output the Current Question exactly as it is.
#    - If NO (it uses "এতে", "এটি", "সেখানে" or implicitly references the past topic), replace that reference with the subject from Step 1.
# 3. Keep the exact tone, intent, and words of the Current Question.

# You must respond ONLY with a JSON object matching this schema:
# {{
#     "reasoning": "Briefly state if the current question already has a subject or needs rewriting.",
#     "rewritten_query": "The final Bengali query string here"
# }}
# """

# CONTEXT = """
# <conversation>
# <previous>{previous}</previous>
# <current>{current}</current>
# </conversation>

# Respond only in JSON format.
# """

# # Enable JSON mode in Ollama for Qwen
# llm = ChatOllama(
#     model="gemma3:4b", 
#     temperature=0,
#     format="json"  # Forces JSON validation at the inference level
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", CONTEXT)
# ])

# chain = prompt | llm 

# # Test case where current question has its own crop (Wheat) and should NOT be changed
# current = "গমে প্রথম সেচ কখন দিতে হবে?"
# previous = "ব্রি ধান ১১ সেচ কিভাবে দিবো?"

# result = chain.invoke(
#     {
#         "previous": previous,
#         "current": current,
#     }
# )

# # Parse the structured JSON output safely
# try:
#     output_data = json.loads(result.content)
#     rewritten = output_data.get("rewritten_query", current)
#     reasoning = output_data.get("reasoning", "")
# except Exception:
#     rewritten = result.content  # Fallback

# print("Previous Ques: ", previous)
# print("Current Ques:  ", current)
# print("Reasoning:     ", reasoning)
# print("Rewritten:     ", rewritten)


import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# System prompt rewritten in clear English but optimized to preserve both languages
SYSTEM_PROMPT = """You are an agricultural query rewriter supporting both English and Bengali (Bangla). 
Your single job is to resolve missing references or pronouns in the Current Question using context from the Previous Question.

Rules:
1. Detect the main agricultural subject (crop, disease, pest, fertilizer, tool) from the Previous Question.
2. Check the Current Question:
   - If it ALREADY explicitly mentions its own crop, disease, or agricultural subject, do NOT rewrite it. Return it exactly as it is.
   - If it uses pronouns, implied references, or short follow-ups (e.g., "it", "this", "এটিতে", "কখন দিবো?"), replace or complete the reference using the subject from the Previous Question.
3. CRITICAL: Maintain the exact language (English or Bangla) of the Current Question. Never translate Bangla to English or English to Bangla.
4. Keep the original intent and tone. Do not add extra information or answer the question.

You must respond ONLY with a JSON object matching this schema:
{{
    "rewritten_query": "The rewritten question in its original language"
}}"""

CONTEXT = """
<conversation>
<previous>{previous}</previous>
<current>{current}</current>
</conversation>

Respond ONLY as a valid JSON object.
"""

# Initializing Gemma 3 4B via Ollama
llm = ChatOllama(
    model="gemma3:4b-it-qat",  # Quantization-Aware Trained Gemma 3 4B
    temperature=0,
    format="json"              # Forces strict structural JSON output
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", CONTEXT)
])

chain = prompt | llm 

def rewrite_query(prev, curr):
    result = chain.invoke({"previous": prev, "current": curr})
    try:
        output_data = json.loads(result.content)
        return output_data.get("rewritten_query", curr)
    except Exception:
        return result.content

# ---- MULTILINGUAL TEST SUITE ----

# print("--- Test 1: Bangla Follow-up (Should rewrite in Bangla) ---")
# # Subject: বারি আম ৪. Current question uses "এটিতে"
# print("Result:", rewrite_query("বারি আম ৪ এর ফলন কেমন?", "এটিতে কখন সার দিতে হবে?"))

# print("\n--- Test 2: English Follow-up (Should rewrite in English) ---")
# # Subject: Boro Rice. Current question uses implicit "it"
# print("Result:", rewrite_query("What are the features of Boro Rice?", "How often should I irrigate?"))

print("\n--- Test 3: ")
print("Result:", rewrite_query("ব্রি ধান ১১ সেচ কিভাবে দিবো?", "গমে প্রথম সেচ কখন দিতে হবে?"))


