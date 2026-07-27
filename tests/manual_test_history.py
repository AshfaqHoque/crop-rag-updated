from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You rewrite follow-up questions.

Rules:

- Extract the main agricultural subject from the previous question.
- If the current question already explicitly mentions a crop, disease, fertilizer, pesticide, or other agricultural subject, return it unchanged.
- Otherwise, if the current question uses a referring expression (it, its, this, that, these, those, they, them), replace that reference with the previous subject.
- Preserve wording as much as possible.
- Never answer the question.
- Never add information.
- Output only the rewritten query.
"""
CONTEXT = """
Previous Question:
{previous}

Current Question:
{current}
"""
llm = ChatOllama(model="gemma3:4b", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", CONTEXT)
])

chain = prompt | llm 


previous = "ভুট্টার চাষ কীভাবে করব?"
current = "এতে কতবার সেচ দিতে হবে?"

result = chain.invoke(
    {
        "previous": previous,
        "current": current,
    }
)

print(result.content)