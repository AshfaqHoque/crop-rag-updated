from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You rewrite follow-up questions.

Rules:

- Extract the main agricultural subject from the previous question.
- If the current question already explicitly mentions its own crop, disease, fertilizer, pesticide, pest, or other agricultural subject, return the current question exactly unchanged. Do not combine it with the previous question.
- Otherwise, if the current question is clearly a follow-up question or uses a referring expression (it, this, that), replace the reference with the previous subject.
- Preserve the current question's wording and intent as much as possible.
- Never return the previous question itself.
- Never answer the question.
- Never add information.
- Output only the rewritten query.

Examples:

Previous: "বোরো ধানের বৈশিষ্ট্যগুলো কী কী?"
Current: "এতে কীভাবে সেচ দেব?"
Output: "বোরো ধানে কীভাবে সেচ দেব?"

Previous: "ফল আর্মিওয়ার্মের লক্ষণ কী?"
Current: "এটি কীভাবে দমন করব?"
Output: "ফল আর্মিওয়ার্ম কীভাবে দমন করব?"

Previous: "আমন ধানে কতবার সেচ দিতে হয়?"
Current: "ব্লাস্ট রোগ কীভাবে দমন করব?"
Output: "ব্লাস্ট রোগ কীভাবে দমন করব?"
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


current = "গমে প্রথম সেচ কখন দিতে হবে?"
previous = "ব্রি ধান ১১ সেচ কিভাবে দিবো?"

result = chain.invoke(
    {
        "previous": previous,
        "current": current,
    }
)

print("previous ques: ", previous)
print("current ques: ", current)
print("rewritten ques: ", result.content)