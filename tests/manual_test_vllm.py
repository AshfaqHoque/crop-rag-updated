from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:8091/v1",
    model="gemma4:12b",
    api_key="not-needed",
    # extra_body={
    #     "chat_template_kwargs": {"enable_thinking": True}
    # }
)

messages = [
    HumanMessage(content="What is 25 multiplied by 17?")
]

response = llm.invoke(messages)

print("TYPE:", type(response))
print("CONTENT:", response.content)
print("ADDITIONAL:", response.additional_kwargs)
print("METADATA:", response.response_metadata)