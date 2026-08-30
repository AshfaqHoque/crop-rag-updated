from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END

llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", call_model)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()

def get_response(user_input: str):
    input = {"messages": [("human", user_input)]}
    result = graph.invoke(input)
    return result["messages"][-1].content

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = get_response(user_input)
        print(f"AI: {response}")