import time 
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.config import get_stream_writer

llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

# --- Node 1: fake intent/context check ---
def check_query(state: MessagesState):
    writer = get_stream_writer()
    writer({"status": "checking previous query..."})
    time.sleep(0.8)  # simulate work
    return {}

# --- Node 2: fake retrieval ---
def retrieve(state: MessagesState):
    writer = get_stream_writer()
    writer({"status": "retrieving relevant context..."})
    time.sleep(0.8)  # simulate DB/vector search
    return {}

# --- Node 3: actual generation (this one streams tokens) ---
def call_model(state: MessagesState):
    writer = get_stream_writer()
    writer({"status": "generating response..."})
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("check_query", check_query)
builder.add_node("retrieve", retrieve)
builder.add_node("chat", call_model)
builder.add_edge(START, "check_query")
builder.add_edge("check_query", "retrieve")
builder.add_edge("retrieve", "chat")
builder.add_edge("chat", END)

graph = builder.compile()

def get_response(user_input: str):
    input = {"messages": [("human", user_input)]}
    full_response = ""
    for stream_mode, chunk in graph.stream(input, stream_mode=["custom", "messages"]):
        if stream_mode == "custom":
            print(f"\n{chunk['status']}")
        elif stream_mode == "messages":
            msg_chunk, metadata = chunk
            if msg_chunk.content:
                if full_response == "":
                    print("AI: ", end="", flush=True)
                print(msg_chunk.content, end="", flush=True)
                full_response += msg_chunk.content
                time.sleep(0.10)
    print()
    return full_response

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = get_response(user_input)