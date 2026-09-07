import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/chat"

st.set_page_config(
    page_title="Crop RAG Assistant",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Crop RAG Assistant")

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def ask_backend(message: str, language_type: str = "english") -> dict:
    """Send the message to the FastAPI /chat endpoint and return the parsed response."""
    payload = {
        "session_id": st.session_state.session_id,
        "message": message,
        "language_type": language_type,
    }
    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# --- Render existing conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask about crops...")

if user_input:
    # Show + store the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the backend and show the assistant's reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_backend(user_input)
            answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})