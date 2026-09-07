import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/chat"

st.set_page_config(
    page_title="Crop RAG Assistant",
    page_icon="🌾",
    layout="wide",
)

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "language_type" not in st.session_state:
    st.session_state.language_type = "english"


def ask_backend(message: str, language_type: str) -> tuple[str | None, str | None]:
    """Returns (answer, error_message). Exactly one of them will be non-None."""
    payload = {
        "session_id": st.session_state.session_id,
        "message": message,
        "language_type": language_type,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["answer"], None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Can't reach the backend. Is `uvicorn app.main:app` running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "⚠️ The backend took too long to respond. Try again, or check the server logs."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 422:
            return None, "⚠️ The request was rejected — check the message isn't empty or too long."
        return None, f"⚠️ Backend error ({status}). Check the server logs for details."
    except (KeyError, ValueError):
        return None, "⚠️ Got an unexpected response format from the backend."


# --- Sidebar ---
with st.sidebar:
    st.header("🌾 Crop RAG Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.session_state.language_type = st.radio(
        "Language",
        options=["english", "bangla"],
        index=["english", "bangla"].index(st.session_state.language_type),
        horizontal=True,
    )

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# --- Main chat area ---
st.title("🌾 Crop RAG Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about crops...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, error = ask_backend(user_input, st.session_state.language_type)

        if error:
            st.error(error)
            answer = None  # don't save a failed turn as if it were a real answer
        else:
            st.markdown(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})