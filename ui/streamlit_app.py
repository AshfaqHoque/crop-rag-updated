"""Streamlit interface for the crop advisory chatbot."""

import json
import os
import uuid

import requests
import streamlit as st

from app.core.config import get_settings

settings = get_settings()
API_URL = os.getenv("API_URL", settings.ui_api_url)
REQUEST_TIMEOUT = settings.ui_request_timeout_seconds

AVATARS = {
    "user": "🧑",
    "assistant": "🌱",
}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AUNKUR AI",
    page_icon="🌱",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "conversations": {},
        "active_session_id": str(uuid.uuid4()),
        "language_type": "english",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def stream_backend(message: str,language_type: str,session_id: str,):
    """Stream assistant response from the backend."""

    payload = {
        "session_id": session_id,
        "message": message,
        "language_type": language_type,
    }
    with requests.post(API_URL,json=payload,stream=True,timeout=REQUEST_TIMEOUT,) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            content = chunk.get("content")
            if content:
                yield content

def get_backend_error(error: Exception) -> str:
    """Convert backend exceptions into user-friendly messages."""

    if isinstance(error, requests.exceptions.ConnectionError):
        return "⚠️ Can't reach the backend. Is the API server running?"
    if isinstance(error, requests.exceptions.Timeout):
        return "⚠️ The backend took too long to respond. Please try again."
    if isinstance(error, requests.exceptions.HTTPError):
        status = error.response.status_code
        if status == 422:
            return "⚠️ The request was rejected. Please check your message."
        return f"⚠️ Backend error ({status}). Check the server logs."
    if isinstance(error, (json.JSONDecodeError, KeyError)):
        return "⚠️ Received an unexpected response from the backend."
    return "⚠️ Something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def create_new_chat():
    st.session_state.active_session_id = str(uuid.uuid4())

def get_active_conversation():
    return st.session_state.conversations.get(
        st.session_state.active_session_id
    )

def save_message_pair(user_message: str, assistant_message: str):
    session_id = st.session_state.active_session_id
    conversation = get_active_conversation()
    messages = [
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": assistant_message,
        },
    ]
    if conversation:
        conversation["messages"].extend(messages)
        return
    title = user_message[:40].strip()
    if len(user_message) > 40:
        title += "..."
    st.session_state.conversations[session_id] = {
        "title": title,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🌱 AUNKUR AI")

    if st.button("＋ New chat",use_container_width=True,):
        create_new_chat()
        st.rerun()

    st.divider()
    st.caption("CHATS")

    if not st.session_state.conversations:
        st.caption("No conversations yet.")
    else:
        for session_id, conversation in reversed(list(st.session_state.conversations.items())):
            is_active = (session_id == st.session_state.active_session_id)
            col_chat, col_delete = st.columns([5, 1])
            with col_chat:
                if st.button(
                    conversation["title"],
                    key=f"chat_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_session_id = session_id
                    st.rerun()

            with col_delete:
                if st.button("🗑️",key=f"delete_{session_id}",):
                    del st.session_state.conversations[session_id]
                    if is_active:
                        create_new_chat()
                    st.rerun()

    st.divider()
    st.caption("LANGUAGE")
    st.session_state.language_type = st.radio(
        "Response language",
        ["english", "bangla", "arabic"],
        horizontal=True,
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

conversation = get_active_conversation()
if conversation:
    for message in conversation["messages"]:
        with st.chat_message(message["role"],avatar=AVATARS[message["role"]],):
            st.markdown(message["content"])
else:
    st.title("Crop Advisory AI")
    st.write("Ask about crops, fertilizers, pests, diseases, or farming practices.")


# ---------------------------------------------------------------------------
# New message
# ---------------------------------------------------------------------------

user_input = st.chat_input("Message Crop Advisory AI...")

if user_input:
    # Show user message immediately.
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)
    # Stream assistant response.
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            loading = st.empty()
            loading.markdown(
                """
                <style>
                    .aunkur-loader {
                        display: inline-flex;
                        align-items: center;
                        gap: 5px;
                        color: #5f6f52;
                        font-size: 0.9rem;
                    }
                    .aunkur-loader span {
                        width: 7px;
                        height: 7px;
                        border-radius: 50%;
                        background: #7a9d54;
                        animation: aunkur-bounce 1.2s infinite ease-in-out;
                    }
                    .aunkur-loader span:nth-child(2) {
                        animation-delay: 0.15s;
                    }
                    .aunkur-loader span:nth-child(3) {
                        animation-delay: 0.3s;
                    }
                    @keyframes aunkur-bounce {
                        0%, 60%, 100% { transform: translateY(0); opacity: 0.45; }
                        30% { transform: translateY(-4px); opacity: 1; }
                    }
                </style>
                <div class="aunkur-loader">
                    <span></span><span></span><span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            def stream_with_loading():
                for chunk in stream_backend(
                    message=user_input,
                    language_type=st.session_state.language_type,
                    session_id=st.session_state.active_session_id,
                ):
                    loading.empty()
                    yield chunk

            answer = st.write_stream(stream_with_loading())
        except Exception as error:
            st.error(get_backend_error(error))
            answer = None
    # Persist completed exchange.
    if answer:
        save_message_pair(user_input, answer)
        st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    "Crop Advisory AI can make mistakes. "
    "Verify important farming decisions."
)