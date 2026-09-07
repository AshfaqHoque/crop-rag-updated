import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/chat"

st.set_page_config(
    page_title="AUNKUR AI",
    page_icon="🌱",
    layout="wide",
)

# --- Custom styling ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            max-width: 800px;
            padding-top: 2.5rem;
            padding-bottom: 7rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #171717;
            border-right: 1px solid #2a2a2a;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e5e5;
        }

        section[data-testid="stSidebar"] button {
            background-color: transparent !important;
            border: 1px solid transparent !important;
            text-align: left !important;
            font-weight: 400 !important;
            font-size: 0.88rem !important;
            border-radius: 8px !important;
            transition: background-color 0.15s ease;
        }
        section[data-testid="stSidebar"] button:hover {
            background-color: #2a2a2a !important;
            border-color: #2a2a2a !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            background-color: #2f2f2f !important;
            border-color: #2f2f2f !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {
            border: 1px solid #3d3d3d !important;
            font-weight: 500 !important;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.25rem;
        }
        .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: #8a8a8a;
            margin: 1rem 0 0.4rem 0.25rem;
            text-transform: uppercase;
        }
        .sidebar-empty {
            font-size: 0.82rem;
            color: #6b6b6b;
            padding: 0.4rem 0.25rem;
            font-style: italic;
        }

        [data-testid="stChatMessage"] {
            padding: 0.6rem 0.75rem;
            margin-bottom: 0.25rem;
        }
        [data-testid="stChatMessageContent"] p {
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .empty-state {
            text-align: center;
            margin-top: 14vh;
        }
        .empty-state .icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }
        .empty-state h2 {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.35rem;
        }
        .empty-state p {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .footer-note {
            text-align: center;
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state setup ---
# conversations only holds chats that have completed at least one exchange
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # {session_id: {"title": str, "messages": [...]}}

if "language_type" not in st.session_state:
    st.session_state.language_type = "english"

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = str(uuid.uuid4())


def ask_backend(message: str, language_type: str, session_id: str) -> tuple[str | None, str | None]:
    payload = {
        "session_id": session_id,
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
    st.markdown('<div class="sidebar-brand">🌱 &nbsp;AUNKUR AI</div>', unsafe_allow_html=True)

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.active_session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Chats</div>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<div class="sidebar-empty">No conversations yet</div>', unsafe_allow_html=True)
    else:
        for session_id in reversed(list(st.session_state.conversations.keys())):
            convo = st.session_state.conversations[session_id]
            is_active = session_id == st.session_state.active_session_id

            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    convo["title"],
                    key=f"select_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_session_id = session_id
                    st.rerun()
            with col2:
                if st.button("×", key=f"delete_{session_id}"):
                    del st.session_state.conversations[session_id]
                    if session_id == st.session_state.active_session_id:
                        st.session_state.active_session_id = str(uuid.uuid4())
                    st.rerun()

    st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)
    st.session_state.language_type = st.radio(
        "Language",
        options=["english", "bangla"],
        index=["english", "bangla"].index(st.session_state.language_type),
        horizontal=True,
        label_visibility="collapsed",
    )

# --- Main chat area ---
AVATARS = {"user": "🧑", "assistant": "🌱"}
active_session_id = st.session_state.active_session_id
active_convo = st.session_state.conversations.get(active_session_id)
existing_messages = active_convo["messages"] if active_convo else []

if not existing_messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🌱</div>
            <h2>Crop Advisory AI</h2>
            <p>Ask about crops, fertilizers, pests, or farming practices — in English or Bangla.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in existing_messages:
        with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
            st.markdown(msg["content"])

user_input = st.chat_input("Message Crop Advisory AI...")

if user_input:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            answer, error = ask_backend(user_input, st.session_state.language_type, active_session_id)

        if error:
            st.error(error)
            answer = None
        else:
            st.markdown(answer)

    if answer is not None:
        updated_messages = existing_messages + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ]
        if active_convo:
            active_convo["messages"] = updated_messages
        else:
            title = user_input[:36] + ("..." if len(user_input) > 36 else "")
            st.session_state.conversations[active_session_id] = {
                "title": title,
                "messages": updated_messages,
            }
        st.rerun()

st.markdown(
    '<div class="footer-note">Crop Advisory AI can make mistakes. Verify important farming decisions.</div>',
    unsafe_allow_html=True,
)