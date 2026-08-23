import logging
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.services.llm.client import (
    get_chat_llm,
    get_groq_chat_llm,
    get_ollama_chat_llm,
    get_structured_llm,
    invoke_text,
)

MODULE = "app.services.llm.client"


@patch(f"{MODULE}.ChatGroq")
@patch(f"{MODULE}.get_settings")
def test_chat_client_uses_configured_groq_model(mock_settings, mock_chat_groq):
    mock_settings.return_value = SimpleNamespace(
        groq_api_key=SecretStr("gsk_test"),
        groq_chat_model="openai/gpt-oss-120b",
        llm_temperature=0.0,
    )
    get_groq_chat_llm.cache_clear()

    client = get_groq_chat_llm()

    assert client is mock_chat_groq.return_value
    mock_chat_groq.assert_called_once_with(
        api_key="gsk_test",
        model="openai/gpt-oss-120b",
        temperature=0.0,
        max_retries=0,
    )
    get_groq_chat_llm.cache_clear()


@patch(f"{MODULE}.ChatOllama")
@patch(f"{MODULE}.get_settings")
def test_ollama_chat_client_remains_available(mock_settings, mock_chat_ollama):
    mock_settings.return_value = SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="gemma3:4b",
        llm_temperature=0.1,
    )
    get_ollama_chat_llm.cache_clear()

    client = get_ollama_chat_llm()

    assert client is mock_chat_ollama.return_value
    mock_chat_ollama.assert_called_once_with(
        base_url="http://localhost:11434",
        model="gemma3:4b",
        reasoning=False,
        temperature=0.1,
    )
    get_ollama_chat_llm.cache_clear()


@patch(f"{MODULE}.get_ollama_chat_llm")
@patch(f"{MODULE}.get_groq_chat_llm")
@patch(f"{MODULE}.get_settings")
def test_chat_provider_selects_groq(mock_settings, mock_groq, mock_ollama):
    mock_settings.return_value = SimpleNamespace(chat_provider="groq")
    get_chat_llm.cache_clear()

    assert get_chat_llm() is mock_groq.return_value
    mock_groq.assert_called_once_with(None)
    mock_ollama.assert_not_called()
    get_chat_llm.cache_clear()


@patch(f"{MODULE}.get_chat_llm")
@patch(f"{MODULE}.get_settings")
def test_groq_structured_output_uses_json_schema(mock_settings, mock_get_llm):
    mock_settings.return_value = SimpleNamespace(chat_provider="groq")
    schema = type("Schema", (), {})

    result = get_structured_llm(schema)

    mock_get_llm.return_value.with_structured_output.assert_called_once_with(
        schema,
        method="json_schema",
    )
    assert result is mock_get_llm.return_value.with_structured_output.return_value


@patch(f"{MODULE}.get_chat_llm")
@patch(f"{MODULE}.get_settings")
def test_ollama_structured_output_keeps_default_method(mock_settings, mock_get_llm):
    mock_settings.return_value = SimpleNamespace(chat_provider="ollama")
    schema = type("Schema", (), {})

    get_structured_llm(schema)

    mock_get_llm.return_value.with_structured_output.assert_called_once_with(schema)


@patch(f"{MODULE}.get_chat_llm")
@patch(f"{MODULE}.get_settings")
def test_text_invocation_logs_provider_and_model(mock_settings, mock_get_llm, caplog):
    mock_settings.return_value = SimpleNamespace(
        chat_provider="groq",
        chat_model="openai/gpt-oss-120b",
    )
    mock_get_llm.return_value.invoke.return_value = SimpleNamespace(content="answer")

    with caplog.at_level(logging.INFO, logger=MODULE):
        assert invoke_text("question") == "answer"

    assert "provider=groq model=openai/gpt-oss-120b operation=text" in caplog.text
