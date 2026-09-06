from unittest.mock import patch

from app.services.pipeline.nodes.compress_chunk import get_context_compressor

MODULE = "app.services.pipeline.nodes.compress_chunk"


def test_context_compressor_binds_max_tokens():
    get_context_compressor.cache_clear()

    with (
        patch(f"{MODULE}.get_chat_llm") as mock_get_chat_llm,
        patch(f"{MODULE}.LLMChainExtractor.from_llm") as mock_from_llm,
    ):
        llm = mock_get_chat_llm.return_value
        bound_llm = llm.bind.return_value

        get_context_compressor()

        llm.bind.assert_called_once_with(max_tokens=1000)
        mock_from_llm.assert_called_once_with(
            bound_llm,
            prompt=get_context_compressor.__globals__["CUSTOM_DEFAULT_PROMPT"],
        )

    get_context_compressor.cache_clear()