from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.services.pipeline.nodes.generate_capability import generate_capability
from app.services.pipeline.registry import CropInfo

MODULE = "app.services.pipeline.nodes.generate_capability"


@patch(f"{MODULE}.invoke_text", return_value="I can help with these topics.")
@patch(f"{MODULE}.get_known_crops")
def test_generate_capability_includes_coverage_and_crop_registry(mock_crops, mock_invoke):
    mock_crops.return_value = [CropInfo(crop_name="Boro Paddy", crop_bangla_name="বোরো ধান")]
    state = {"raw_query": "What crops and topics can you answer about?", "language_type": "english"}

    result = generate_capability(state)

    prompt = mock_invoke.call_args.args[0][0].content
    assert "Aunkur AI" in prompt
    assert "Boro Paddy (বোরো ধান)" in prompt
    assert "seed" in prompt
    assert "intercultural operations" in prompt
    assert "general information about a crop" in prompt
    assert "varieties" in prompt
    assert "pests" in prompt
    assert "diseases" in prompt
    assert "fertilizers" in prompt
    assert "Aunkur company" in prompt
    assert "Porokh soil-testing device" in prompt
    assert result["answer"] == "I can help with these topics."
    assert isinstance(result["messages"][0], AIMessage)