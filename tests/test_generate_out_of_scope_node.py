import pytest
from langchain_core.messages import AIMessage

from app.services.pipeline.nodes.generate_out_of_scope import generate_out_of_scope


@pytest.mark.parametrize(
    ("language_type", "expected"),
    [
        ("bn", "দুঃখিত, এই বিষয়ে আমি আপনাকে সাহায্য করতে পারব না। আপনি কি কৃষি সম্পর্কে কিছু জানতে চান?"),
        ("en", "Sorry, I can't help with this topic. Would you like to know something about agriculture?"),
        ("ar", "عذرًا، لا أستطيع مساعدتك في هذا الموضوع. هل ترغب في معرفة شيء عن الزراعة؟"),
    ],
)
def test_generate_out_of_scope_uses_requested_language(language_type, expected):
    result = generate_out_of_scope({"language_type": language_type})

    assert result["answer"] == expected
    assert result["messages"] == [AIMessage(content=expected)]