"""Return a fixed response for requests outside the assistant's scope."""

from langchain_core.messages import AIMessage

from app.services.pipeline.state import PipelineState

_FALLBACK_MESSAGES = {
    "bn": "দুঃখিত, এই বিষয়ে আমি আপনাকে সাহায্য করতে পারব না। আপনি কি কৃষি সম্পর্কে কিছু জানতে চান?",
    "en": "Sorry, I can't help with this topic. Would you like to know something about agriculture?",
    "ar": "عذرًا، لا أستطيع مساعدتك في هذا الموضوع. هل ترغب في معرفة شيء عن الزراعة؟",
}


def generate_out_of_scope(state: PipelineState) -> PipelineState:
    language_type = state.get("language_type", "en")
    fallback_message = _FALLBACK_MESSAGES.get(language_type, _FALLBACK_MESSAGES["en"])
    return {
        **state,
        "messages": [AIMessage(content=fallback_message)],
        "answer": fallback_message,
    }