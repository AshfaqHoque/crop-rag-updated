from app.services.pipeline.registry import get_known_crops

NODE_THINKING = {
    "rewrite_query": {"en": "Understanding your question...", "bn": "আপনার প্রশ্ন বোঝা হচ্ছে..."},
    "route": {"en": "Determining how best to help...", "bn": "কীভাবে সাহায্য করা যায় তা নির্ধারণ করা হচ্ছে..."},
    "decompose_query": {"en": "Breaking down your question...", "bn": "আপনার প্রশ্ন বিশ্লেষণ করা হচ্ছে..."},
    "retrieve_rerank_and_compress": {"en": "Retrieving crop data...", "bn": "তথ্য সংগ্রহ করা হচ্ছে..."},
    "retrieve_company": {"en": "Looking up company information...", "bn": "কোম্পানির তথ্য খোঁজা হচ্ছে..."},
    "retrieve_soil_test": {"en": "Retrieving soil test information...", "bn": "মাটি পরীক্ষার তথ্য সংগ্রহ করা হচ্ছে..."},
    "generate": {"en": "Preparing your answer...", "bn": "আপনার উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_company": {"en": "Preparing your answer...", "bn": "আপনার উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_soil_test": {"en": "Preparing your answer...", "bn": "আপনার উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_chitchat": {"en": "Preparing a response...", "bn": "উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_capability": {"en": "Preparing a response...", "bn": "উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_meaningless": {"en": "Preparing a response...", "bn": "উত্তর প্রস্তুত করা হচ্ছে..."},
    "generate_out_of_scope": {"en": "Preparing a response...", "bn": "উত্তর প্রস্তুত করা হচ্ছে..."},
    "handle_agronomist_request": {"en": "Connecting you with an agronomist...", "bn": "কৃষি বিশেষজ্ঞের সাথে সংযোগ করা হচ্ছে..."},
}

def get_node_thinking_message(node: str, lang: str = "en", crops: list[str] | None = None) -> str | None:
    m = NODE_THINKING.get(node)
    if not m:
        return None

    if node == "retrieve_rerank_and_compress" and crops:
        if lang == "bn":
            bn_names = {
                crop.crop_name: crop.crop_bangla_name
                for crop in get_known_crops()
            }
            crops = [bn_names.get(crop) or crop for crop in crops]

        names = (", ").join(crops)

        return (
            f"{names} এর তথ্য সংগ্রহ করা হচ্ছে..."
            if lang == "bn"
            else f"Retrieving information about {names}..."
        )

    return m.get("bn" if lang == "bn" else "en")