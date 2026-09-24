"""Handle user acceptance of the agronomist offer (HITL escalation)."""

from langchain_core.messages import AIMessage

from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

# Fixed confirmation shown when the user accepts
_CONFIRMATION = "আপনার মেসেজটি এগ্রোনমিস্টের কাছে পৌঁছানো হয়েছে।"


def handle_agronomist_request(state: PipelineState) -> PipelineState:
    """
    User accepted the previous offer to talk to an agronomist.
    Return a fixed confirmation and set status="human" so the API layer can act on it.
    """
    logger.info("handle_agronomist_request session=%s → status=human",state.get("session_id"),)
    
    return {
        **state,
        "answer": _CONFIRMATION,
        "status": "human",
        "messages": [AIMessage(content=_CONFIRMATION)],
    }