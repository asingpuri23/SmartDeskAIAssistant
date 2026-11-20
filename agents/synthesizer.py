"""
agents/synthesizer.py — Merges responses from one or both agents
into a single coherent reply for the customer.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from config import llm
from state import SmartdeskAIState
from agents.prompts import SYNTHESIZER_PROMPT
from logger import setup_logger

logger = setup_logger("smartdeskai.synthesizer")


def synthesizer_node(state: SmartdeskAIState) -> dict:
    """Combine agent outputs into a final, friendly answer."""
    logger.info("Combining agent responses...")

    rag_resp = state.get("rag_response", "")
    jira_resp = state.get("jira_response", "")

    combined = ""
    if rag_resp:
        combined += f"[RAG Agent]\n{rag_resp}\n\n"
    if jira_resp:
        combined += f"[JIRA Agent]\n{jira_resp}\n"

    if not combined.strip():
        combined = "I could not find relevant information. Please try rephrasing your query."

    final = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(
                content=f"Customer query: {state['user_query']}\n\nAgent outputs:\n{combined}"
            ),
        ]
    )
    logger.info("Final answer ready (%d chars)", len(final.content))

    return {
        "final_answer": final.content,
        "messages": [final],
    }
