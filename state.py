"""
state.py — Shared state schema for the SnackStack multi-agent graph.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SmartdeskAIState(TypedDict):
    """State that flows through every node in the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    route: list[str]            # e.g. ["rag_agent", "jira_agent"]
    interrupt_response: str      # e.g. "yes" or "no" from user when asked to create a JIRA ticket

    # Agent-local message buffers (isolated via add_messages reducer)
    rag_messages: Annotated[list[AnyMessage], add_messages]
    jira_messages: Annotated[list[AnyMessage], add_messages]

    # Agent outputs
    rag_response: str
    jira_response: str
    final_answer: str