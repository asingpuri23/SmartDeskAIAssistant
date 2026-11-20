"""
agents/rag_agent.py — RAG Agent.

Uses LangChain's retrieval-augmented generation capabilities.
- rag_agent_node  → calls the RAG pipeline, stores response in rag_messages
- rag_tools_node  → executes tool calls, stores results in rag_messages
- should_continue_rag → conditional edge: tool calls? → tools : → synthesizer
"""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from config import llm
from state import SmartdeskAIState
from agents.prompts import RAG_AGENT_PROMPT
from tools.rag_tools import rag_tools
from logger import setup_logger

logger = setup_logger("smartdeskai.rag_agent")

# LLM with menu tools bound
it_support_llm = llm.bind_tools(rag_tools)
tool_map = {t.name: t for t in rag_tools}


def rag_agent_node(state: SmartdeskAIState) -> dict:
    """Call the LLM. On first invocation build the initial prompt
    using persisted conversation history; on subsequent invocations
    (after tool results) continue from rag_messages."""

    existing = state.get("rag_messages") or []

    if not existing:
        # First call — conversation history is persisted via MemorySaver
        logger.info("Processing IT support query...")
        history = state.get("messages", [])[-6:]

        all_msgs = [
            SystemMessage(content=RAG_AGENT_PROMPT),
            *history,
            HumanMessage(content=state["user_query"]),
        ]
    else:
        # Subsequent call — tool results already in rag_messages
        logger.info("RAG agent re-invoked after tool results")
        all_msgs = existing

    response = it_support_llm.invoke(all_msgs)
    all_msgs = [*all_msgs, response]

    has_tools = bool(getattr(response, "tool_calls", None))
    logger.info("Tool calls: %s", has_tools)

    update: dict = {"rag_messages": all_msgs}

    if not has_tools:
        update["rag_response"] = response.content
        update["messages"] = [response]

    return update


def rag_tools_node(state: SmartdeskAIState) -> dict:
    """Execute tool calls from the last AI message in rag_messages."""

    all_msgs = list(state["rag_messages"])
    last_msg = all_msgs[-1]
    tool_names = [tc["name"] for tc in last_msg.tool_calls]
    logger.info("Executing tools: %s", tool_names)

    for tc in last_msg.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        all_msgs.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"rag_messages": all_msgs}


def should_continue_rag(state: SmartdeskAIState) -> str:
    """Conditional edge: route to tools or synthesizer."""
    rag_msgs = state.get("rag_messages") or []
    if rag_msgs:
        last = rag_msgs[-1]
        logger.info("RAG agent last message: %s", last.content)
        if getattr(last, "tool_calls", None):
            return "rag_tools_node"
        elif('not found' in last.content.lower() or 'no results' in last.content.lower() or 'couldn\'t find' in last.content.lower()):
            return "jira_agent_node"
    return "synthesizer_node"