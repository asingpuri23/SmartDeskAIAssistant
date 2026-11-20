"""
agents/jira_agent.py — JIRA Agent.

Uses LangChain's retrieval-augmented generation capabilities.
- jira_agent_node  → calls the JIRA API, stores response in jira_messages
- jira_tools_node  → executes tool calls, stores results in jira_messages
- should_continue_jira → conditional edge: tool calls? → tools : → synthesizer
"""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.types import interrupt
from config import llm
from state import SmartdeskAIState
from agents.prompts import JIRA_AGENT_PROMPT
from tools.jira_tools import jira_tools
from logger import setup_logger

logger = setup_logger("smartdeskai.jira_agent")

jira_llm = llm.bind_tools(jira_tools)
tool_map = {t.name: t for t in jira_tools}

def jira_agent_node(state: SmartdeskAIState) -> dict:
    """Call the LLM. On first invocation build the initial prompt
    using persisted conversation history; on subsequent invocations
    (after tool results) continue from jira_messages."""

    existing = state.get("jira_messages") or []

    checkForRagResponse = True

    interrupt_response = state.get("user_query", None)
    logger.info(f"Interrupt response: {interrupt_response}")

    if interrupt_response:
        if interrupt_response.lower() == "yes":
            logger.info("User confirmed to create a JIRA ticket. Continuing...")
            checkForRagResponse = False
        elif interrupt_response.lower() == "no":
            logger.info("User declined to create a JIRA ticket. Exiting JIRA agent.")
            return {"jira_messages": existing, "jira_response": "User declined to create a JIRA ticket."}

    if checkForRagResponse:
        rag_response = state.get("rag_messages", [])
        last_rag_response = rag_response[-1] if rag_response else None
        if last_rag_response:
            if ('not found' in last_rag_response.content.lower()) or ('no relevant information' in last_rag_response.content.lower() or 'couldn\'t find' in last_rag_response.content.lower()):
                logger.info("RAG agent did not find relevant information. Invoking JIRA agent with user confirmation...")
                user_confirmation_response = interrupt(
                    "Should I create a JIRA ticket for this issue? Please respond with 'yes' or 'no'."
                )
                if user_confirmation_response.lower() != 'yes':
                    logger.info("User declined to create a JIRA ticket. Exiting JIRA agent.")
                    return {"jira_messages": existing, "jira_response": "User declined to create a JIRA ticket."}

    if not existing:
        # First call — conversation history is persisted via MemorySaver
        logger.info("Processing IT support query...")
        history = state.get("messages", [])[-6:]

        all_msgs = [
            SystemMessage(content=JIRA_AGENT_PROMPT),
            *history,
            HumanMessage(content=state["user_query"]),
        ]
    else:
        # Subsequent call — tool results already in jira_messages
        logger.info("JIRA agent re-invoked after tool results")
        all_msgs = existing

    response = jira_llm.invoke(all_msgs)
    all_msgs = [*all_msgs, response]

    has_tools = bool(getattr(response, "tool_calls", None))
    logger.info("Tool calls: %s", has_tools)

    update: dict = {"jira_messages": all_msgs}

    if not has_tools:
        update["jira_response"] = response.content
        update["messages"] = [response]

    return update


def jira_tools_node(state: SmartdeskAIState) -> dict:
    """Execute tool calls from the last AI message in jira_messages."""

    all_msgs = list(state["jira_messages"])
    last_msg = all_msgs[-1]
    tool_names = [tc["name"] for tc in last_msg.tool_calls]
    logger.info("Executing tools: %s", tool_names)

    for tc in last_msg.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        all_msgs.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"jira_messages": all_msgs}


def should_continue_jira(state: SmartdeskAIState) -> str:
    """Conditional edge: route to tools or synthesizer."""
    jira_msgs = state.get("jira_messages") or []
    if jira_msgs:
        last = jira_msgs[-1]
        if getattr(last, "tool_calls", None):
            return "jira_tools_node"
    return "synthesizer_node"