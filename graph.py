"""
graph.py — Build and compile the SmartDesk AI LangGraph StateGraph.

Graph topology (conditional edges shown as dashed):
    START → orchestrator ─ ─ ─ ┬─ rag_agent_node ─ ─ ─ rag_tools_node ────┐
                                │       ↑________________________ ________┘
                                |       └─ ─ ─ ─ → jira_agent_node → jira_tools_node ──┐
                                |       |               |______________________________|
                                |       |               ↓                       
                                │       └─ ─ ─ ─ → synthesizer_node → END   
                                │
                                └─ jira_agent_node ─ ─ ─ jira_tools_node ───┐
                                        ↑___________________________________┘
                                        └─ ─ ─ ─ → synthesizer_node → END

    Orchestrator dispatches agents conditionally via Send().
    Each agent uses a conditional edge to route to its tools node
    or directly to the synthesizer.

HITL:
    order_agent_node uses interrupt() to pause execution when the user
    query is missing an order ID / tracking ID / email.  The caller
    resumes with Command(resume=<user_input>).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import SmartdeskAIState
from agents import (
    orchestrator_node,
    rag_agent_node,
    rag_tools_node,
    should_continue_rag,
    jira_agent_node,
    jira_tools_node,
    should_continue_jira,
    synthesizer_node,
)
from logger import setup_logger

logger = setup_logger("smartdeskai.graph")


def build_graph():
    """Construct, compile, and return the SnackStack graph."""

    builder = StateGraph(SmartdeskAIState)

    # ── Nodes ───────────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("rag_agent_node", rag_agent_node)
    builder.add_node("rag_tools_node", rag_tools_node)
    builder.add_node("jira_agent_node", jira_agent_node)
    builder.add_node("jira_tools_node", jira_tools_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    # ── Edges ───────────────────────────────────────────────

    # START → orchestrator (fixed)
    builder.add_edge(START, "orchestrator")
    # orchestrator → agent(s) routing handled via Command + Send()

    # RAG agent: conditional → tools (loop) or synthesizer
    builder.add_conditional_edges(
        "rag_agent_node",
        should_continue_rag,
        {"rag_tools_node": "rag_tools_node", "jira_agent_node": "jira_agent_node", "synthesizer_node": "synthesizer_node"},
    )
    builder.add_edge("rag_tools_node", "rag_agent_node")

    # Jira agent: conditional → tools (loop) or synthesizer
    builder.add_conditional_edges(
        "jira_agent_node",
        should_continue_jira,
        {"jira_tools_node": "jira_tools_node", "synthesizer_node": "synthesizer_node"},
    )
    builder.add_edge("jira_tools_node", "jira_agent_node")

    # Synthesizer → END (fixed)
    builder.add_edge("synthesizer_node", END)

    # ── Compile with memory ─────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Smartdesk AI graph compiled")
    return graph