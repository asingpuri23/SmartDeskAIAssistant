"""
agents/orchestrator.py — Classifies the user query and dispatches
to the correct specialist agent(s) using LangGraph's Send() API.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, Send

from config import llm
from state import SmartdeskAIState
from agents.prompts import ORCHESTRATOR_PROMPT
from logger import setup_logger

logger = setup_logger("smartdeskai.orchestrator")

class OrchestratorDecision(BaseModel):
    """The orchestrator's routing decision."""

    reasoning: str = Field(description="Brief explanation of why these agents were chosen")
    agents: list[Literal["rag_agent", "jira_agent"]] = Field(
        description="List of agents to dispatch. Always at least one.",
        min_length=1,
    )


routing_llm = llm.with_structured_output(OrchestratorDecision)


def orchestrator_node(
    state: SmartdeskAIState,
) -> Command[Literal["rag_agent_node", "jira_agent_node"]]:
    """Classify the query and fan out to one or both agents."""

    query = state["user_query"]
    logger.info("Routing query: %s", query[:60])

    # Conversation history is persisted via MemorySaver — just use it
    history = state.get("messages", [])[-6:]

    decision: OrchestratorDecision = routing_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        *history,
        HumanMessage(content=query),
    ])
    logger.info("Agents: %s | Reason: %s", decision.agents, decision.reasoning)

    # Reset per-turn buffers BEFORE sending to agents
    clean_state = {
        **state,
        "rag_messages": [],
        "jira_messages": [],
        "rag_response": "",
        "jira_response": "",
    }

    sends = [Send(f"{agent}_node", clean_state) for agent in decision.agents]
    return Command(goto=sends, update={"route": decision.agents})
