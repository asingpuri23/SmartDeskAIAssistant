"""
main.py — SmartDesk-AI Assistant entry point.

"""

import argparse
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
import sys
from openai import OpenAI


sys.path.append("C:/Users/aditi/workspace_uplevel/SmartDeskAI/SmartDeskAIAgent/smartdesk-ai/smartdesk-ai")  # Add the path to your module

from graph import build_graph
from logger import setup_logger

logger = setup_logger("smartdesk-ai.main")

openai_client = OpenAI()


SAFE_FALLBACK = (
    "I'm sorry, I can only answer questions about IT support or create a JIRA ticket. "
)


def moderation_check(query):
    """Use OpenAI Moderation API to flag harmful content (free)."""
    mod_response = openai_client.moderations.create(input=query)
    result = mod_response.results[0]
    if result.flagged:
        cats = [c for c, v in result.categories.model_dump().items() if v]
        return SAFE_FALLBACK, f"moderation:{cats}"
    return None, None

class SmartdeskAIAssistant:
    """Wires the compiled graph to CLI"""

    def __init__(self):
        self.graph = build_graph()
        self.thread_id = str(uuid.uuid4())

    @property
    def _config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def reset(self):
        """Start a fresh conversation thread."""
        self.thread_id = str(uuid.uuid4())
        logger.info("Conversation reset (thread: %s)", self.thread_id[:8])

    def ask(self, query: str) -> str:
        """Send a query through the graph, handling any HITL interrupts."""

        result_moderation, reason = moderation_check(query)

        if result_moderation is not None:
            return result_moderation
    
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)], "user_query": query},
            self._config,
        )

        # Check for HITL interrupts (e.g. order agent asking for an ID)
        snapshot = self.graph.get_state(self._config)
        for task in snapshot.tasks:
            if not getattr(task, "interrupts", None):
                continue
            for intr in task.interrupts:
                prompt = intr.value

                # Collect user response via mic or keyboard
                user_input = input(f"\n{prompt}\nYou > ").strip() or "unknown"

                logger.info("Resuming with: %s", user_input)

            # Resume the graph with the user's answer
            result = self.graph.invoke(Command(update={"interrupt_response": user_input, "messages": [SystemMessage(content=prompt), HumanMessage(content=user_input)]}),  self._config)

        answer = result.get("final_answer", "Sorry, I could not process your request.")

        return answer


# ── Interactive CLI loops ───────────────────────────────────

def run_text_loop(assistant: SmartdeskAIAssistant):
    """REPL: type queries, get agent responses."""
    logger.info("SmartDesk-AI — Text mode")
    logger.info("Type a query and press Enter. Commands: reset | quit")

    while True:
        try:
            query = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "reset":
            assistant.reset()
            continue

        answer = assistant.ask(query)
        logger.info("SmartDesk-AI: %s", answer)

    logger.info("Goodbye — have a good Day!")


# ── Entrypoint ──────────────────────────────────────────────
agent = SmartdeskAIAssistant()

def main():
    parser = argparse.ArgumentParser(description="SmartDesk-AI Assistant")
    parser.add_argument("--voice",     action="store_true", help="Full voice: mic + speaker")
    parser.add_argument("--voice-out", action="store_true", help="Text input + voice output")
    parser.add_argument("--tts-voice", default="nova",      help="TTS voice (default: nova)")
    args = parser.parse_args()

    speak = args.voice or args.voice_out
    run_text_loop(agent)



if __name__ == "__main__":
    main()
