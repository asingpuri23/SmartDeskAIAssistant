from db_utils import insert_chat_history, get_chat_history, create_chat_history
from fastapi import FastAPI, HTTPException
from utils import get_or_create_session_id, history_to_lc_messages, append_message
from pydantic_models import QueryInput, QueryResponse
from logger import setup_logger
from main import SmartdeskAIAssistant
from langchain_core.messages import HumanMessage, AIMessage

agent = SmartdeskAIAssistant()
logger = setup_logger("smartdesk-ai.fastapimain")
app = FastAPI()
create_chat_history()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """
    Main chat endpoint using the LangGraph agent with routing, RAG, and web search capabilities.
    """
    session_id = get_or_create_session_id(query_input.session_id)
    logger.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    try:
        # Convert chat history to LangChain messages
        chat_history = get_chat_history(session_id)
        messages = history_to_lc_messages(chat_history)
        # Add current user message

        messages = append_message(messages, HumanMessage(content=query_input.question))
        # Invoke the LangGraph agent
        # config = {"configurable": {"thread_id": session_id}}
        
        #result = agent.graph.invoke(
        #    {"messages": messages}
        #)
        result = agent.ask(query_input.question)

        # Get the last AI message
        last_message = next((m for m in reversed(result["messages"])
                           if isinstance(m, AIMessage)), None)

        if last_message:
            answer = last_message.content
        else:
            answer = "I apologize, but I couldn't generate a response at this time."

        # Store the conversation
        insert_chat_history(session_id, query_input.question, answer, query_input.model.value)
        logger.info(f"Session ID: {session_id}, AI Response: {answer}")

        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
