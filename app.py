import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph
from main import SmartdeskAIAssistant

assistant = SmartdeskAIAssistant()

st.set_page_config(page_title="SmartDesk AI Agent", page_icon="🤖")
st.title("SmartDesk AI Agent")

# Initialize session state variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle new user input
if prompt := st.chat_input("Ask the agent something..."):
    # Display and store user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the LangGraph agent
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #result = graph.invoke({"messages": [user_msg]}, config=config)
            #ai_msg = result["messages"][-1]
            #st.markdown(ai_msg.content)
            result = assistant.ask(prompt)
            st.markdown(result)
    # Store assistant response
    st.session_state.messages.append(result)