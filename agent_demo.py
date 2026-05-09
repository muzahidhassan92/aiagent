
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools

st.title("AI Agent")

with st.sidebar:
    st.header("Agent Configuration")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the agent.")
    st.stop()

# ------------------------------
# 1. LLM setup
# ------------------------------
if OPENAI_API_KEY:   
    OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'")
else:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)


# ------------------------------
# 2. Tools (Wikipedia + DuckDuckGo)
# ------------------------------
tools = load_tools(["wikipedia", "ddg-search"])


# ------------------------------
# 3. ReAct-style system prompt
# ------------------------------
react_system_prompt = """
You are a ReAct-style AI agent.

Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search).
3. OBSERVATION: Read the tool result and decide the next step.

Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer to the user.
"""


# ------------------------------
# 4. Create Agent (new v1 API)
# ------------------------------

#TODO: Create Agent
agent = create_agent(llm, tools, system_prompt=react_system_prompt)

# ------------------------------
# 5. Streamlit UI
# ------------------------------


task = st.text_input("Assign me a task")

if task:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": task}
        ]
    })

    if isinstance(result, dict) and "messages" in result:
        final_answer = result["messages"]  # Get the last message as the final answer
        st.write("Final Answer:")
        for message in final_answer:
            st.write(message.content)
