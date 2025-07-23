import os
import math
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent   
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



os.environ["GOOGLE_API_KEY"] = ""

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Tools
def square_root(a: str) -> float:
    if int(a) < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")
    return math.sqrt(int(a))
tools = [Tool.from_function(func=square_root, name="Squarer", description="Use this tool to calculate the square root of a non-negative number.")]


# Prompt
prompt = hub.pull("hwchase17/react-chat")


# Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Executor with memory
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Store chat history
chat_history = ChatMessageHistory()

# Create the agent with history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Converse
while True:
    user_input = input("User: ")
    if user_input.lower() in {"quit", "exit"}:
        break
    try:
        
        response = agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "any_string"}}
        )
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")