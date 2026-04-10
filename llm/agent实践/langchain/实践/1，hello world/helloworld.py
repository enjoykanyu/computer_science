from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

agent = create_react_agent(
    llm,
    tools=[get_weather],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

messages = result.get("messages", [])
for msg in messages[-1:]:  # 只打印最后一条 AI 回复
    if hasattr(msg, "content") and msg.content:
        print(msg.content)