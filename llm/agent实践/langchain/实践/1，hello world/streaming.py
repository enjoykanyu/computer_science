from langchain.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent



llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

agent = create_agent(
    llm,
    tools=[],
)

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "搜索下 AI 相关的新闻还有总结下搜索到的内容"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")