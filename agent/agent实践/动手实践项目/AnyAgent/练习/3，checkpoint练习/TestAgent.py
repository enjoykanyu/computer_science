from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def echo(state: MessagesState):
    print("current messages:", state["messages"])
    # 在 MessagesState 下，返回的 list 也会被自动追加，而不是覆盖
    return {"messages": [AIMessage(content="echo: got it")]}

# 图
builder = StateGraph(MessagesState)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)

# 加 checkpoint
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 调用
config = {"configurable": {"thread_id": "demo-1"}}

print("--- 第一次调用 ---")
# LangChain 的 HumanMessage 对象传入
result1 = graph.invoke({"messages": [HumanMessage(content="hello")]}, config)
print(result1)

print("--- 第二次调用 ---")
result2 = graph.invoke({"messages": [HumanMessage(content="world")]}, config)
print(result2)