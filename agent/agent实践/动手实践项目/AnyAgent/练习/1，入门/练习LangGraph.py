from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages  # ← 改用 add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY", "")
baseUrl = os.getenv("OPENAI_BASE_URL", "https://api.xiaomimimo.com/v1")
model = os.getenv("OPENAI_MODEL", "mimo-v2.5-pro")

# 定义图编译状态
class MyState(TypedDict):
    messages:Annotated[list,add_messages] # 消息追加策略，消息不会被覆盖
    count:int

llm = ChatOpenAI(
    model=model,
    base_url=baseUrl,
    api_key=apiKey,
    temperature=0.7
)

def greet_node(state: MyState) -> dict:
    """问候节点：构造用户消息"""
    print(f"【greet_node】收到状态: {state}")

    from langchain_core.messages import HumanMessage

    # 将用户输入包装为 HumanMessage 追加到消息列表
    user_input = f"你好！这是第 {state.get('count', 0) + 1} 次对话，请简短介绍一下你自己。"
    return {
        "messages": [HumanMessage(content=user_input)],
        "count": state.get("count", 0) + 1,
    }


def llm_node(state: MyState) -> dict:
    """
    ★ 大模型节点 ★
    核心逻辑：将历史消息发给 LLM，获取回复
    """
    print(f"【llm_node】收到状态，消息数: {len(state['messages'])}")

    # 调用大模型，传入完整消息历史
    # LLM 会根据所有历史消息生成回复
    response = llm.invoke(state["messages"])

    print(f"【llm_node】LLM 回复: {response.content[:50]}...")

    # 返回 AI 的回复（AIMessage 会被追加到 messages）
    return {
        "messages": [response],  # response 是 AIMessage
    }


def farewell_node(state: MyState) -> dict:
    """告别节点：总结对话"""
    print(f"【farewell_node】收到状态: {state}")

    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="谢谢，再见！")],
        # 不返回 count，保持当前值
    }

# 构建图
builder = StateGraph(MyState)
builder.add_node("greet",greet_node)
builder.add_node("llm",llm_node)
builder.add_node("farewell",farewell_node)

builder.add_edge(START,"greet")
builder.add_edge("greet","llm")
builder.add_edge("llm","farewell")
builder.add_edge("farewell",END)

# 编译图
graph= builder.compile()

# 执行

if __name__ =="__main__":
    initial_state = {
        "messages": [],   # 空消息历史，由节点逐步填充
        "count": 0,
    }

    print("=" * 50)
    print("初始状态:", initial_state)
    print("=" * 50)

    final_state = graph.invoke(initial_state)

    print("=" * 50)
    print("最终状态:", final_state)
    print("=" * 50)

    # 打印完整对话历史
    print("\n📜 完整对话历史:")
    for i, msg in enumerate(final_state["messages"], 1):
        role = msg.__class__.__name__    # HumanMessage / AIMessage
        print(f"  {i}. [{role}] {msg.content}")