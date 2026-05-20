from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

class MyState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "mimo-v2.5-pro"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.xiaomimicrobot.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=0.7,
)

SYSTEM_PROMPT = "你是一个友好的AI助手，请用简洁的中文回答问题。"
EXIT_KEYWORDS = {"退出", "再见", "拜拜", "quit", "exit", "bye", "结束"}

def llm_node(state: MyState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MyState)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)  # ← 传入!

def chat():
    print("💬 多轮对话（输入 '退出' 结束）")
    print("=" * 50)
    thread_id = "user-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT)]}, config)

    while True:
        user_text = input("\n👤 你: ").strip()
        if not user_text:
            continue

        if any(kw in user_text.lower() for kw in EXIT_KEYWORDS):
            print("🤖 AI: 感谢聊天，期待下次再见！👋")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config
        )

        ai_msg = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        print(f"🤖 AI: {ai_msg.content}")

def demo_all_features():
    print("=" * 60)
    print("1️⃣  自动记忆：同一 thread_id 自动续接对话")
    print("=" * 60)

    config_a = {"configurable": {"thread_id": "alice"}}

    graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT)]}, config_a)
    r1 = graph.invoke({"messages": [HumanMessage(content="我叫小明")]}, config_a)
    print(f"第1轮: {last_ai(r1)}")

    r2 = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config_a)
    print(f"第2轮: {last_ai(r2)}")

    config_b = {"configurable": {"thread_id": "bob"}}

    graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT)]}, config_b)
    r3 = graph.invoke({"messages": [HumanMessage(content="我叫小红")]}, config_b)
    print(f"Bob的第1轮: {last_ai(r3)}")

    # Alice 的记忆不受影响
    r4 = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config_a)
    print(f"Alice的续聊: {last_ai(r4)}")  # 还是"小明"

    # Bob 的记忆也是独立的
    r5 = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config_b)
    print(f"Bob的续聊:   {last_ai(r5)}")  # 是"小红"

    history = list(graph.get_state_history(config_a))
    # 分支回放 — 从某个历史点重新开始
    print("\n" + "=" * 60)
    print("分支回放：从某个历史checkpoint重新开始")
    print("=" * 60)

    # 找到 Alice 刚说完"我叫小明"的那个checkpoint
    for state in history:
        if len(state.values.get("messages", [])) == 2:  # system + 第1轮human
            fork_config = state.config
            break

    print(f"从checkpoint分支，重新提问...")
    r6 = graph.invoke(
        {"messages": [HumanMessage(content="我不叫小明，我叫大明！")]},
        fork_config  # ← 从旧的checkpoint继续
    )
    print(f"分支后的回复: {last_ai(r6)}")


def last_ai(result):
    return [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content



if __name__ =="__main__":
    demo_all_features()