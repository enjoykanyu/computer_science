from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver   # ← 新增
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()


# ─── 状态定义 ───────────────────────────────────────
class MyState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── LLM ────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "mimo-v2.5-pro"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.xiaomimicrobot.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=0.7,
)

SYSTEM_PROMPT = "你是一个友好的AI助手，请用简洁的中文回答问题。"
EXIT_KEYWORDS = {"退出", "再见", "拜拜", "quit", "exit", "bye", "结束"}


# ─── 节点 ───────────────────────────────────────────
def llm_node(state: MyState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ─── 图 ─────────────────────────────────────────────
builder = StateGraph(MyState)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

# 编译时传入 checkpointer
checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)


# 多轮对话（不再手动管理 messages）
def chat():
    print("💬 多轮对话（输入 '退出' 结束）")
    print("=" * 50)
    # checkpoint 按 thread_id 自动管理状态
    thread_id = "user-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    # 初始化：注入 system prompt
    graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT)]}, config)

    while True:
        user_text = input("\n👤 你: ").strip()
        if not user_text:
            continue

        if any(kw in user_text.lower() for kw in EXIT_KEYWORDS):
            print("🤖 AI: 感谢聊天，期待下次再见！👋")
            break

        # 只传当前消息，历史由 checkpoint 自动携带！
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config  # 同一个 thread_id → 自动加载历史
        )

        ai_msg = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        print(f"🤖 AI: {ai_msg.content}")

def demo_all_features():
    # 不用手动管理消息历史
    config_a = {"configurable": {"thread_id": "alice"}}
    graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT)]}, config_a)
    r1 = graph.invoke({"messages": [HumanMessage(content="我是谁")]}, config_a)
    print(f"线程a的第1轮: {last_ai(r1)}")
    r2 = graph.invoke({"messages": [HumanMessage(content="我叫小明")]}, config_a)
    print(f"线程a的第2轮: {last_ai(r2)}")
    r3 = graph.invoke({"messages": [HumanMessage(content="我是谁？")]}, config_a)
    print(f"线程a的第3轮: {last_ai(r3)}")

    print("=" * 60)
    print("多线程隔离对于隔离")
    config_b = {"configurable": {"thread_id": "bob"}}
    r4 = graph.invoke({"messages": [HumanMessage(content="我叫做小红")]}, config_b)
    print(f"线程b的第1轮: {last_ai(r4)}")
    r5 = graph.invoke({"messages": [HumanMessage(content="我是谁？")]}, config_a)
    print(f"线程a的第4轮: {last_ai(r5)}")
    r6 = graph.invoke({"messages": [HumanMessage(content="我是谁？")]}, config_b)
    print(f"线程b的2轮: {last_ai(r6)}")

def last_ai(result):
    return [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content


if __name__ =="__main__":
    demo_all_features()
