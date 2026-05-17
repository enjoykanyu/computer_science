from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
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
    """唯一节点：调用 LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ─── 图：极简单轮 ───────────────────────────────────
builder = StateGraph(MyState)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()


# ─── 外部循环控制多轮 ────────────────────────────────
def chat():
    print("💬 多轮对话（输入 '退出' 结束）")
    print("=" * 50)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        # ✅ 用户输入在外部，不在节点里
        user_text = input("\n👤 你: ").strip()
        if not user_text:
            continue

        # ✅ 退出检测在调 LLM 之前
        if any(kw in user_text.lower() for kw in EXIT_KEYWORDS):
            print("🤖 AI: 感谢聊天，期待下次再见！👋")
            break

        # ✅ 每轮 invoke 一次图，传入完整历史
        messages.append(HumanMessage(content=user_text))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]  # 更新历史（含 AI 回复）

        # 取最后一条 AI 消息打印
        ai_msg = [m for m in messages if isinstance(m, AIMessage)][-1]
        print(f"🤖 AI: {ai_msg.content}")

    # 打印统计
    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    print(f"\n📊 共 {human_count} 轮对话")


if __name__ == "__main__":
    chat()