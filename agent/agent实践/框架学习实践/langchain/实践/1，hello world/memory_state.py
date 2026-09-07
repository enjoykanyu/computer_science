"""
示例1: 状态模式 (state_schema) - 简单扩展数据字段
作用: 存储用户偏好，供后续使用
"""
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from typing import TypedDict

# 1️⃣ 定义自定义状态（扩展字段）
class CustomState(TypedDict):
    messages: list  # 必须包含 messages
    user_preferences: dict  # 自定义字段：用户偏好

# 2️⃣ 定义工具 - 在工具中读取用户偏好
@tool
def get_weather(city: str, runtime) -> str:
    """查询城市天气，会根据用户偏好调整回答风格"""
    # 从 runtime.state 读取用户偏好
    prefs = runtime.state.get("user_preferences", {})
    style = prefs.get("style", "normal")
    
    weather = f"{city} 今天晴，25°C"
    
    # 根据偏好调整回答
    if style == "technical":
        return f"【技术模式】{weather}。气温25°C，相对湿度60%，紫外线指数中等。"
    elif style == "simple":
        return f"【简洁模式】{weather}，适合出门！"
    else:
        return f"{weather}，天气不错！"

# 3️⃣ 初始化模型
model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

# 4️⃣ 创建 Agent - 使用自定义状态
agent = create_agent(
    model,
    tools=[get_weather],
    state_schema=CustomState  # 指定状态模式
)

# 5️⃣ 运行 - 传入用户偏好
print("=" * 50)
print("测试1: 技术风格偏好")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}],
    "user_preferences": {"style": "technical"}
})
print(result["messages"][-1].content)

print("\n" + "=" * 50)
print("测试2: 简洁风格偏好")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "武汉天气怎么样？"}],
    "user_preferences": {"style": "simple"}
})
print(result["messages"][-1].content)

print("\n" + "=" * 50)
print("测试3: 默认风格（无偏好）")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "杭州天气怎么样？"}]
})
print(result["messages"][-1].content)
