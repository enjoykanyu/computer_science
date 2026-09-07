"""
示例2: 中间件模式 (AgentMiddleware) - 主动干预执行流程
作用: 在调用模型前自动注入用户偏好到系统提示
"""
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from typing import Any

# 1️⃣ 定义工具（普通工具，不处理偏好）
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city} 今天晴，25°C，天气不错！"

# 2️⃣ 定义自定义中间件
class PreferenceMiddleware(AgentMiddleware):
    """
    中间件：在调用 LLM 前自动注入用户偏好
    """
    
    def before_model(self, state: dict, runtime) -> dict[str, Any] | None:
        """
        在调用模型前执行
        根据用户偏好修改系统提示
        """
        # 获取用户偏好
        prefs = state.get("user_preferences", {})
        style = prefs.get("style", "normal")
        
        # 根据偏好生成额外的系统提示
        if style == "technical":
            extra_prompt = "你是一个技术专家，回答时使用专业术语，提供详细的技术细节。"
        elif style == "simple":
            extra_prompt = "你是一个简洁的助手，回答时用最简单的语言，不超过两句话。"
        else:
            extra_prompt = "你是一个友好的助手，用自然的语言回答。"
        
        print(f"[中间件] 检测到偏好: {style}")
        print(f"[中间件] 注入系统提示: {extra_prompt[:30]}...")
        
        # 返回额外的上下文（会被合并到系统提示）
        return {"system_prompt_addon": extra_prompt}

# 3️⃣ 初始化模型
model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

# 4️⃣ 创建 Agent - 使用中间件
agent = create_agent(
    model,
    tools=[get_weather],
    middleware=[PreferenceMiddleware()]  # 注册中间件
)

# 5️⃣ 运行测试
print("=" * 50)
print("测试1: 技术风格（中间件自动注入技术专家人设）")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍下langchain框架"}],
    "user_preferences": {"style": "technical"}
})
print(result["messages"][-1].content)

print("\n" + "=" * 50)
print("测试2: 简洁风格（中间件自动注入简洁人设）")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍下langchain框架"}],
    "user_preferences": {"style": "simple"}
})
print(result["messages"][-1].content)

print("\n" + "=" * 50)
print("测试3: 默认风格")
print("=" * 50)
result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍下langchain框架"}]
})
print(result["messages"][-1].content)
