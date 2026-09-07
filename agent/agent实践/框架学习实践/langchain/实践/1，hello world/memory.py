# from langchain.agents import AgentState, create_agent
# from langchain.agents.middleware import AgentMiddleware
# from typing import Any

# # 定义状态：存储用户偏好和对话历史
# class CustomState(AgentState):
#     user_preferences: dict = {}  # 用户偏好
#     call_count: int = 0          # 调用次数

# # 自定义中间件
# class PreferenceMiddleware(AgentMiddleware):
#     state_schema = CustomState
    
#     def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
#         """在调用模型前，注入用户偏好"""
        
#         # 获取用户偏好
#         style = state.user_preferences.get("style", "normal")
        
#         # 根据偏好修改系统提示
#         if style == "technical":
#             extra_prompt = "请用技术术语详细解释。"
#         elif style == "simple":
#             extra_prompt = "请用通俗语言简单解释。"
#         else:
#             extra_prompt = ""
        
#         # 更新调用计数
#         state.call_count += 1
#         print(f"第 {state.call_count} 次调用")
        
#         # 返回额外的上下文
#         return {"extra_system_prompt": extra_prompt}

# # 创建 Agent
# agent = create_agent(
#     model,
#     tools=tools,
#     middleware=[PreferenceMiddleware()]
# )

# # 使用：带上用户偏好
# result = agent.invoke({
#     "messages": [{"role": "user", "content": "什么是神经网络？"}],
#     "user_preferences": {"style": "technical"}  # 偏好技术解释
# })



from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})