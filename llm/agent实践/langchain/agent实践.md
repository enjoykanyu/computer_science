# langchain框架
LangChain 是一种简单的方式，可以开始构建完全由 LLMs 助理和应用程序。只需 10 条代码行，您就可以连接到 OpenAI、Anthropic、Google 和 更多 。LangChain 提供预构建的代理架构和模型集成，可帮助您快速入门，并将 LLMs 无缝地集成到代理和应用程序中。（来源于官网介绍）

这里采用了ollama 得install pip install langchain-ollama

### 第一个程序（hello world）
+ 快速上手，引入langchain

pip install -U langchain

实践

```python
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
```



打印

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775836454436-c1ae9da6-933c-44e7-9ec9-8238bbf82870.png" width="1590" title="" crop="0,0,1,1" id="szOtB" class="ne-image">

### **<font style="color:rgb(3, 7, 16);"> 构建一个现实世界的代理（工具扩展）</font>**
1，系统提示词



```python
SYSTEM_PROMPT = """你是气象预报专家，说话喜欢双关语。

你有两个工具可以使用：

- get_weather_for_location：使用这个工具获取特定地点的天气
- get_user_location：使用这个工具获取用户的位置

如果用户询问天气，请确保你知道位置。如果你能从问题中判断他们指的是自己所在的地方，使用 get_user_location 工具来查找他们的位置。
```



2，定义工具类



```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

注意：<font style="color:rgb(45, 51, 25);">工具应该有良好的文档说明：它们的名称、描述和参数名称都将成为模型提示的一部分。LangChain 的 </font>`<font style="color:rgb(45, 51, 25);">@tool</font>`<font style="color:rgb(45, 51, 25);"> 装饰器添加元数据，并通过 </font>`<font style="color:rgb(45, 51, 25);">ToolRuntime</font>`<font style="color:rgb(45, 51, 25);"> 参数启用运行时注入。更多内容请参考工具指南。</font>



<font style="color:rgb(45, 51, 25);">3，配置相关模型</font>

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

4，定义响应格式



```python
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None
```

5，增加记忆



```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

6，运行当前的智能体



```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
```



但由于ollama部署的本地大模型规格太小了 用了llama3.2:3b和qwen3:0.6b 大模型调用过程都只调用了一个工具 因此大模型改成了第三方服务商

注意这里先在.env这里存入你的model_key和模型服务地址还有模型名称

+ 下载相关依赖

```shell
pip install --upgrade pip && pip install langchain langchain-core langgraph langchain-ollama langchain-openai
```

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI



# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    print(f"🔧 [工具调用] get_weather_for_location(city='{city}')") 
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    print(f"开始调用工具get_user_location: {runtime.context.user_id}")
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
# tool.py 开头添加
from dotenv import load_dotenv
import os



load_dotenv()  # 自动加载 .env 文件
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

# Configure model
model = ChatOpenAI(
   model=model,
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)


print("首次回答问题")
print(response['structured_response'])

# print("=== Response Keys ===")
# print(response.keys() if hasattr(response, 'keys') else response)

# # 提取回答
# messages = response.get("messages", [])
# if messages:
#     last_msg = messages[-1]
#     print("\n=== 首次回答 ===")
#     print(last_msg.content)
# # ResponseFormat(
# #     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
# #     weather_conditions="It's always sunny in Florida!"
# # )


# # Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print("最终回答问题")
print(response['structured_response'])

# print("=== Response Keys ===")
# print(response.keys() if hasattr(response, 'keys') else response)

# # 提取回答
# messages = response.get("messages", [])
# if messages:
#     last_msg = messages[-1]
#     print("\n=== 最终回答 ===")
#     print(last_msg.content)
# # ResponseFormat(
# #     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
# #     weather_conditions=None
# # )
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775890877512-980a56fb-8e05-4902-b34e-83c72ef7df83.png" width="1004" title="" crop="0,0,1,1" id="cyxUD" class="ne-image">

可以看到成功调用了两个工具

同时第二次回答的时候记住了之前的回答，拥有了记忆

# langgraph框架