# LangChain框架
LangChain 是一种简单的方式，可以开始构建完全由 LLMs 助理和应用程序。只需 10 条代码行，您就可以连接到 OpenAI、Anthropic、Google 和 更多 。LangChain 提供预构建的代理架构和模型集成，可帮助您快速入门，并将 LLMs 无缝地集成到代理和应用程序中。（来源于官网介绍）

这里采用了ollama 得install pip install langchain-ollama



### 快速实践上手
##### 第一个程序（hello world）
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

##### **<font style="color:rgb(3, 7, 16);">构建一个现实世界的代理（工具扩展）</font>**
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



### 核心组件
##### Agents
<font style="color:rgb(3, 7, 16);">Agents将语言模型与工具结合，创建能够推理任务、决定使用哪些工具，并持续地工作以寻求解决方案的系统。</font>

<font style="color:rgb(3, 7, 16);">LLM 代理在循环中运行工具以实现目标。代理会一直运行，直到满足停止条件——即模型发出最终输出或达到迭代限制。（来源于langchain官网）</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775891660287-c3fe9de9-d438-455c-b14c-c735364c02d0.png" width="403" title="" crop="0,0,1,1" id="uef1e84a7" class="ne-image">



##### model（模型）
<font style="color:rgb(3, 7, 16);">模型是agents的推理引擎，即作为推理执行的llm引擎，它可以通过多种方式指定，支持静态和动态模型选择。</font>  


+ 静态模型

<font style="color:rgb(3, 7, 16);">静态模型在创建代理时配置一次，并在整个执行过程中保持不变。这是最常见和直接的方法。</font>  


例如这样的模型在全过程执行中，模型固定为最开始设置的

```shell
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)
```





+ 动态模型

<font style="color:rgb(3, 7, 16);">动态模型是在</font>运行时<font style="color:rgb(3, 7, 16);">，基于当前</font>状态<font style="color:rgb(3, 7, 16);">以及上下文。这可以实现复杂的路由逻辑和成本优化。</font>

要使用动态模型，请使用 @wrap_model_call 装饰器创建中间件，该中间件会修改请求中的模型

地址（代码文件：[dynamic_model.py](file:///Users/kanyu/Desktop/project/kanyu_server/new_project/computer_science/llm/agent实践/langchain/实践/1，hello%20world/dynamic_model.py)）



```shell
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # 自动加载 .env 文件

basic_model = os.getenv("OPENAI_CHATGPT_BASIC")
advanced_model = os.getenv("OPENAI_CHATGPT_ADVANCED")
base_url = os.getenv("OPENAI_BASE_URL")


basic_model = init_chat_model(
   model="qwen3:0.6b",
    model_provider="ollama",
    temperature=0
)
advanced_model = init_chat_model(
   model="qwen3:1.7b",
    model_provider="ollama",
    temperature=0
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 1:
        print("使用高级模型")
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        print("使用基础模型")
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    middleware=[dynamic_model_selection]
)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "详细介绍下skill和mcp tool的区别"}]},
)
print(response['messages'])
messages = response.get("messages", [])
if messages:
    last_msg = messages[-1]
    print("\n=== 最终回答 ===")
    print(last_msg.content)
```

首先对话轮次只有一次

{"messages": [{"role": "user", "content": "详细介绍下skill和mcp tool的区别"}]},

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775893884204-390ce195-6891-49fb-aa2f-dd646d446914.png" width="510" title="" crop="0,0,1,1" id="u698d6a46" class="ne-image">

可以看到使用了基础模型



修改成两轮对话

    {"messages": [{"role": "user", "content": "详细介绍下skill和mcp tool的区别"}, {"role": "user", "content": "在langchain和langgraph分别实践下"}]},

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775893941977-0d57d574-f506-4678-a541-6166e48a5d64.png" width="492" title="" crop="0,0,1,1" id="u9f4eac9b" class="ne-image">

可以看到切换成了高级模型



##### tools工具
+ 静态工具

<font style="color:rgb(3, 7, 16);">静态工具在创建代理时定义，并在执行过程中保持不变。这是最常见和直接的方法。</font>  
<font style="color:rgb(45, 51, 25) !important;background-color:rgb(212, 232, 168) !important;">工具装饰器可用于自定义工具名称、描述、参数模式和其他属性。</font>

比如像这样的

（来源于langchain官网）

```shell
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

+ 动态工具

<font style="color:rgb(3, 7, 16);">使用动态工具时，代理可用的工具集在运行时进行修改，而不是一开始就全部定义。并非每种工具都适用于每种情况。工具过多可能会使模型不堪重负（导致上下文过载），并增加错误；工具过少则会限制功能。动态工具选择能够根据认证状态、用户权限、功能标志或对话阶段来调整可用工具集</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775894230426-2df84d19-9c3e-4984-80c0-6396a4a21248.png" width="484" title="" crop="0,0,1,1" id="u52be2a8d" class="ne-image">



##### **<font style="color:rgb(3, 7, 16);">System prompt  系统提示</font>**
<font style="color:rgb(3, 7, 16);">像这样在创建agent的时候给出system_prompt</font>

```shell
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">system_prompt</font>`<font style="color:rgb(3, 7, 16);"> 参数可以接受一个 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">str</font>`<font style="color:rgb(3, 7, 16);"> 或一个 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">SystemMessage</font>`<font style="color:rgb(3, 7, 16);"> 。使用 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">SystemMessage</font>`<font style="color:rgb(3, 7, 16);"> 可以让你对提示结构有更多的控制，这对于特定提供者的功能（如 Anthropic 的提示缓存）很有用：</font>

比如

```shell
system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
```

<font style="color:rgb(3, 7, 16);">带有 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">{"type": "ephemeral"}</font>`<font style="color:rgb(3, 7, 16);"> 的 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">cache_control</font>`<font style="color:rgb(3, 7, 16);"> 字段会指示 Anthropic 缓存该内容块，从而减少使用相同系统提示的重复请求的延迟和成本。</font>

<font style="color:rgb(3, 7, 16);"></font>

##### <font style="color:rgb(3, 7, 16);">name名称</font>
<font style="color:rgb(3, 7, 16);">为代理设置可选的 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">name</font>`<font style="color:rgb(3, 7, 16);"> 。在多代理系统中将代理作为子图添加时，这用作节点标识符</font>

<font style="color:rgb(3, 7, 16);">比如像这样</font>

<font style="color:rgb(3, 7, 16);"></font>

```shell
agent = create_agent(
    model,
    tools,
    name="research_assistant"
)
```



##### <font style="color:rgb(3, 7, 16);">Invocation调用</font>
<font style="color:rgb(3, 7, 16);">您可以通过向其 </font>[<font style="color:rgb(17, 24, 39);">State</font>](https://docs.langchain.com/oss/python/langgraph/graph-api#state)<font style="color:rgb(3, 7, 16);"> 传递更新来调用代理。所有代理的</font>[<font style="color:rgb(17, 24, 39);">消息序列</font>](https://docs.langchain.com/oss/python/langgraph/use-graph-api#messagesstate)<font style="color:rgb(3, 7, 16);">都包含在它们的状态中；要调用代理，请传递新消息：</font>

```shell
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```



##### <font style="color:rgb(3, 7, 16);">Structured output  结构化输出</font>
<font style="color:rgb(3, 7, 16);">在某些情况下，您可能希望代理返回特定格式的输出。LangChain 提供了通过 </font>[<font style="color:rgb(17, 24, 39);">response_format</font>](https://reference.langchain.com/python/langchain/agents/factory/create_agent)<font style="color:rgb(3, 7, 16);"> 参数构造输出的策略。</font>



###### <font style="color:rgb(3, 7, 16);">ToolStrategy  工具策略</font>
`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">ToolStrategy</font>`<font style="color:rgb(3, 7, 16);"> 使用人工工具调用来生成结构化输出。这适用于任何支持工具调用的模型。当没有或不可靠的提供程序原生结构化输出（通过 </font>[<font style="color:rgb(17, 24, 39);">ProviderStrategy</font>](https://docs.langchain.com/oss/python/langchain/agents#providerstrategy)<font style="color:rgb(3, 7, 16);">）时，应使用 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">ToolStrategy</font>`<font style="color:rgb(3, 7, 16);">。</font>



```shell
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```



###### <font style="color:rgb(3, 7, 16);">ProviderStrategy  提供者策略</font>
`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">ProviderStrategy</font>`<font style="color:rgb(3, 7, 16);"> 使用模型提供程序的原生结构化输出生成。这更可靠，但仅适用于支持原生结构化输出的提供程序：</font>

<font style="color:rgb(3, 7, 16);"></font>

```shell
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4.1",
    response_format=ProviderStrategy(ContactInfo)
)
```



从 `langchain 1.0` 开始，如果模型支持原生结构化输出，简单地传递一个模式（例如 `response_format=ContactInfo`）将默认为 `ProviderStrategy`，否则将回落到 `ToolStrategy`。



##### Memory记忆
agents通过消息状态自动维护对话历史记录。还可以配置代理使用自定义状态模式来记住对话期间的其他信息。

存储在状态中的信息可以被视为代理的</font>[<font style="color:rgb(17, 24, 39);">短期记忆</font>]<font style="color:rgb(3, 7, 16);">：  
自定义状态模式必须作为 `TypedDict` 扩展 AgentState  

有两种方法可以定义自定义状态：  
通过 middleware（首选）、通过create_agent</font>


###### Defining state via middleware通过中间件定义状态

地址（代码文件：[memory_middleware.py](file:///Users/kanyu/Desktop/project/kanyu_server/new_project/computer_science/llm/agent实践/langchain/实践/1，hello%20world/memory_middleware.py)）

```shell
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

```



执行流程

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775898111595-76c6dd45-3597-4c15-bed2-5f226d664f36.png" width="499" title="" crop="0,0,1,1" id="ue16dc6e2" class="ne-image">



输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775899287715-955d0a0b-4e90-4841-b3b0-dcbda9f571f8.png" width="472" title="" crop="0,0,1,1" id="uf956af53" class="ne-image">

可以看到根据三个不同的风格输出



###### <font style="color:rgb(3, 7, 16);">通过 </font>`<font style="color:rgb(3, 7, 16);background-color:rgba(238, 238, 239, 0.5);">state_schema</font>`<font style="color:rgb(3, 7, 16);"> 定义状态</font>
<font style="color:rgb(3, 7, 16);">使用 </font>[<font style="color:rgb(17, 24, 39);">state_schema</font>](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema)<font style="color:rgb(3, 7, 16);"> 参数作为定义仅在工具中使用的自定义状态的快捷方式。</font>

地址（代码文件：[memory_state.py](file:///Users/kanyu/Desktop/project/kanyu_server/new_project/computer_science/llm/agent实践/langchain/实践/1，hello%20world/memory_state.py)）

```shell
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

```



**<font style="color:rgb(3, 7, 16);">输出</font>**

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775898835667-249506a7-07bf-471e-8665-f746682100fb.png" width="468" title="" crop="0,0,1,1" id="u84b2c8ad" class="ne-image">

**<font style="color:rgb(3, 7, 16);"></font>**

<font style="color:rgb(3, 7, 16);">会依据不同的风格输出</font>

**<font style="color:rgb(3, 7, 16);"></font>**

##### <font style="color:rgb(3, 7, 16);">Streaming 流式输出</font>
<font style="color:rgb(3, 7, 16);">代理可以通过 </font>`<font style="color:rgb(17, 24, 39);background-color:rgba(238, 238, 239, 0.5);">invoke</font>`<font style="color:rgb(3, 7, 16);"> 调用，以获取最终响应。如果代理执行多个步骤，这可能需要一段时间。为了显示中间进度，我们可以按需回传消息。</font>

<font style="color:rgb(3, 7, 16);">地址（代码文件：</font>[streaming.py](file:///Users/kanyu/Desktop/project/kanyu_server/new_project/computer_science/llm/agent实践/langchain/实践/1，hello%20world/streaming.py)<font style="color:rgb(3, 7, 16);">）</font>

```shell
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
```



可以在整个搜索过程中自定义打印，<font style="color:rgb(3, 7, 16);">显示中间进度</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775899976208-5e74c7a0-1065-4a16-ab1e-5b238d8591c5.png" width="696" title="" crop="0,0,1,1" id="u29899dcc" class="ne-image">

<font style="color:rgb(3, 7, 16);"></font>



##### <font style="color:rgb(3, 7, 16);">Middleware中间件</font>
**<font style="color:rgb(3, 7, 16);"></font>**

**<font style="color:rgb(3, 7, 16);"></font>**

**<font style="color:rgb(3, 7, 16);"></font>**

**<font style="color:rgb(3, 7, 16);"></font>**

**<font style="color:rgb(3, 7, 16);"></font>**


# langgraph框架