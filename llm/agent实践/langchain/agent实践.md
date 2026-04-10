# langchain框架
LangChain 是一种简单的方式，可以开始构建完全由 LLMs 助理和应用程序。只需 10 条代码行，您就可以连接到 OpenAI、Anthropic、Google 和 更多 。LangChain 提供预构建的代理架构和模型集成，可帮助您快速入门，并将 LLMs 无缝地集成到代理和应用程序中。（来源于官网介绍）
### 快速上手

- 引入langchain

pip install -U langchain

实践

这里采用了ollama 得install pip install langchain-ollama

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

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775836454436-c1ae9da6-933c-44e7-9ec9-8238bbf82870.png" width="795" title="" crop="0,0,1,1" id="u84abb334" class="ne-image">

可以看到本次langchain调用成功了


# langgraph框架