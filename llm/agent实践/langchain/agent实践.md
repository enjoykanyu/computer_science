# langchain框架
LangChain 是一种简单的方式，可以开始构建完全由 LLMs 助理和应用程序。只需 10 条代码行，您就可以连接到 OpenAI、Anthropic、Google 和 更多 。LangChain 提供预构建的代理架构和模型集成，可帮助您快速入门，并将 LLMs 无缝地集成到代理和应用程序中。（来源于官网介绍）
### 快速上手

- 引入langchain

pip install -U langchain

实践

这里采用了ollama 得install pip install langchain-ollama

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="ollama:wen3:0.6b",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```




# langgraph框架