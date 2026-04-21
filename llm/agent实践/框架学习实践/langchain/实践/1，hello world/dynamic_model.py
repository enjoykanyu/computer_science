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
    {"messages": [{"role": "user", "content": "详细介绍下skill和mcp tool的区别"}, {"role": "user", "content": "在langchain和langgraph分别实践下"}]},
)
print(response['messages'])
messages = response.get("messages", [])
if messages:
    last_msg = messages[-1]
    print("\n=== 最终回答 ===")
    print(last_msg.content)