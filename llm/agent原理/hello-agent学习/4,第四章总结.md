### 封装基本大模型调用
采用封装ollama基本客户端调用函数

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        apiKey = os.environ.get("OLLAMA_API_KEY", "ollama")  # Ollama不需要真实密钥
        baseUrl = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")  # 可选: deepseek-r1:8b, qwen3:1.7b
    
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775445668474-175eb51c-13c4-43a0-8b28-a8b0fd9c243a.png" width="444" title="" crop="0,0,1,1" id="u5bda099d" class="ne-image">

可以看到当前封装成功，可以成功调用对应的本地ollama部署大模型



### reAct模式
没有reAct的时候 一类采用的微思维链的模式，会推理和思考，但缺少行动能力；一类采用的纯行动，没有推理和思考

reAct模式：

thought（思考）：思考下一步的执行规划动作

action（动作）：选择需调用的工具去执行

observing（观察）：观察工具是否执行完成了

reAct模式会不断重复这个过程直到完成预定设置的任务



当然reAct模式还需封装工具



### 工具的封装
注意这里没有采用goggle搜索改成了tavily

```python
from typing import Dict, Any
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载 .env 文件中的环境变量
load_dotenv()
def search(query: str) -> str:
    """
    一个基于Tavily的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [Tavily] 网页搜索: {query}")
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "错误:TAVILY_API_KEY 未在 .env 文件中配置。"

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", include_answer=True)
        
        # 智能解析:优先寻找最直接的答案
        if response.get("answer"):
            return response["answer"]
        
        if response.get("results"):
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('content', '')}"
                for i, res in enumerate(response["results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['macbook最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "vivo最新的手机旗舰型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
       
```



返回

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775455193375-14287c7d-630e-4a3e-a599-4eef746cad1f.png" width="456" title="" crop="0,0,1,1" id="u92c05237" class="ne-image">


