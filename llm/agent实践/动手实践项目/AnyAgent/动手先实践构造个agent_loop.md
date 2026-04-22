# 用LangChain构建第一个agent_loop
框架选择langchain、langgraph

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
# ─── 1. 定义工具 ───
@tool
def calculator(expression: str) -> str:
  """计算数学表达式。输入如 '2+3*4'"""
  try:
    return str(eval(expression))
  except Exception as e:
    return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
  """获取指定城市的天气信息"""
  return f"{city}今天晴朗，25°C"

tools = [calculator, get_weather]
load_dotenv()  # 自动加载 .env 文件
apiKey = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama不需要真实密钥
baseUrl = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # 可选: deepseek-r1:8b, qwen3:1.7b

# Configure model
llm = ChatOpenAI(
  model=model,
  api_key=apiKey,
  base_url=baseUrl,
  temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ─── 2. Agent Loop 核心 ───
async def agent_loop(user_input: str, max_turns: int = 10):
  messages = [HumanMessage(content=user_input)]

  for turn in range(max_turns):
    print(f"\n--- Turn {turn + 1} ---")

    # 调用 LLM
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # 检查是否有工具调用
    if response.tool_calls:
      print(f"🔧 模型决定调用 {len(response.tool_calls)} 个工具:")
      for tc in response.tool_calls:
        print(f"   → {tc['name']}({tc['args']})")

      # 执行每个工具
      for tc in response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        # 找到对应工具并执行
        selected_tool = next(t for t in tools if t.name == tool_name)
        result = selected_tool.invoke(tool_args)

        print(f"   ← {result}")
        messages.append(
          ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
      # 继续下一轮循环
    else:
      # 无工具调用 → 返回最终答案
      print(f"✅ 最终回答: {response.content}")
      return response.content

  return "达到最大轮次限制"

# ─── 3. 运行 ───
if __name__ == "__main__":
  asyncio.run(agent_loop("北京和上海的天气怎么样？另外帮我算一下 123*456"))
```

注意这里使用的llm_with_tools = llm.bind_tools(tools)  
llm_with_tools.ainvoke和常规的使用create_agent.invoke有区别

其中.bind_tools()为LLM绑定工具  
ainvoke() 支持高并发场景

+ 执行流程  
  <img src="img.png" title="null" crop="0,0,1,1" id="RQI4M" class="ne-image"><img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776497025755-01a82d18-f85e-4972-a336-45e23bda2266.png" width="852" title="" crop="0,0,1,1" id="uc79ef8fa" class="ne-image">



+ 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776519436769-cd982f84-99d8-4d71-bdf6-ecb1684a7bc2.png" width="492" title="" crop="0,0,1,1" id="ua3b7b387" class="ne-image">



# 重构成LangGraph
```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
import operator
import os
# ─── 1. 定义工具 ───
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '2+3*4'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}今天晴朗，25°C"

tools = [calculator, get_weather]
load_dotenv()  # 自动加载 .env 文件
apiKey = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama不需要真实密钥
baseUrl = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # 可选: deepseek-r1:8b, qwen3:1.7b

# Configure model
llm = ChatOpenAI(
    model=model,
    api_key=apiKey,
    base_url=baseUrl,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ─── 2. Agent Loop 核心 ───
async def agent_loop(user_input: str, max_turns: int = 10):
    messages = [HumanMessage(content=user_input)]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        # 调用 LLM
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 检查是否有工具调用
        if response.tool_calls:
            print(f"🔧 模型决定调用 {len(response.tool_calls)} 个工具:")
            for tc in response.tool_calls:
                print(f"   → {tc['name']}({tc['args']})")

            # 执行每个工具
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                # 找到对应工具并执行
                selected_tool = next(t for t in tools if t.name == tool_name)
                result = selected_tool.invoke(tool_args)

                print(f"   ← {result}")
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            # 继续下一轮循环
        else:
            # 无工具调用 → 返回最终答案
            print(f"✅ 最终回答: {response.content}")
            return response.content

    return "达到最大轮次限制"



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turns: Annotated[int, operator.add]

def call_model(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "turns": 1}

def execute_tools(state):
    """执行工具调用"""
    last = state["messages"][-1]   
    print(f"🔧 模型决定调用 {len(last.tool_calls)} 个工具:")
    results = []
    for tc in last.tool_calls:
        selected = next(t for t in tools if t.name == tc["name"])
        print(f"   → {tc['name']}({tc['args']})工具 {selected.name}")
        result = selected.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if hasattr(last, 'tool_calls') and last.tool_calls else END
#编排图节点
graph = StateGraph(AgentState) # 创建图，指定状态类型为 AgentState
graph.add_node("model", call_model) # 添加 model 节点
graph.add_node("tools", execute_tools) # 添加 tools 节点
graph.set_entry_point("model") # 设置入口点为 model
graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END}) # 添加条件边，根据 should_continue 判断是否继续调用 tools
graph.add_edge("tools", "model") #增加边 tools 节点执行完后， 无条件 回到 model
app = graph.compile() # 编译图 把声明式的图编译成可执行的 app

def chat(user_input: str) -> str:
    """与助手对话"""
    print(f"\n👤 用户：{user_input}")
    
    messages = [HumanMessage(content=user_input)]
    result = app.invoke({"messages": messages})
    
    return result["messages"][-1].content
# ─── 3. 运行 ───
if __name__ == "__main__":
    # asyncio.run(agent_loop("北京和上海的天气怎么样？另外帮我算一下 123*456"))
    # print(chat("北京和上海的天气怎么样？另外帮我算一下 2的32次方"))
    result = chat("帮我算 123*900等于多少呢")
    print(result)

```



+ 流程图

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776519534832-7b72101e-4559-40dc-b30f-455751c2e66a.png" width="374" title="" crop="0,0,1,1" id="u4bffc9da" class="ne-image">



+ 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776519386510-c8859c80-7de0-4868-8abb-4b87c06f76ae.png" width="422" title="" crop="0,0,1,1" id="uea73c990" class="ne-image">

+ 各个节点作用

```python
graph = StateGraph(AgentState) # 创建图，指定状态类型为 AgentState
graph.add_node("model", call_model) # 添加 model 节点
graph.add_node("tools", execute_tools) # 添加 tools 节点
graph.set_entry_point("model") # 设置入口点为 model
graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END}) # 添加条件边，根据 should_continue 判断是否继续调用 tools
graph.add_edge("tools", "model") #增加边 tools 节点执行完后， 无条件 回到 model
app = graph.compile() # 编译图 把声明式的图编译成可执行的 app
```



# tool系统


```python
import asyncio
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

CONCURRENCY_SAFE = {"read_file", "grep_search", "list_files"}
READONLY_TOOLS = {"read_file", "grep_search", "list_files", "web_search"}

@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    try: return open(path).read()[:5000]
    except Exception as e: return f"错误: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """写入文件"""
    try:
        open(path, 'w').write(content)
        return f"已写入 {path}"
    except Exception as e: return f"错误: {e}"

@tool
def bash(command: str) -> str:
    """执行shell命令"""
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout[:5000] or result.stderr

tools = [read_file, write_file, bash]

def check_permission(tool_name, args, mode="default"):
    if mode == "plan" and tool_name not in READONLY_TOOLS:
        return {"action": "deny", "message": "Plan模式只允许只读工具"}
    if tool_name == "bash" and any(w in args.get("command","") for w in ["rm ", "delete", "drop"]):
        return {"action": "confirm", "message": f"⚠️ 危险命令: {args['command']}"}
    return {"action": "allow"}

async def execute_tools_parallel(tool_calls, mode="default"):
    results = []
    safe_batch = []
    for tc in tool_calls:
        perm = check_permission(tc["name"], tc["args"], mode)
        if perm["action"] == "deny":
            results.append(ToolMessage(content=perm["message"], tool_call_id=tc["id"]))
        elif perm["action"] == "confirm":
            answer = input(f"{perm['message']} 允许?(y/n): ")
            if answer.lower() != 'y':
                results.append(ToolMessage(content="用户拒绝", tool_call_id=tc["id"]))
                continue
            selected = next(t for t in tools if t.name == tc["name"])
            result = selected.invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            selected = next(t for t in tools if t.name == tc["name"])
            if tc["name"] in CONCURRENCY_SAFE:
                safe_batch.append((tc, selected))
            else:
                result = selected.invoke(tc["args"])
                results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    if safe_batch:
        parallel_results = await asyncio.gather(*[
            asyncio.to_thread(s.invoke, tc["args"]) for tc, s in safe_batch
        ])
        for (tc, _), result in zip(safe_batch, parallel_results):
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return results


if __name__ == "__main__":
    import os
    
    THIS_FILE = os.path.abspath(__file__)  # 当前文件的绝对路径
    
    async def test_tools():
        print("=== 测试1: 直接调用工具 ===")
        result = read_file.invoke({'path': THIS_FILE})
        print(f"read_file(当前文件): {result[:100]}...")
        print(f"bash('echo hello'): {bash.invoke({'command': 'echo hello'})}")
        
        print("\n=== 测试2: 权限检查 ===")
        print(f"plan模式调用write_file: {check_permission('write_file', {'path': 'test.txt'}, 'plan')}")
        result = check_permission('bash', {'command': 'rm -rf /tmp/test'})
        print(f"危险命令rm: {result}")
        
        print("\n=== 测试3: 模拟 tool_calls 并行执行 ===")
        tool_calls = [
            {"id": "call_1", "name": "read_file", "args": {"path": THIS_FILE}},
            {"id": "call_2", "name": "bash", "args": {"command": "ls -la"}},
        ]
        results = await execute_tools_parallel(tool_calls, mode="default")
        for r in results:
            print(f"结果: {r.content[:100]}...")
    
    asyncio.run(test_tools())
```

+ 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776527895358-99265fd4-4e05-4a4a-b83d-b97a7f684954.png" width="652" title="" crop="0,0,1,1" id="u3379cfcd" class="ne-image">



# RAG
### 文档加载
<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">"文档加载"在不同框架中的含义略有差异：</font>

+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">LangChain</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">中：</font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">DocumentLoader</font>**`<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">负责将各种格式的文件转换为统一的</font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">Document</font>**`<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">对象</font>
+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">RAGFlow</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">中：</font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">Parser</font>**`<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> </font><font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">负责将二进制文件解析为带位置信息的文本段落（sections）</font>
+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">LangGraph</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"> 中：文档加载是 Graph 中的一个节点（Node），可以有条件地路由到不同的加载器</font>

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">"文档加载"是离线阶段的第一步，核心任务是：</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">将任意格式的原始文件，转换为统一的、可处理的文本结构</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">。</font>

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>

##### <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">加载pdf文档</font>
```python
# ============================================================  
# 1. PDF 加载（文字层 PDF）  
# ============================================================  
from langchain_community.document_loaders import PyPDFLoader  
  
loader = PyPDFLoader("report.pdf")  
docs = loader.load()  
print(f"PDF 共 {len(docs)} 页")  
print(docs[0].page_content[:200])  
print(docs[0].metadata)  # {'source': 'report.pdf', 'page': 0}  
```

打印

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776665182744-2d274638-ae98-47b8-ac64-d8a2b8fbae9a.png" width="474" title="" crop="0,0,1,1" id="u199f2deb" class="ne-image">

惰性加载



```python
# 惰性加载（大文件推荐）  
for doc in loader.lazy_load():  
    print(doc.metadata["page"], doc.page_content[:50])  
```

+ 区别
  - <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">返回类型：</font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">Iterator[Document]</font>**`<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">（生成器）</font>
  - <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">每次迭代才读取下一页，</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">前一页可以被 GC 回收</font>**
  - <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">不能随机访问，不能 </font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">len()</font>**`<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">，只能顺序遍历</font>

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776665418937-821d9784-18a7-4208-a112-a9faa03335c2.png" width="502" title="" crop="0,0,1,1" id="ubf995127" class="ne-image">



##### 加载word文档


```python
# ============================================================  
# 2. Word 文档  
# ============================================================  
from langchain_community.document_loaders import Docx2txtLoader  
  
loader = Docx2txtLoader("contract.docx")  
docs = loader.load()  # 整个文档作为一个 Document  
```



##### 加载scv文件


```python
# ============================================================  
# 3. CSV（每行一个 Document）  
# ============================================================  
from langchain_community.document_loaders import CSVLoader  
  
loader = CSVLoader(  
    file_path="data.csv",  
    source_column="url",          # 指定哪列作为 metadata["source"]  
    csv_args={"delimiter": ","},  
)  
docs = loader.load()  
print(docs[0].page_content)  # "column1: value1\ncolumn2: value2\n..."  
  
```



##### 加载json文件


```python
# ============================================================  
# 4. JSON（用 jq 表达式提取）  
# ============================================================  
from langchain_community.document_loaders import JSONLoader  
  
loader = JSONLoader(  
    file_path="data.json",  
    jq_schema=".records[].content",  # jq 表达式  
    text_content=True,  
)  
docs = loader.load()  
```



##### 加载网页
```python
# ============================================================  
# 5. 网页  
# ============================================================  
from langchain_community.document_loaders import WebBaseLoader
import bs4
# 设置请求头，模拟真实浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 示例1：爬取有文章内容的页面（如技术博客）
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/tutorial/index.html"],  # Python 官方文档
    header_template=headers,
    bs_kwargs={"parse_only": bs4.SoupStrainer("div")},  # 只解析 <div> 标签
)

docs = loader.load()  
print(docs[0].page_content[:200])  # 打印文章内容
```

打印

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776666120441-007b9fcf-ae6d-4f73-b93b-606e588a14ad.png" width="369" title="" crop="0,0,1,1" id="ued13fdd3" class="ne-image">



##### 加载markdown文件
```python
# ============================================================  
# 6. Markdown（保留标题层级）  
# ============================================================  
from langchain_text_splitters import MarkdownHeaderTextSplitter  
  
md_text = """  
# 第一章  
## 1.1 背景  
这是背景内容。  
## 1.2 目标  
这是目标内容。  
# 第二章  
正文内容。  
"""  
  
splitter = MarkdownHeaderTextSplitter(  
    headers_to_split_on=[  
        ("#", "H1"),  
        ("##", "H2"),  
        ("###", "H3"),  
    ]  
)  
docs = splitter.split_text(md_text)  
# 每个 Document 的 metadata 包含 {"H1": "第一章", "H2": "1.1 背景"}  
  
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776666304599-8e62adc1-bdc0-438c-9ece-70377c7b2dd9.png" width="510" title="" crop="0,0,1,1" id="u96f0967d" class="ne-image">



##### 加载代码文件
```python
# ============================================================  
# 7. 代码文件  
# ============================================================  
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter  
  
loader = TextLoader("main.py")  
docs = loader.load()  
  
splitter = RecursiveCharacterTextSplitter.from_language(  
    language=Language.PYTHON,  
    chunk_size=500,  
    chunk_overlap=50,  
)  
chunks = splitter.split_documents(docs)
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776666425797-a17ede01-7a67-42fa-8984-d7b7e2c67df5.png" width="525" title="" crop="0,0,1,1" id="u40cb5922" class="ne-image">



##### 通用路由加载器
可以在调用的时候指定哪类加载器进行加载指定的文档

```python
from pathlib import Path  
from langchain_community.document_loaders import (  
    PyPDFLoader, Docx2txtLoader, CSVLoader,  
    TextLoader, BSHTMLLoader, UnstructuredMarkdownLoader,  
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,  
)  
from langchain_core.documents import Document  
  
LOADER_MAP = {  
    ".pdf":  PyPDFLoader,  
    ".docx": Docx2txtLoader,  
    ".csv":  CSVLoader,  
    ".txt":  TextLoader,  
    ".md":   UnstructuredMarkdownLoader,  
    ".html": BSHTMLLoader,  
    ".htm":  BSHTMLLoader,  
    ".xlsx": UnstructuredExcelLoader,  
    ".xls":  UnstructuredExcelLoader,  
    ".pptx": UnstructuredPowerPointLoader,  
}  
  
def load_file(file_path: str) -> list[Document]:  
    ext = Path(file_path).suffix.lower()  
    loader_cls = LOADER_MAP.get(ext)  
    if not loader_cls:  
        raise ValueError(f"不支持的文件类型: {ext}")  
      
    # TextLoader 需要指定编码  
    if loader_cls == TextLoader:  
        return loader_cls(file_path, encoding="utf-8", autodetect_encoding=True).load()  
    return loader_cls(file_path).load()
```

### 分块
##### 分块概念
<font style="color:rgb(51, 51, 51);">RAG分块是指将长文档切分成多个较小文本片段的过程，每个片段称为"chunk"</font>

##### <font style="color:rgb(51, 51, 51);">需分块原因</font>
+ **向量嵌入模型限制**

<font style="color:rgb(51, 51, 51);">嵌入模型有严格的输入长度限制。每个块需都在模型处理范围内</font>

+ **检索精度需求**
  - <font style="color:rgb(51, 51, 51);">整个文档作为单一向量会导致：</font>
    * <font style="color:rgb(51, 51, 51);">语义稀释：无关内容干扰相关内容的向量表示</font>
    * <font style="color:rgb(51, 51, 51);">无法精确定位：无法找到文档中的具体相关段落</font>
    * <font style="color:rgb(51, 51, 51);">检索噪音增加：大量无关信息降低检索质量</font>
+ **上下文窗口优化**
  - <font style="color:rgb(51, 51, 51);">LLM的上下文窗口有限，分块可以：</font>
    * <font style="color:rgb(51, 51, 51);">减少token消耗</font>
    * <font style="color:rgb(51, 51, 51);">提高处理效率</font>
    * <font style="color:rgb(51, 51, 51);">降低计算成本</font>

##### 分块策略与原理
```python
"""
分块策略对比：

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ 固定大小分块      │ 递归字符分块     │ 语义分块        │ 文档结构分块     │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ chunk_size=512  │ 按分隔符递归     │ 语义边界检测    │ 按标题/段落      │
│ overlap=50      │ 保持语义完整     │ 嵌入模型判断    │ 保持文档结构     │
│ 简单高效         │ 灵活适应        │ 语义最优        │ 结构清晰        │
│ 可能截断语义      │ 需要调参        │ 计算成本高      │ 依赖文档格式     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

推荐：递归字符分块 + 文档结构分块组合使用
"""

# 分块策略实现
class ChunkingStrategy:
    
    @staticmethod
    def recursive_chunking(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ) -> List[Document]:
        """
        递归字符分块 - 推荐
        
        原理：
        1. 按分隔符优先级分割（段落 > 句子 > 词）
        2. 递归处理直到满足大小要求
        3. 保持语义完整性
        """
        separators = separators or [
            "\n\n",    # 段落
            "\n",      # 行
            "。",      # 中文句号
            "！",      # 中文感叹号
            "？",      # 中文问号
            ".",       # 英文句号
            "!",       # 英文感叹号
            "?",       # 英文问号
            " ",       # 空格
            "",        # 字符
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={"chunk_index": i, "chunk_size": len(chunk)}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    @staticmethod
    def semantic_chunking(
        text: str,
        embedding_model: Any,
        breakpoint_threshold: float = 0.8,
    ) -> List[Document]:
        """
        语义分块 - 高级
        
        原理：
        1. 将文本分割成句子
        2. 计算相邻句子的语义相似度
        3. 在相似度低的点分割
        """
        from langchain_experimental.text_splitter import SemanticChunker
        
        splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold,
        )
        
        return splitter.create_documents([text])
```



+ 固定大小分块

原理：<font style="color:rgb(51, 51, 51);">最简单粗暴，按字符数量硬切，不管语义</font>

```python
def fixed_size_chunking(text: str, chunk_size: int = 100, overlap: int = 20):
    """
    固定大小分块
    text: 原始文本
    chunk_size: 每块多少个字符
    overlap: 相邻两块重叠多少字符（防止语义断裂）
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一块从 (start + chunk_size - overlap) 开始
        start += chunk_size - overlap

    return chunks

# 测试
text = "人工智能是计算机科学的一个分支。它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"

chunks = fixed_size_chunking(text, chunk_size=50, overlap=10)
for i, chunk in enumerate(chunks):
    print(f"[块{i+1}] {chunk}")
    print()
```

    - 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776588205742-b372d75c-ab18-4944-ad48-2ddc64695bba.png" width="505" title="" crop="0,0,1,1" id="u5f6eb2f9" class="ne-image">

可以看到这里依据50个字符大小分割了，即使这里有文字被截断了



+ 递归字符分块

<font style="color:rgb(51, 51, 51);">按优先级尝试分隔符（先按段落</font>`**<font style="color:rgb(51, 51, 51);">\n\n</font>**`<font style="color:rgb(51, 51, 51);">切，切不够小再按句子</font>`**<font style="color:rgb(51, 51, 51);">。</font>**`<font style="color:rgb(51, 51, 51);">切，再不行按空格切……），保证语义完整性</font>

<font style="color:rgb(51, 51, 51);">下载依赖 </font><font style="color:rgb(106, 115, 125);background-color:rgb(229, 229, 229);">pip install langchain-text-splitters </font>

<font style="color:rgb(106, 115, 125);background-color:rgb(229, 229, 229);"></font>

```python
# 递归字符分块
# 
from langchain_text_splitters import RecursiveCharacterTextSplitter  
  
def recursive_chunking(text: str, chunk_size: int = 100, overlap: int = 20):  
    """  
    递归字符分块  
    分隔符优先级：段落 > 换行 > 句号 > 逗号 > 空格 > 单字符  
    """  
    splitter = RecursiveCharacterTextSplitter(  
        chunk_size=chunk_size,  
        chunk_overlap=overlap,  
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],  
    )  
    chunks = splitter.split_text(text)  
    return chunks  
  
# 测试  
text = """  
人工智能是计算机科学的一个分支。  
它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。  
  
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。  
深度学习是其中最重要的技术之一。  
"""  
  
chunks = recursive_chunking(text, chunk_size=60, overlap=10)  
for i, chunk in enumerate(chunks):  
    print(f"[块{i+1}] {chunk}")  
    print()    
```

<font style="color:rgb(106, 115, 125);background-color:rgb(229, 229, 229);"></font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776588694788-0b133ec7-4964-4639-827a-d3b31bb35838.png" width="516" title="" crop="0,0,1,1" id="XNjDB" class="ne-image">

+ 语义分块

<font style="color:rgb(51, 51, 51);">不按字符数切，而是按</font>**<font style="color:rgb(51, 51, 51);">语义相似度</font>**<font style="color:rgb(51, 51, 51);">切。计算相邻句子的向量相似度，相似度突然下降的地方就是分块边界</font>

<font style="color:rgb(51, 51, 51);">下载依赖</font><font style="color:rgb(106, 115, 125);background-color:rgb(229, 229, 229);">pip install langchain-experimental sentence-transformers </font>

```python
# 语义分块策略
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_ollama import OllamaEmbeddings  

def semantic_chunking(text: str):  
    """  
    语义分块：相邻句子语义差异大的地方自动切分  
    """  
    # 使用本地 Ollama embedding 模型
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )  
      
    splitter = SemanticChunker(  
        embeddings=embeddings,  
        breakpoint_threshold_type="percentile",  # 按百分位数判断断点  
        breakpoint_threshold_amount=95,           # 差异超过95%分位才切  
    )  
      
    chunks = splitter.split_text(text)  
    return chunks  
  
# 测试  
text = """  
苹果是一种水果，富含维生素C。苹果有红色、绿色等多种颜色。苹果可以直接吃，也可以榨汁。  
  
量子计算机利用量子叠加和纠缠原理工作。它比传统计算机快得多。量子比特可以同时表示0和1。  
  
今天天气很好，阳光明媚。适合出去散步。公园里的花都开了。  
"""  
  
# 语义分块会把"苹果"、"量子计算机"、"天气"三个主题自动分成3块  
chunks = semantic_chunking(text)  
for i, chunk in enumerate(chunks):  
    print(f"[块{i+1}] {chunk}")  
    print()
```

<font style="color:rgb(51, 51, 51);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776601651322-edb25090-4ce6-48ad-9059-bd56978a235a.png" width="411" title="" crop="0,0,1,1" id="u18412fe8" class="ne-image">

这里仍然没有分块成功

原因：

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">预处理没有解决分句问题</font>

`<font style="color:rgb(51, 51, 51);">SemanticChunker</font>`<font style="color:rgb(51, 51, 51);"> 内部的分句正则是 </font>`<font style="color:rgb(51, 51, 51);">r"(?<=[.?!])\s+"</font>`<font style="color:rgb(51, 51, 51);"> —— 只认英文标点 </font>`<font style="color:rgb(51, 51, 51);">.?!</font>`<font style="color:rgb(51, 51, 51);">，不认中文的 </font>`<font style="color:rgb(51, 51, 51);">。！？</font>`<font style="color:rgb(51, 51, 51);">。</font>

<font style="color:rgb(51, 51, 51);">你的预处理把 </font>`<font style="color:rgb(51, 51, 51);">。</font>`<font style="color:rgb(51, 51, 51);"> 变成了 </font>`<font style="color:rgb(51, 51, 51);">。\n</font>`<font style="color:rgb(51, 51, 51);">，但分句器看的是 </font>`<font style="color:rgb(51, 51, 51);">。</font>`<font style="color:rgb(51, 51, 51);"> 后面有没有英文句号，所以整段文本仍然被当成1个句子，没有相邻句子可以比较距离，自然无法分块。</font>



因此作出如下修改

# 关键修复：把中文标点替换成英文句号+空格

    # SemanticChunker 的分句器只认英文标点 .?!  

    text = re.sub(r'[。！？]', '. ', text)  

    text = re.sub(r'\s+', ' ', text).strip()



```python
import re
import numpy as np
from langchain_ollama import OllamaEmbeddings


def split_chinese_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_chunking(text: str, similarity_threshold: float = 0.49) -> list[str]:
    """
    自定义中文语义分块
    1. 按中文标点分句
    2. 计算相邻句子的余弦相似度
    3. 相似度低于阈值处切分
    """
    sentences = split_chinese_sentences(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    sentence_embeddings = np.array(embeddings.embed_documents(sentences))

    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
        similarities.append(sim)

    print("句子间相似度:")
    for i, sim in enumerate(similarities):
        print(f"  句子{i+1} <-> 句子{i+2}: {sim:.4f}")

    chunks = []
    current_chunk = sentences[0]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i + 1]
        else:
            current_chunk += sentences[i + 1]
    chunks.append(current_chunk)

    return chunks

text = """  
苹果是一种水果，富含维生素C。苹果有红色、绿色等多种颜色。苹果可以直接吃，也可以榨汁。量子计算机利用量子叠加和纠缠原理工作。它比传统计算机快得多。量子比特可以同时表示0和1。今天天气很好，阳光明媚。适合出去散步。公园里的花都开了。天气相当好，可以出去玩。 人工智能相当强大。苹果很好吃。  
"""  
  
chunks = semantic_chunking(text)  
for i, chunk in enumerate(chunks):  
    print(f"[块{i+1}] {chunk}")  
    print()

```

输出



+ 文档结构分块

<font style="color:rgb(51, 51, 51);">利用文档本身的结构（标题、段落、列表）来切分，而不是靠字符数或语义。</font>

<font style="color:rgb(51, 51, 51);"></font>

```python
import re  
from typing import List, Dict  
  
def structure_chunking(markdown_text: str) -> List[Dict]:  
    """  
    文档结构分块：按 Markdown 标题层级切分  
    适合有明确结构的文档（技术文档、报告、教材等）  
    """  
    chunks = []  
    current_chunk = {"title": "前言", "level": 0, "content": ""}  
      
    for line in markdown_text.split("\n"):  
        # 检测标题行（# 一级标题，## 二级标题，### 三级标题）  
        header_match = re.match(r'^(#{1,3})\s+(.+)', line)  
          
        if header_match:  
            # 遇到新标题，保存当前块  
            if current_chunk["content"].strip():  
                chunks.append(current_chunk.copy())  
              
            # 开始新块  
            level = len(header_match.group(1))  # 几个#就是几级  
            title = header_match.group(2)  
            current_chunk = {  
                "title": title,  
                "level": level,  
                "content": ""  
            }  
        else:  
            # 普通内容行，追加到当前块  
            current_chunk["content"] += line + "\n"  
      
    # 别忘了最后一块  
    if current_chunk["content"].strip():  
        chunks.append(current_chunk)  
      
    return chunks  
  
  
# 测试  
markdown_doc = """  
# 人工智能简介  
  
人工智能（AI）是计算机科学的重要分支。  
  
## 机器学习  
  
机器学习是AI的核心技术。  
通过大量数据训练模型，让机器自动学习规律。  
  
### 深度学习  
  
深度学习使用多层神经网络。  
在图像识别、语音识别领域表现优异。  
  
## 自然语言处理  
  
NLP让计算机理解人类语言。  
ChatGPT就是NLP技术的典型应用。  
"""  
  
chunks = structure_chunking(markdown_doc)  
for chunk in chunks:  
    print(f"[{'#'*chunk['level']} {chunk['title']}]")  
    print(chunk['content'].strip())  
    print()
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776602624626-0537cb00-5537-4cf5-8c95-4d3f63e1ec89.png" width="434" title="" crop="0,0,1,1" id="u8ab1e80e" class="ne-image">



### embedding**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">向量化</font>**选择
常见的embedding

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776603129387-f802d9ce-7d68-4c15-b0bb-47373e0a0f76.png" width="556" title="" crop="0,0,1,1" id="ub3f5d3ea" class="ne-image">

##### 概念
<font style="color:rgb(51, 51, 51);">Embedding（嵌入）是将</font>**<font style="color:rgb(51, 51, 51);">离散的符号</font>**<font style="color:rgb(51, 51, 51);">（文字、词、句子）映射到</font>**<font style="color:rgb(51, 51, 51);">连续的高维向量空间</font>**<font style="color:rgb(51, 51, 51);">的过程，使得语义相近的内容在向量空间中距离也相近。</font>

```plain
"苹果"   → [0.12, -0.34, 0.89, ...]   ← 768维向量  
"香蕉"   → [0.15, -0.31, 0.85, ...]   ← 语义相近，向量相近  
"汽车"   → [-0.72, 0.56, -0.23, ...]  ← 语义不同，向量远离
```

<font style="color:rgb(51, 51, 51);">在 RAG 中的位置</font>

```plain
文本块 ──► [Embedding模型] ──► 向量 ──► 向量数据库  
用户问题 ──► [Embedding模型] ──► 向量 ──► 相似度搜索
```



**<font style="color:rgb(51, 51, 51);">维度（Dimension）</font>**

+ <font style="color:rgb(51, 51, 51);">维度越高，表达能力越强，但存储和计算成本越高</font>
+ <font style="color:rgb(51, 51, 51);">512维 vs 3072维：存储差6倍，检索速度差约3倍</font>
+ <font style="color:rgb(51, 51, 51);"> 支持的向量维度：512 / 768 / 1024 / 1536 / 2048 / 4096 / 6144 / 8192 / 10240</font>

**<font style="color:rgb(51, 51, 51);">最大 Token 数（Context Window）</font>**

+ <font style="color:rgb(51, 51, 51);">超出 token 限制的文本会被截断，导致语义丢失</font>
+ <font style="color:rgb(51, 51, 51);">长文档场景必须选 8k+ token 的模型（如 bge-m3、Qwen3-Embedding）</font>

**<font style="color:rgb(51, 51, 51);">语言支持</font>**

+ <font style="color:rgb(51, 51, 51);">纯英文模型（bge-small-en）用于中文会严重退化</font>
+ <font style="color:rgb(51, 51, 51);">中文场景推荐：bge-m3、Qwen3-Embedding、text-embedding-3-large</font>

**<font style="color:rgb(51, 51, 51);">对称 vs 非对称检索</font>**

+ **<font style="color:rgb(51, 51, 51);">对称</font>**<font style="color:rgb(51, 51, 51);">：问题和文档语义相近（FAQ 匹配）</font>
+ **<font style="color:rgb(51, 51, 51);">非对称</font>**<font style="color:rgb(51, 51, 51);">：短问题检索长文档（RAG 的典型场景）</font>
+ <font style="color:rgb(51, 51, 51);">E5/BGE 系列通过 </font>`**<font style="color:rgb(51, 51, 51);">query:</font>**`<font style="color:rgb(51, 51, 51);"> / </font>`**<font style="color:rgb(51, 51, 51);">passage:</font>**`<font style="color:rgb(51, 51, 51);"> 前缀区分两种模式</font>



### **<font style="color:rgb(51, 51, 51);">索引存储</font>**
<font style="color:rgb(51, 51, 51);">为什么需要"索引存储"</font>

<font style="color:rgb(51, 51, 51);">RAG 的索引存储做的事情：</font>

1. **<font style="color:rgb(51, 51, 51);">建索引（离线）</font>**<font style="color:rgb(51, 51, 51);">：把所有文档切块、向量化，存入一个支持快速检索的数据库</font>
2. **<font style="color:rgb(51, 51, 51);">查索引（在线）</font>**<font style="color:rgb(51, 51, 51);">：用户提问时，把问题也向量化，在数据库里快速找到最相似的块</font>

<font style="color:rgb(51, 51, 51);">RAG 的核心操作是：</font>**<font style="color:rgb(51, 51, 51);">给定一个查询向量，在数百万个文档向量中找出最相似的 K 个</font>**<font style="color:rgb(51, 51, 51);">。</font>

```plain
暴力做法：  
  查询向量 q 与每个文档向量 d_i 计算余弦相似度  
  时间复杂度：O(N × D)  
  N=100万文档，D=1536维 → 每次查询需要 15亿次浮点运算 → 太慢  
  
索引的作用：  
  用空间换时间，把 O(N) 降到 O(log N) 甚至 O(1)  
  代价：精度可能略有损失（近似最近邻 ANN）
```

<font style="color:rgb(51, 51, 51);">LLM（大语言模型）有两个硬伤：</font>

1. **<font style="color:rgb(51, 51, 51);">知识截止日期</font>**<font style="color:rgb(51, 51, 51);">：训练数据有时间限制，不知道最新信息</font>
2. **<font style="color:rgb(51, 51, 51);">上下文窗口有限</font>**<font style="color:rgb(51, 51, 51);">：一次最多处理几万 token，塞不下整本书</font>

<font style="color:rgb(51, 51, 51);">RAG 的解决思路：</font>**<font style="color:rgb(51, 51, 51);">不把所有知识塞给 LLM，而是先检索相关片段，再让 LLM 回答</font>**<font style="color:rgb(51, 51, 51);">。</font>

```plain
没有 RAG：  
用户问题 ──────────────────────────────▶ LLM ──▶ 答案（可能胡说）  
  
有 RAG：  
用户问题 ──▶ [检索引擎] ──▶ 相关片段 ──▶ LLM ──▶ 答案（有依据）  
                ↑  
           [索引存储] ← 提前把文档处理好存进来
```

**<font style="color:rgb(51, 51, 51);">索引存储就是那个"检索引擎的数据库"</font>**<font style="color:rgb(51, 51, 51);">，它决定了检索的速度和质量。</font>

##### <font style="color:rgb(51, 51, 51);">向量索引常见的算法</font>
<font style="color:rgb(51, 51, 51);">向量库不是简单地把所有向量存起来，然后一个个比较（那样太慢）。它会建立</font>**<font style="color:rgb(51, 51, 51);">近似最近邻索引（ANN Index）</font>**<font style="color:rgb(51, 51, 51);">，让搜索从 O(n) 变成 O(log n)。</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776667742057-46d18c27-66d9-44ce-8a39-46fedbb1bea9.png" width="553" title="" crop="0,0,1,1" id="ub73b1139" class="ne-image">

###### Flat
    - <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">原理</font>

```plain
存储：把所有向量直接存在一个大矩阵里  
查询：逐一计算查询向量与所有向量的距离，取最小的 K 个  
  
[d1] [d2] [d3] [d4] ... [dN]  
  ↑    ↑    ↑    ↑         ↑  
  全部计算距离，排序取 Top-K
```

    - <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">特点</font>

| **<font style="color:rgb(51, 51, 51);">维度</font>** | **<font style="color:rgb(51, 51, 51);">说明</font>** |
| --- | --- |
| <font style="color:rgb(51, 51, 51);">精度</font> | <font style="color:rgb(51, 51, 51);">100%（精确最近邻，不是近似）</font> |
| <font style="color:rgb(51, 51, 51);">查询速度</font> | <font style="color:rgb(51, 51, 51);">O(N × D)，最慢</font> |
| <font style="color:rgb(51, 51, 51);">构建速度</font> | <font style="color:rgb(51, 51, 51);">O(N)，最快（直接存）</font> |
| <font style="color:rgb(51, 51, 51);">内存</font> | <font style="color:rgb(51, 51, 51);">O(N × D)，全量存储</font> |
| <font style="color:rgb(51, 51, 51);">适合场景</font> | <font style="color:rgb(51, 51, 51);">数据量 < 10万，精度要求极高</font> |




实践

先下载依赖 pip install faiss-cpu

```python
import faiss 
import numpy as np  
  
d = 1536  # 向量维度  
N = 10000  # 文档数量  
  
# 构建索引  
index = faiss.IndexFlatL2(d)  # L2 距离（欧氏距离）  
# 或者：faiss.IndexFlatIP(d)  # 内积（余弦相似度需先归一化）  
  
# 添加向量  
vectors = np.random.rand(N, d).astype('float32')  
index.add(vectors)  
  
# 查询  
query = np.random.rand(1, d).astype('float32')  
distances, indices = index.search(query, k=5)  # 返回最近的5个  
print(f"最近邻索引: {indices[0]}")  
print(f"距离: {distances[0]}")
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776690819111-85e51bc8-f2e3-4ade-981f-3afe2410bba4.png" width="504" title="" crop="0,0,1,1" id="u6f396550" class="ne-image">

###### IVF（<font style="color:rgb(51, 51, 51);">Inverted File Index，倒排文件索引</font>）
**<font style="color:rgb(51, 51, 51);">核心思想：先聚类，再在小范围内搜索</font>**

```plain
构建阶段：  
  1. 用 K-Means 把所有向量聚成 nlist 个簇（如 1000 个）  
  2. 每个向量归属到最近的簇中心  
  
  簇1: [d3, d7, d15, ...]  
  簇2: [d1, d9, d22, ...]  
  ...  
  簇1000: [d5, d11, d88, ...]  
  
查询阶段：  
  1. 找到查询向量最近的 nprobe 个簇（如 10 个）  
  2. 只在这 nprobe 个簇内做精确搜索  
  3. 返回 Top-K  
  
时间复杂度：O(nprobe × N/nlist × D)  
  nlist=1000, nprobe=10 → 只搜索 1% 的数据
```

+ <font style="color:rgb(51, 51, 51);">图示</font>

```plain
                      查询向量 q  
                         │  
                         ▼  
              找最近的 nprobe 个簇中心  
                    /    |    \  
                   ▼     ▼     ▼  
                 簇3   簇7   簇15  
                  │     │     │  
                  └─────┴─────┘  
                         │  
                    在这些簇内  
                    精确搜索  
                         │  
                         ▼  
                      Top-K 结果
```

+ <font style="color:rgb(51, 51, 51);">关键参数</font>

| **<font style="color:rgb(51, 51, 51);">参数</font>** | **<font style="color:rgb(51, 51, 51);">含义</font>** | **<font style="color:rgb(51, 51, 51);">调优建议</font>** |
| --- | --- | --- |
| `**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">nlist</font>**` | <font style="color:rgb(51, 51, 51);">簇的数量</font> | <font style="color:rgb(51, 51, 51);">通常</font><font style="color:rgb(51, 51, 51);"> </font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">4*sqrt(N)</font>**`<br/><font style="color:rgb(51, 51, 51);"> </font><font style="color:rgb(51, 51, 51);">~</font><font style="color:rgb(51, 51, 51);"> </font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">16*sqrt(N)</font>**` |
| `**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">nprobe</font>**` | <font style="color:rgb(51, 51, 51);">查询时搜索的簇数</font> | <font style="color:rgb(51, 51, 51);">越大越准确，越慢；通常</font><font style="color:rgb(51, 51, 51);"> </font>`**<font style="color:rgb(51, 51, 51);background-color:rgb(229, 229, 229);">nlist/10</font>**` |


+ **<font style="color:rgb(0, 0, 0);">原理</font>**<font style="color:rgb(13, 13, 13);">：类似现实中按行政区找人。先用 K-Means 算法把所有向量聚类成 </font>`**<font style="color:rgb(13, 13, 13);">nlist</font>**`<font style="color:rgb(13, 13, 13);"> 个簇（类似划分行政区），每个簇有一个中心点。查询时，先算 Query 离哪些中心点最近，只在这些中心点所在的簇里暴力搜索。</font>
+ **<font style="color:rgb(0, 0, 0);">核心参数</font>**<font style="color:rgb(13, 13, 13);">：</font>
  - `**<font style="color:rgb(13, 13, 13);">nlist</font>**`<font style="color:rgb(13, 13, 13);">：划分多少个区域（建库时设定）。</font>
  - `**<font style="color:rgb(13, 13, 13);">nprobe</font>**`<font style="color:rgb(13, 13, 13);">：查询时探查几个区域（检索时设定）。</font>`**<font style="color:rgb(13, 13, 13);">nprobe</font>**`<font style="color:rgb(13, 13, 13);"> 越大，越慢但越准。</font>
+ **<font style="color:rgb(0, 0, 0);">优缺点</font>**<font style="color:rgb(13, 13, 13);">：速度快，内存可控；但处于两个区域交界处的向量容易被漏召回。</font>

实践

```plain
# IVF 聚类索引
import numpy as np
import faiss
import time

# ------- 1. 准备数据 -------
d = 64
nb = 100000
nq = 1
k = 5
np.random.seed(1234)
db_vectors = np.random.random((nb, d)).astype('float32')
query_vectors = np.random.random((nq, d)).astype('float32')

# ------- 2. 先跑一次 Flat，拿到标准答案 indices_flat -------
index_flat = faiss.IndexFlatIP(d)
index_flat.add(db_vectors)
distances_flat, indices_flat = index_flat.search(query_vectors, k) # 这里生成了 indices_flat！

# ------- 3. 开始跑 IVF -------
print("="*50 + " IVF 聚类索引 " + "="*50)
nlist = 100 
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# IVF 必须先 train
index_ivf.train(db_vectors)
index_ivf.add(db_vectors)

# nprobe 控制查几个桶
index_ivf.nprobe = 10 

start_time = time.time()
distances_ivf, indices_ivf = index_ivf.search(query_vectors, k)
ivf_time = (time.time() - start_time) * 1000

# 现在这行就不会报错了！
recall_ivf = len(set(indices_flat[0]) & set(indices_ivf[0])) / k

print(f"耗时: {ivf_time:.2f} 毫秒")
print(f"Flat标准答案 ID: {indices_flat[0]}")
print(f"IVF查出来的  ID: {indices_ivf[0]}")
print(f"⚠️ 召回率: {recall_ivf * 100}%")
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776823722811-7554e0a0-c2e7-4eeb-a8c3-ad06e93189c7.png" width="398" title="" crop="0,0,1,1" id="u44a5aea9" class="ne-image">

开源看到召回率只有40%

提高查询桶的数量到30% index_ivf.nprobe = 30

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776823834160-1586ef99-d4fe-4727-b46c-822b039d8c47.png" width="403" title="" crop="0,0,1,1" id="uf33dfcdc" class="ne-image">

为啥呢

<font style="color:rgb(13, 13, 13);">在完全均匀分布的高维空间里：</font>**<font style="color:rgb(0, 0, 0);">你最相似的 5 个向量，极有可能散布在完全不同的几个切瓜区域里。</font>**

<font style="color:rgb(13, 13, 13);">标准答案：</font>`<font style="color:rgb(13, 13, 13);">[70195, 56312, 21414, 9304, 14750]</font>`

+ <font style="color:rgb(13, 13, 13);">假设 70195 和 9304 刚好离查询点比较近，它们所在的区域被你探查到了（nprobe=10 就能找到）。</font>
+ <font style="color:rgb(13, 13, 13);">但 56312、21414、14750 虽然和查询点距离近，</font>**<font style="color:rgb(0, 0, 0);">它们却处在另外几个离查询点中心较远的西瓜块里</font>**<font style="color:rgb(13, 13, 13);">。</font>
+ <font style="color:rgb(13, 13, 13);">即便你把 </font>`**<font style="color:rgb(13, 13, 13);">nprobe</font>**`<font style="color:rgb(13, 13, 13);"> 扩大到了 30（查了 30 个片区），那 3 个人所在的片区可能排在第 31、35、40 名，</font>**<font style="color:rgb(0, 0, 0);">依然在你的探查盲区里！</font>**

<font style="color:rgb(13, 13, 13);">这就是 IVF 在随机高维数据上的软肋：</font>**<font style="color:rgb(0, 0, 0);">聚类中心（片区大门）离你近，不代表你最想找的人就在这几个片区里。</font>**

修改成70 index_ivf.nprobe = 70

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776823924186-086d0786-0d43-46a4-93f5-610897cb54d6.png" width="402" title="" crop="0,0,1,1" id="u641a66c1" class="ne-image">

可以看到召回率提高了



+ HNSW



##### 全文索引算法
为啥有向量索引还需全文索引呢



##### <font style="color:rgb(51, 51, 51);">常见索引数据库</font>
<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>

### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">查询向量化</font>**


### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">相似度搜索</font>**


### <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">重排序（Reranking）</font>


### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">上下文构建</font>**


### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">提示构建</font>**


### <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">LLM生成</font>
**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>**

### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">结果后处理</font>**







