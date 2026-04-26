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

##### <font style="color:rgb(0, 0, 0);">ANN (近似最近邻)</font>
+ **<font style="color:rgb(0, 0, 0);">定义</font>**<font style="color:rgb(13, 13, 13);">：不要求找到绝对最近的邻居，只要求在极短时间内找到大概率很近的邻居。</font>
+ **<font style="color:rgb(0, 0, 0);">类比</font>**<font style="color:rgb(13, 13, 13);">：在人群中找最像你朋友的人，KNN是给所有人量一遍三围（极准极慢），ANN是只看大概身高体型（略准极快）。</font>
+ **<font style="color:rgb(0, 0, 0);">一句话记忆</font>**<font style="color:rgb(13, 13, 13);">：用微小精度换取指数级速度。</font>

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



###### HNSW
<font style="color:rgb(13, 13, 13);">HNSW 的全称是 </font>**<font style="color:rgb(0, 0, 0);">Hierarchical</font>**<font style="color:rgb(13, 13, 13);"> Navigable Small World。核心就在于 </font>**<font style="color:rgb(0, 0, 0);">Hierarchical（分层）</font>**<font style="color:rgb(13, 13, 13);">。</font>

<font style="color:rgb(13, 13, 13);">想象一栋有三层楼的大楼，每层都铺了一张网：</font>

+ **<font style="color:rgb(0, 0, 0);">顶层（Layer 2）</font>**<font style="color:rgb(13, 13, 13);">：网眼最大，节点最少，线最长。只有几个核心枢纽（比如北京、上海、广州）。</font>
+ **<font style="color:rgb(0, 0, 0);">中层（Layer 1）</font>**<font style="color:rgb(13, 13, 13);">：网眼中等，节点增多，线变短。有区域中心城市（比如省会城市）。</font>
+ **<font style="color:rgb(0, 0, 0);">底层（Layer 0）</font>**<font style="color:rgb(13, 13, 13);">：网眼最密，</font>**<font style="color:rgb(0, 0, 0);">包含所有的 10 万个向量节点</font>**<font style="color:rgb(13, 13, 13);">，线最短。是具体的街道和楼宇。</font>

**<font style="color:rgb(0, 0, 0);">一个节点出现在高层，它必然在所有的低层也存在！</font>**<font style="color:rgb(13, 13, 13);">（广州在 Layer 2，那它也在 Layer 1 和 Layer 0）。</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776909451464-90ec62c4-2d3c-48ea-a254-e642bc1cb6dd.png" width="997" title="" crop="0,0,1,1" id="u1b51d59b" class="ne-image">

(_<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">实线是同层的朋友关系，虚线是下楼梯的通道</font>_)

搜索过程

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776909665459-7360307a-21bb-4cbc-a4f1-762a1e075e4a.png" width="1135" title="" crop="0,0,1,1" id="u4cba4d9b" class="ne-image">



1. **<font style="color:rgb(0, 0, 0);">Layer 2</font>**<font style="color:rgb(13, 13, 13);">：从入口 </font>`**<font style="color:rgb(13, 13, 13);">C</font>**`<font style="color:rgb(13, 13, 13);"> 开始。发现 </font>`**<font style="color:rgb(13, 13, 13);">D</font>**`<font style="color:rgb(13, 13, 13);"> 离 </font>`**<font style="color:rgb(13, 13, 13);">Q</font>**`<font style="color:rgb(13, 13, 13);"> 更近，跳到 </font>`**<font style="color:rgb(13, 13, 13);">D</font>**`<font style="color:rgb(13, 13, 13);">。在 Layer 2 找不到比 </font>`**<font style="color:rgb(13, 13, 13);">D</font>**`<font style="color:rgb(13, 13, 13);"> 更近的了，</font>**<font style="color:rgb(0, 0, 0);">下楼梯</font>**<font style="color:rgb(13, 13, 13);">。</font>
2. **<font style="color:rgb(0, 0, 0);">Layer 1</font>**<font style="color:rgb(13, 13, 13);">：以 </font>`**<font style="color:rgb(13, 13, 13);">D</font>**`<font style="color:rgb(13, 13, 13);"> 为起点。发现 </font>`**<font style="color:rgb(13, 13, 13);">A</font>**`<font style="color:rgb(13, 13, 13);"> 离 </font>`**<font style="color:rgb(13, 13, 13);">Q</font>**`<font style="color:rgb(13, 13, 13);"> 更近，跳到 </font>`**<font style="color:rgb(13, 13, 13);">A</font>**`<font style="color:rgb(13, 13, 13);">。在 Layer 1 找不到比 </font>`**<font style="color:rgb(13, 13, 13);">A</font>**`<font style="color:rgb(13, 13, 13);"> 更近的了，</font>**<font style="color:rgb(0, 0, 0);">下楼梯</font>**<font style="color:rgb(13, 13, 13);">。</font>
3. **<font style="color:rgb(0, 0, 0);">Layer 0</font>**<font style="color:rgb(13, 13, 13);">：以 </font>`**<font style="color:rgb(13, 13, 13);">A</font>**`<font style="color:rgb(13, 13, 13);"> 为起点。这是最底层，不仅找最近，还要把 </font>`**<font style="color:rgb(13, 13, 13);">A</font>**`<font style="color:rgb(13, 13, 13);"> 的朋友、朋友的朋友都扫一遍（范围由 </font>`**<font style="color:rgb(13, 13, 13);">efSearch</font>**`<font style="color:rgb(13, 13, 13);"> 决定），最终锁定最近的人。</font>

<font style="color:rgb(13, 13, 13);"></font>

<font style="color:rgb(13, 13, 13);">实践</font>

<font style="color:rgb(13, 13, 13);"></font>

```plain
# ==========================================
# 1. 准备统一的模拟数据
# ==========================================
d = 64        # 向量维度
nb = 100000   # 知识库数据量 (10万条)
nq = 1        # 提问数量
k = 5         # 想要查找的 Top-K 结果数

np.random.seed(1234)
db_vectors = np.random.random((nb, d)).astype('float32')
query_vectors = np.random.random((nq, d)).astype('float32')

print("🚀 数据准备完毕！知识库: 10万条64维向量\n")


# ==========================================
# 2. Flat 暴力检索 (必须先跑这个，拿到标准答案！)
# ==========================================
print("="*50 + " 1. Flat 暴力检索 " + "="*50)
index_flat = faiss.IndexFlatIP(d) 
index_flat.add(db_vectors)

start_time = time.time()
distances_flat, indices_flat = index_flat.search(query_vectors, k) # 这里生成了标准答案 indices_flat
flat_time = (time.time() - start_time) * 1000

print(f"耗时: {flat_time:.2f} 毫秒")
print(f"最相似的 Top-{k} ID: {indices_flat[0]}")
print("👉 这便是【标准答案】，后面都要跟它比！\n")


# ==========================================
# 3. HNSW 图索引
# ==========================================
print("="*50 + " 2. HNSW 图索引 " + "="*50)
M = 32
index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(db_vectors)
index_hnsw.hnsw.efSearch = 64

start_time = time.time()
distances_hnsw, indices_hnsw = index_hnsw.search(query_vectors, k)
hnsw_time = (time.time() - start_time) * 1000

recall_hnsw = len(set(indices_flat[0]) & set(indices_hnsw[0])) / k

print(f"耗时: {hnsw_time:.2f} 毫秒")
print(f"最相似的 Top-{k} ID: {indices_hnsw[0]}")
print(f"✅ 召回率: {recall_hnsw * 100}%\n")

```

<font style="color:rgb(13, 13, 13);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1776909810081-ca937c5d-4397-4278-bef1-a941cc9eeb36.png" width="409" title="" crop="0,0,1,1" id="u0fe48410" class="ne-image">



###### LSH
**<font style="color:rgb(51, 51, 51);">相似的向量以高概率映射到同一个桶（bucket），不相似的以高概率映射到不同桶</font>**<font style="color:rgb(51, 51, 51);">，从而实现近似最近邻搜索（ANN）</font>

**<font style="color:rgb(0, 0, 0);">动制造冲突——让高维空间中距离近的向量，以极大概率映射到同一个哈希桶中。</font>**<font style="color:rgb(13, 13, 13);"> 从而将暴力计算降维为哈希表的</font>_<font style="color:rgb(13, 13, 13);">O</font>_<font style="color:rgb(13, 13, 13);">(1)查询。</font>

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">实践</font>

```plain
import numpy as np  
from collections import defaultdict  
  
class LSH:  
    """  
    基于随机超平面投影的 LSH（适用于余弦相似度）  
    """  
    def __init__(self, n_planes: int, n_tables: int, dim: int):  
        """  
        n_planes: 每张哈希表的超平面数量（越多越精确，但越慢）  
        n_tables: 哈希表数量（越多召回率越高）  
        dim:      向量维度  
        """  
        self.n_planes = n_planes  
        self.n_tables = n_tables  
        self.dim = dim  
        # 每张表随机生成 n_planes 个超平面法向量  
        self.planes = [  
            np.random.randn(n_planes, dim) for _ in range(n_tables)  
        ]  
        self.tables = [defaultdict(list) for _ in range(n_tables)]  
  
    def _hash(self, vec: np.ndarray, table_idx: int) -> tuple:  
        """将向量投影到超平面，取符号作为哈希码"""  
        projections = self.planes[table_idx] @ vec  # shape: (n_planes,)  
        return tuple((projections > 0).astype(int))  
  
    def index(self, vec: np.ndarray, item_id):  
        """将向量加入索引"""  
        for i in range(self.n_tables):  
            key = self._hash(vec, i)  
            self.tables[i][key].append(item_id)  
  
    def query(self, vec: np.ndarray) -> set:  
        """返回候选集合（与查询向量在同一桶的所有 item）"""  
        candidates = set()  
        for i in range(self.n_tables):  
            key = self._hash(vec, i)  
            candidates.update(self.tables[i][key])  
        return candidates  
  
  
def cosine_similarity(a, b):  
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)  
  
  
# ── 测试 ──────────────────────────────────────────────────────────────────────  
np.random.seed(42)  
DIM = 64  
N = 1000  
  
# 生成 1000 个随机向量作为数据库  
db_vecs = np.random.randn(N, DIM)  
db_vecs /= np.linalg.norm(db_vecs, axis=1, keepdims=True)  # 归一化  
  
# 建立 LSH 索引  
lsh = LSH(n_planes=10, n_tables=5, dim=DIM)  
for idx, vec in enumerate(db_vecs):  
    lsh.index(vec, idx)  
  
# 查询向量（与 db_vecs[0] 非常相似）  
query = db_vecs[0] + np.random.randn(DIM) * 0.05  
query /= np.linalg.norm(query)  
  
# LSH 候选集  
candidates = lsh.query(query)  
print(f"候选集大小: {len(candidates)} / {N}")  
  
# 在候选集中精确计算余弦相似度，找最近邻  
best_id, best_sim = -1, -1  
for cid in candidates:  
    sim = cosine_similarity(query, db_vecs[cid])  
    if sim > best_sim:  
        best_sim, best_id = sim, cid  
  
# 暴力搜索（ground truth）  
true_sims = [cosine_similarity(query, db_vecs[i]) for i in range(N)]  
true_best = int(np.argmax(true_sims))  
  
print(f"LSH 找到: id={best_id}, 相似度={best_sim:.4f}")  
print(f"暴力搜索: id={true_best}, 相似度={true_sims[true_best]:.4f}")  
print(f"结果一致: {best_id == true_best}")
```

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777037376564-b018359c-d655-4d0b-a62c-2410b3559142.png" width="416" title="" crop="0,0,1,1" id="udb0c4a73" class="ne-image">

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);"></font>

##### 全文索引算法
为啥有向量索引还需全文索引呢

**第一，解决朴素词频统计的失真问题。** 最简单的文本相关性想法是：一篇文档里某个词出现越多，就越相关。但这会导致一个明显的偏差——一篇 10000 词的长文档，哪怕某关键词只出现了 5 次，也可能得分远高于一篇 200 词的精准短文档（关键词出现 3 次）。BM25 引入了文档长度归一化，专门解决这个问题。

**第二，解决词频的边际效应递减问题。** 朴素 TF-IDF 中词频（TF）与相关性呈线性关系，但现实中并非如此——一个词出现 10 次和出现 11 次的相关性差距，远小于出现 1 次和出现 2 次的差距。BM25 用一个非线性的饱和函数来模拟这种"边际递减"，让词频对分数的贡献趋于饱和上限。

**第三，提供一个在向量检索出现之前最优雅的稀疏检索解决方案。** 在深度学习嵌入向量流行之前，BM25 几乎是工业界信息检索的标准答案，Elasticsearch 和 Solr 的默认相关性算法就是 BM25。即便在今天，BM25 对于**精确关键词匹配**的场景依然优于很多向量方法，因此在混合检索（Hybrid Search）中占据不可替代的位置。

BM25全文索引和向量索引算法

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777041305526-53bf85d4-0291-4c63-ae30-dd6907c39e4e.png" width="980" title="" crop="0,0,1,1" id="ue2aa39c5" class="ne-image">

BM25（Best Matching 25）是概率检索框架下的排序函数，其核心公式为：

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777041692312-24cd34a1-7f16-4a87-a1a9-fbae13e7fefe.png" width="433" title="" crop="0,0,1,1" id="u9dd8539c" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777041738601-0a52f619-e943-4643-befc-0c1950b4333c.png" width="342" title="" crop="0,0,1,1" id="u20ff3351" class="ne-image">



BM25 是**词袋模型（Bag of Words）**，它完全无视词序。"猫追狗"和"狗追猫"对 BM25 来说是一样的文档，因为两者包含的词集合相同，只是顺序不同。这使得它无法理解语义关系："汽车"和"轿车"对 BM25 是两个完全不同的词，它不知道它们是近义词。这个取舍换来的是极高的计算效率和可解释性——你随时可以说清楚为什么某个文档得了高分。

另一个取舍是：BM25 无需训练数据，索引建立后可以立即查询，这与需要训练嵌入模型的语义检索形成鲜明对比。这使得 BM25 在冷启动场景下极具价值。

+ 实践



```plain
import math  
from collections import defaultdict  
  
# ============================================================  
# Step 1: 文本预处理  
# ============================================================  
  
def tokenize(text: str) -> list[str]:  
    """  
    将文本分词并小写化。  
    ① 这行做了什么：把字符串按空格切分，并统一转小写  
    ② 为什么这样写：BM25 是词袋模型，大小写不影响语义，统一小写避免重复计数  
    ③ 如果改成不 lower()：'Python' 和 'python' 会被视为不同词，召回率下降  
    """  
    return text.lower().split()  
  
# ============================================================  
# Step 2: 构建倒排索引  
# ============================================================  
  
class BM25:  
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):  
        """  
        corpus: 文档列表，每个元素是一段文本  
        k1: 词频饱和参数，控制 TF 的影响上限，推荐 1.2~2.0  
        b:  长度归一化参数，0=不归一化，1=完全归一化，推荐 0.75  
        """  
        self.k1 = k1  
        self.b = b  
        self.corpus = corpus  
  
        # 对每篇文档分词，得到词列表  
        # ① 这行做了什么：将所有文档转为词列表的列表  
        # ② 为什么用列表推导：简洁且高效，避免显式 for 循环  
        self.tokenized_corpus = [tokenize(doc) for doc in corpus]  
  
        # 文档数量 N  
        self.N = len(corpus)  
  
        # 每篇文档的长度（词数）  
        # ① 这行做了什么：计算每篇文档的词数  
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]  
  
        # 平均文档长度 avgdl  
        # ③ 如果 corpus 为空：会触发 ZeroDivisionError，生产中需加保护  
        self.avgdl = sum(self.doc_lengths) / self.N  
  
        # 构建倒排索引：term → {doc_id: tf}  
        # defaultdict(dict) 避免 KeyError，自动初始化  
        self.inverted_index = defaultdict(dict)  
        self._build_index()  
  
        # 计算每个词的 IDF  
        self.idf = {}  
        self._compute_idf()  
  
    def _build_index(self):  
        """  
        构建倒排索引。  
        ① 这行做了什么：遍历每篇文档的每个词，统计词频  
        ② 为什么用 defaultdict(int)：避免判断 key 是否存在，直接 += 1  
        """  
        for doc_id, tokens in enumerate(self.tokenized_corpus):  
            # 统计当前文档中每个词的词频  
            term_freq = defaultdict(int)  
            for token in tokens:  
                term_freq[token] += 1  # 词频 +1  
  
            # 将词频写入倒排索引  
            for term, tf in term_freq.items():  
                self.inverted_index[term][doc_id] = tf  
  
    def _compute_idf(self):  
        """  
        计算每个词的 IDF（BM25 改进版公式）。  
        公式：IDF = log((N - n + 0.5) / (n + 0.5) + 1)  
        ① +0.5 平滑：避免 n=0 时除零，也避免 n=N 时 IDF 为负  
        ② +1 偏移：确保 IDF 始终为正（即使某词出现在所有文档中）  
        ③ 如果用原始 TF-IDF 的 IDF：log(N/n)，当 n=N 时 IDF=0，  
           某些实现会出现负值，BM25 的改进版更稳健  
        """  
        for term, postings in self.inverted_index.items():  
            n = len(postings)  # 包含该词的文档数  
            # BM25 IDF 公式（Robertson & Sparck Jones 改进版）  
            self.idf[term] = math.log(  
                (self.N - n + 0.5) / (n + 0.5) + 1  
            )  
  
    def score(self, query: str, doc_id: int) -> float:  
        """  
        计算查询 query 对文档 doc_id 的 BM25 分数。  
        ① 这行做了什么：对查询中每个词，计算其对该文档的贡献并求和  
        ② 为什么分词查询：BM25 是词袋模型，每个词独立计分后求和  
        """  
        tokens = tokenize(query)  
        score = 0.0  
        doc_len = self.doc_lengths[doc_id]  
  
        for term in tokens:  
            if term not in self.inverted_index:  
                continue  # 词不在语料库中，贡献为 0（OOV 问题）  
  
            # 获取该词在该文档中的词频，不存在则为 0  
            tf = self.inverted_index[term].get(doc_id, 0)  
  
            # BM25 核心公式分子：tf × (k1 + 1)  
            numerator = tf * (self.k1 + 1)  
  
            # BM25 核心公式分母：tf + k1 × (1 - b + b × doc_len/avgdl)  
            # ① 长度归一化项：b × doc_len/avgdl  
            #    当 doc_len > avgdl（长文档），分母变大，分数降低  
            #    当 doc_len < avgdl（短文档），分母变小，分数升高  
            # ② 如果 b=0：长度归一化消失，退化为纯词频饱和  
            denominator = tf + self.k1 * (  
                1 - self.b + self.b * doc_len / self.avgdl  
            )  
  
            # 该词的贡献 = IDF × TF饱和值  
            score += self.idf[term] * (numerator / denominator)  
  
        return score  
  
    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:  
        """  
        返回 Top-K 最相关文档的 (doc_id, score) 列表。  
        ① sorted(..., reverse=True)：按分数从高到低排序  
        ② [:top_k]：只取前 K 个结果  
        """  
        scores = [  
            (doc_id, self.score(query, doc_id))  
            for doc_id in range(self.N)  
        ]  
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]  
  
  
# ============================================================  
# Step 5: 运行示例  
# ============================================================  
  
corpus = [  
    "Python is a great programming language for data science",  
    "Java is widely used in enterprise software development",  
    "Python and machine learning go hand in hand",  
    "Data science requires statistics and programming skills",  
    "Enterprise software often uses Java Spring framework",  
]  
  
bm25 = BM25(corpus, k1=1.5, b=0.75)  
results = bm25.search("Python data science", top_k=3)  
  
for rank, (doc_id, score) in enumerate(results, 1):  
    print(f"Rank {rank}: [Score={score:.4f}] {corpus[doc_id]}")
```

输出、

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777046040818-7bb3bc9f-061d-4d77-923f-3e9cecd4eab6.png" width="409" title="" crop="0,0,1,1" id="u895d48cc" class="ne-image">



##### <font style="color:rgb(51, 51, 51);">常见索引数据库</font>
###### <font style="color:rgb(0, 0, 0);">FAISS</font>
<font style="color:rgb(13, 13, 13);">只是一个</font>**<font style="color:rgb(0, 0, 0);">算法库（类似NumPy）</font>**<font style="color:rgb(13, 13, 13);">，它没有网络层、没有WAL日志、不支持多进程安全读写</font>

<font style="color:rgb(13, 13, 13);">Milvus/Qdrant的底层可能调用了FAISS，但它们在外面包了一层完整的数据库管理系统（CRUD、权限、分布式）</font>

<font style="color:rgb(51, 51, 51);">可以精细控制索引算法（IVF、PQ、HNSW 等参数）</font><font style="color:rgb(13, 13, 13);"></font>

###### <font style="color:rgb(0, 0, 0);">Milvus</font>
<font style="color:rgb(13, 13, 13);">面向</font>**<font style="color:rgb(0, 0, 0);">大规模生产集群</font>**<font style="color:rgb(13, 13, 13);">，云原生、存储计算分离、多索引、多硬件，偏“重”但可扩展到百亿级</font>

<font style="color:rgb(13, 13, 13);">底层可能调用了FAISS，但它们在外面包了一层完整的数据库管理系统（CRUD、权限、分布式）</font>

+ <font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">实践</font>

```plain
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")

# 0. 如果之前有残留的 demo 集合，先删掉（避免第二次运行报错）
if client.has_collection("demo"):
    client.drop_collection("demo")

# 1. 先准备索引参数
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 200,
    },
)

# 2. 创建集合时，直接把 index_params 传进去！(关键修改)
client.create_collection(
    collection_name="demo",
    dimension=128,
    metric_type="COSINE",
    auto_id=True,
    enable_dynamic_field=True,
    index_params=index_params  # <--- 加上这一行，一步到位建表+建索引
)

# 3. 插入数据（结合上一次的修复，auto_id=True时不传id）
dim = 128
num_vectors = 1000
vectors = np.random.random((num_vectors, dim)).astype(np.float32)
data = [
    {"vector": vectors[i], "text": f"doc-{i}"} 
    for i in range(num_vectors)
]
client.insert(collection_name="demo", data=data)

# 4. 加载集合（如果是通过 index_params 建表，通常建表时已自动加载，但显式调用保证安全）
client.load_collection("demo")

# 5. 查询
query_vector = np.random.random((1, dim)).astype(np.float32)
res = client.search(
    collection_name="demo",
    data=query_vector,
    limit=5,
    output_fields=["text"],
)

for hit in res[0]:
    print("id:", hit["id"], "score:", hit["distance"], "text:", hit["entity"]["text"])


```

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777105447071-703cb869-7130-4a0f-9660-2d5f3cd6e438.png" width="812" title="" crop="0,0,1,1" id="ud62338b4" class="ne-image">

可以看到存入了

###### <font style="color:rgb(0, 0, 0);">Qdrant</font>
<font style="color:rgb(13, 13, 13);">基于</font>**<font style="color:rgb(0, 0, 0);">Rust</font>**<font style="color:rgb(13, 13, 13);">编写，用Segment段文件管理，单节点性能极高，资源占用极小</font>

<font style="color:rgb(13, 13, 13);">其中</font>**<font style="color:rgb(0, 0, 0);">Segment (段)</font>**

+ **<font style="color:rgb(0, 0, 0);">定义</font>**<font style="color:rgb(13, 13, 13);">：向量数据库中数据持久化和索引构建的最小独立单元，类似LSM-Tree中的SSTable。</font>
+ **<font style="color:rgb(0, 0, 0);">类比</font>**<font style="color:rgb(13, 13, 13);">：一本本独立的字典，满了就封存，查询时同时查所有未封存的字典。</font>
+ **<font style="color:rgb(0, 0, 0);">一句话记忆</font>**<font style="color:rgb(13, 13, 13);">：数据落盘与索引的原子单位。</font>

<font style="color:rgb(13, 13, 13);">部署下Qdrant</font>

<font style="color:rgb(13, 13, 13);">docker run -p 6333:6333 qdrant/qdrant:latest</font>

+ 实践

```plain
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 连接 Qdrant（本地服务器或内存模式）
# 如果是本地服务器：client = QdrantClient(host="localhost", port=6333)
# 这里使用内存模式，免安装直接跑：
client = QdrantClient(":memory:")

# 2. 创建 collection
client.recreate_collection(
    collection_name="demo_qdrant",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
)

# 3. 插入点
points = [
    PointStruct(
        id=i,
        vector=np.random.rand(128).astype(np.float32).tolist(),
        payload={"label": i % 10},
    )
    for i in range(1000)
]
client.upsert(collection_name="demo_qdrant", points=points)

# 4. 搜索 (使用最新的 query_points API)
query_vector = np.random.rand(128).astype(np.float32).tolist()

search_results = client.query_points(
    collection_name="demo_qdrant",
    query=query_vector,        # 注意：旧版是 query_vector=，新版是 query=
    limit=5,
)

# 5. 解析并打印结果
# 注意：query_points 返回的是 QueryResponse 对象，实际数据在 .points 属性中
for point in search_results.points:
    print(f"id={point.id}, score={point.score}, payload={point.payload}")
```

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777108474785-2537687d-5ae8-49dc-a518-446131afb459.png" width="527" title="" crop="0,0,1,1" id="u5977d2fb" class="ne-image">

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);"></font>

###### <font style="color:rgb(0, 0, 0);">Chroma</font>
```plain
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# ① 连接内存模式的 Chroma 客户端
client = chromadb.Client()

# ② 创建 Ollama 嵌入函数
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",     # Ollama 默认地址
    model_name="bge-m3",   # Ollama 里的嵌入模型，可换别的
)

# ③ 创建集合，并指定使用 Ollama 嵌入
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=ollama_ef,      # 关键：用 Ollama 而不是内置 MiniLM
)

# ④ 添加文档（现在会用 Ollama 模型来向量化）
collection.add(
    documents=["机器学习是人工智能的子集", "今天天气真好适合出门", "深度学习依赖于神经网络"],
    ids=["doc1", "doc2", "doc3"],
)

# ⑤ 语义查询（同样会走 Ollama 嵌入）
results = collection.query(
    query_texts=["AI 技术"],
    n_results=2,
)

print(results["documents"])
# 预期：最相关的两条中文文档被召回
```



+ 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777117326389-565e2a99-e528-4664-8e77-0c20321622da.png" width="506" title="" crop="0,0,1,1" id="u04ec8aa2" class="ne-image">

<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">向量数据库的基本交互范式是 </font>`**<font style="color:rgb(13, 13, 13);background-color:rgb(236, 236, 236);">Client -> Collection -> Add -> Query</font>**`<font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">，语义检索能够跨过关键字匹配</font>



###### <font style="color:rgb(0, 0, 0);">Elasticsearch (ES)</font>
+ <font style="color:rgb(13, 13, 13);">每个 </font>**<font style="color:rgb(0, 0, 0);">Elasticsearch 索引</font>**<font style="color:rgb(13, 13, 13);"> 由多个 </font>**<font style="color:rgb(0, 0, 0);">分片（shard）</font>**<font style="color:rgb(13, 13, 13);"> 组成，每个分片本身就是一个 Lucene 索引。</font>
+ <font style="color:rgb(13, 13, 13);">Lucene 索引的核心是 </font>**<font style="color:rgb(0, 0, 0);">倒排索引</font>**<font style="color:rgb(13, 13, 13);">：从词项（term）映射到包含它的文档列表。</font>
+ <font style="color:rgb(13, 13, 13);">文本搜索流程：分词 → 倒排索引查找 → BM25 打分 → 排序返回。</font>
+ <font style="color:rgb(13, 13, 13);">向量搜索：通过 kNN 查询（如 </font>`**<font style="color:rgb(13, 13, 13);">knn_search</font>**`<font style="color:rgb(13, 13, 13);"> API）在 HNSW 等图结构上做 ANN 检索</font>

<font style="color:rgb(13, 13, 13);">docker部署es：docker run -d --name es01 -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.0</font>

+ <font style="color:rgb(13, 13, 13);">实践</font>

```plain
from elasticsearch import Elasticsearch
import numpy as np

# ------------------- ① 连接 ES 并创建索引 -------------------
es = Elasticsearch("http://localhost:9200") # 连接本地 ES
# 做什么：实例化 ES 客户端
# 为什么：通过 REST API 与 ES 交互
# 改变：如果开启了安全认证，这里需配置 http_auth 或 api_key

index_name = "rag_hybrid_index"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name) # 清理旧索引

mapping = {
    "mappings": {
        "properties": {
            "text_content": {             # 文本字段
                "type": "text",           # 做什么：定义为全文检索类型
                "analyzer": "standard"    # 为什么：使用标准分词器（IK 需要额外安装）
            },
            "embedding": {                # 向量字段
                "type": "dense_vector",   # 做什么：ES 8.x 使用 dense_vector 类型
                "dims": 4,                # 为什么：必须与后续灌入的向量维度严格一致
                "index": True,            # 做什么：启用向量索引（默认使用 HNSW）
                "similarity": "l2_norm",  # 为什么：使用 L2 距离计算相似度
                "index_options": {        # ✅ 显式配置 HNSW 参数
                    "type": "hnsw",       # 索引类型：HNSW
                    "m": 16,              # 每层最大连接数（默认16，越大越准越慢）
                    "ef_construction": 100 # 构建时搜索宽度（默认100）
                }
            }
        }
    }
    # 注意：ES 8.12 不需要 index.knn 设置，KNN 功能默认内置
}
es.indices.create(index=index_name, body=mapping)

# ------------------- ② 灌入模拟文档 -------------------
docs = [
    {"text_content": "FAISS 是一个向量检索库", "embedding": [1.0, 1.0, 1.0, 1.0]},
    {"text_content": "Elasticsearch 支持全文搜索", "embedding": [2.0, 2.0, 2.0, 2.0]},
    {"text_content": "大模型经常会发生幻觉", "embedding": [3.0, 3.0, 3.0, 3.0]}
]

for i, doc in enumerate(docs):
    es.index(index=index_name, id=i, document=doc)
    # 做什么：将文档以指定 ID 写入 ES
    # 为什么：固定 ID 便于测试，否则 ES 自动生成随机 ID
    # 改变：如果不指定 ID，相同文档重复 index 会生成多个副本

es.indices.refresh(index=index_name) # 强制刷新，确保数据可被立刻搜索
# 做什么：让刚写入的数据从内存 Buffer 刷入 Segment 变为可搜索状态
# 为什么：ES 默认 1 秒刷新一次，测试时需立刻可见
# 改变：生产环境中频繁 refresh 会严重影响写入性能

# ------------------- ③ 执行混合检索 -------------------
query_embedding = [1.1, 1.1, 1.1, 1.1]
query_text = "向量"

# ES 8.12 KNN 查询语法：使用顶层 knn 参数，不是 bool.should
# 参考：https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html

# 方式1：纯 KNN 搜索（使用 knn 顶层参数）
try:
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 2,
            "num_candidates": 10
        },
        size=2
    )
    print("✅ KNN 检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")
except Exception as e:
    print(f"❌ KNN 查询失败: {e}")

# 方式2：混合搜索（KNN + 文本过滤）
print("\n" + "="*50)
print("尝试混合搜索（KNN + 文本过滤）...")
try:
    # ES 8.12 支持在 knn 查询中添加 filter
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 2,
            "num_candidates": 10,
            "filter": {
                "match": {"text_content": query_text}
            }
        },
        size=2
    )
    print("✅ 混合检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")
except Exception as e:
    print(f"❌ KNN+Filter 查询失败: {e}")


# 方式3：手动实现混合搜索（免费版可用）
print("\n" + "="*50)
print("方式3：手动实现混合搜索（加权合并得分）...")
try:
    # 分别获取向量搜索结果和文本搜索结果
    knn_response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 5,
            "num_candidates": 10
        },
        size=5
    )
    
    text_response = es.search(
        index=index_name,
        query={"match": {"text_content": query_text}},
        size=5
    )
    
    # 手动合并得分（加权平均）
    from collections import defaultdict
    
    scores = defaultdict(lambda: {"knn": 0, "text": 0, "doc": None})
    
    # 归一化 KNN 得分（转为 0-1 范围）
    knn_hits = knn_response['hits']['hits']
    if knn_hits:
        max_knn = max(hit['_score'] for hit in knn_hits)
        min_knn = min(hit['_score'] for hit in knn_hits)
        knn_range = max_knn - min_knn if max_knn != min_knn else 1
        
        for hit in knn_hits:
            doc_id = hit['_id']
            normalized_score = (hit['_score'] - min_knn) / knn_range
            scores[doc_id]["knn"] = normalized_score
            scores[doc_id]["doc"] = hit['_source']
    
    # 归一化文本得分
    text_hits = text_response['hits']['hits']
    if text_hits:
        max_text = max(hit['_score'] for hit in text_hits)
        min_text = min(hit['_score'] for hit in text_hits)
        text_range = max_text - min_text if max_text != min_text else 1
        
        for hit in text_hits:
            doc_id = hit['_id']
            normalized_score = (hit['_score'] - min_text) / text_range
            scores[doc_id]["text"] = normalized_score
            if scores[doc_id]["doc"] is None:
                scores[doc_id]["doc"] = hit['_source']
    
    # 加权合并 (alpha 控制权重: 0=纯文本, 1=纯向量)
    alpha = 0.5  # 向量权重
    beta = 0.5   # 文本权重
    
    final_scores = []
    for doc_id, data in scores.items():
        combined_score = alpha * data["knn"] + beta * data["text"]
        final_scores.append((doc_id, combined_score, data["doc"]))
    
    # 排序取 Top-2
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✅ 手动混合检索结果 (向量权重={alpha}, 文本权重={beta}):")
    for doc_id, score, doc in final_scores[:2]:
        print(f"  ID: {doc_id}, 合并得分: {score:.4f}, Text: {doc['text_content']}")
        print(f"      (向量得分: {scores[doc_id]['knn']:.4f}, 文本得分: {scores[doc_id]['text']:.4f})")
        
except Exception as e:
    print(f"❌ 手动混合查询失败: {e}")
    print("\n尝试仅文本搜索...")
    # 降级为纯文本搜索
    text_query = {"match": {"text_content": query_text}}
    response = es.search(index=index_name, query=text_query, size=2)
    print("✅ 文本检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")

# 查看索引配置
print("\n" + "="*50)
print("【索引配置详情】")

# 查看 settings 中的索引参数
settings = es.indices.get_settings(index=index_name)
print("\n索引设置 (Settings):")
print(f"  分片数: {settings[index_name]['settings']['index']['number_of_shards']}")
print(f"  副本数: {settings[index_name]['settings']['index']['number_of_replicas']}")

# 查看 mapping 详情
mapping = es.indices.get_mapping(index=index_name)
embedding_config = mapping[index_name]['mappings']['properties']['embedding']
print("\n向量字段配置 (Mapping):")
print(f"  类型: {embedding_config['type']}")
print(f"  维度: {embedding_config['dims']}")
print(f"  索引: {embedding_config['index']}")
print(f"  相似度: {embedding_config['similarity']}")

# 检查是否有 index_options（自定义 HNSW 参数）
if 'index_options' in embedding_config:
    print(f"  索引选项: {embedding_config['index_options']}")
else:
    print("  索引选项: 使用默认值 (HNSW, m=16, ef_construction=100)")

# 验证 HNSW 是否生效 - 查看段信息
print("\n" + "="*50)
print("【验证 HNSW 索引】")
segments = es.indices.segments(index=index_name)
      
```

+ <font style="color:rgb(13, 13, 13);">输出</font>

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777173437215-fa5eabd9-f480-46df-b849-813030dab427.png" width="652" title="" crop="0,0,1,1" id="uac1782c8" class="ne-image">



###### <font style="color:rgb(0, 0, 0);">OpenSearch</font>
<font style="color:rgb(13, 13, 13);">OpenSearch 源自 Elasticsearch 7.10 开源分支，架构（分片、副本、倒排索引）与 ES 高度相似</font>

<font style="color:rgb(13, 13, 13);">向量搜索：通过 k-NN 插件支持，底层可选用 </font>**<font style="color:rgb(0, 0, 0);">Faiss / Lucene / nmslib</font>**<font style="color:rgb(13, 13, 13);"> 引擎</font>

+ 实践
  - docker 启动 docker run -d --name opensearch-node -p 9200:9200 -p 9600:9600 -e 'discovery.type=single-node' -e 'plugins.security.disabled=true' -e 'OPENSEARCH_INITIAL_ADMIN_PASSWORD=MyStrongPass123!' opensearchproject/opensearch:2.11.0



```plain
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import numpy as np

# ------------------- ① 连接 OpenSearch -------------------
host = 'localhost'
port = 9200
auth = ('admin', 'MyStrongPass123!') # OpenSearch 默认强制开启安全认证

# 注意：如果 OpenSearch 没有启用 HTTPS，需要将 use_ssl 设为 False
# 检查方式：curl http://localhost:9200 或 curl https://localhost:9200 -k
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_auth = auth,
    use_ssl = False,  # ✅ 改为 False，你的 OS 容器未启用 HTTPS
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    scheme = 'http'   # 显式指定 http 协议
)
# 做什么：实例化 OpenSearch 客户端
# 为什么：OpenSearch 默认启用 HTTPS 和安全认证，必须配置 auth 和 SSL 参数
# 改变：如果像 ES 那样关闭安全认证，这里可简化，但生产环境强烈不建议

print("OpenSearch 信息:", client.info())

# ------------------- ② 创建 k-NN 索引 -------------------
index_name = "opensearch_rag_index"
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

# OpenSearch 独有的 k-NN 索引配置
index_body = {
    "settings": {
        "index.knn": True  # 做什么：显式开启 k-NN 插件功能
        # 为什么：这是 OpenSearch 向量检索的总开关，不开启后续向量查询无效
        # 注意：index.knn.algo_param.ef_search 在 OpenSearch 2.4+ 已移除
        # ef_search 现在作为查询参数传入，不在索引设置中配置
    },
    "mappings": {
        "properties": {
            "text_content": {"type": "text"},
            "embedding": {
                "type": "knn_vector", # 做什么：使用 OpenSearch 专属的 knn_vector 类型
                # 为什么：与 ES 8.x 的 dense_vector 不同，它直接绑定 k-NN 插件的底层算法
                "dimension": 4,
                "method": {
                    "name": "hnsw",       # 算法名称
                    "space_type": "l2",   # 距离度量
                    "engine": "faiss",    # 做什么：指定底层计算引擎为 Faiss
                    # 为什么：OpenSearch 支持nmslib/faiss/lucene，Faiss 引擎在大规模和高维场景性能更优，且支持 IVF/PQ 等高级压缩
                    # 改变：如果用 nmslib，则不支持磁盘向量；如果用 lucene，则遵循 ES 8.x 的一些特性
                    "parameters": {
                        "m": 16,          # HNSW 图的节点连接数
                        "ef_construction": 256 # 做什么：构建索引时的候选集大小
                        # 为什么：构建时的参数，越大图质量越高，构建越慢
                    }
                }
            },
            "category": {"type": "keyword"} # 用于后续测试过滤
        }
    }
}

client.indices.create(index=index_name, body=index_body)

# ------------------- ③ 批量灌入模拟数据 -------------------
docs = [
    {"text_content": "FAISS 是 Meta 开源的向量库", "embedding": [1.0, 1.0, 1.0, 1.0], "category": "vector_db"},
    {"text_content": "OpenSearch 源自 Elasticsearch 7.10", "embedding": [2.0, 2.0, 2.0, 2.0], "category": "search_engine"},
    {"text_content": "大模型 RAG 架构缓解幻觉", "embedding": [3.0, 3.0, 3.0, 3.0], "category": "llm"},
    {"text_content": "Faiss 支持 IVF 索引加速", "embedding": [1.1, 1.1, 1.1, 1.1], "category": "vector_db"},
]

# 使用 bulk API 批量写入
actions = [
    {"_index": index_name, "_id": i, "_source": doc}
    for i, doc in enumerate(docs)
]
bulk(client, actions)
# 做什么：高效批量写入文档
# 为什么：比循环调用 index 快得多，是生产环境标配

client.indices.refresh(index=index_name)

# ------------------- ④ 执行检索 -------------------
query_embedding = [1.05, 1.05, 1.05, 1.05]

# 方式 1：纯 k-NN 搜索 (使用 OpenSearch 专属的 knn 查询子句)
knn_query = {
    "size": 2,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_embedding,
                "k": 2
            }
        }
    }
}

response = client.search(body=knn_query, index=index_name)
print("\n✅ OpenSearch 纯 KNN 结果:")
for hit in response['hits']['hits']:
    print(f"  ID: {hit['_id']}, Score: {hit['_score']:.4f}, Text: {hit['_source']['text_content']}")

# 方式 2：高效过滤搜索 (Efficient Filtering)
# 业务需求：找向量相似的，但只要 category = "vector_db" 的文档
filter_query = {
    "size": 2,
    "query": {
        "bool": {
            "filter": [ # 做什么：在此处放置过滤条件
                {"term": {"category": "vector_db"}}
            ],
            "must": [ # 做什么：在此处放置 k-NN 查询
                {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": 2
                        }
                    }
                }
            ]
        }
    }
}
# 为什么：OpenSearch 的 k-NN 插件会智能识别 bool.filter，在 HNSW 图遍历时直接跳过不满足条件的节点
# 改变：如果把 knn 放 filter 里，文本放 must 里，逻辑不变，但 must 会参与算分

response = client.search(body=filter_query, index=index_name)
print("\n✅ OpenSearch 高效过滤 KNN 结果:")
for hit in response['hits']['hits']:
    print(f"  ID: {hit['_id']}, Score: {hit['_score']:.4f}, Text: {hit['_source']['text_content']}, Category: {hit['_source']['category']}")
```

注意这里的Docker 容器可能启动时禁用了安全认证（ DISABLE_SECURITY_PLUGIN=true ），所以用 HTTP就行了

+ 输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777175474044-b9367530-5166-4ffd-a868-5996f2a88918.png" width="502" title="" crop="0,0,1,1" id="u51652e71" class="ne-image">



###### <font style="color:rgb(0, 0, 0);">Neo4j</font>
+ <font style="color:rgb(13, 13, 13);">Neo4j 使用 </font>**<font style="color:rgb(0, 0, 0);">属性图模型</font>**<font style="color:rgb(13, 13, 13);">：节点、关系、属性，原生图存储，无索引邻接（index-free adjacency）。</font>
+ <font style="color:rgb(13, 13, 13);">查询语言 Cypher：声明式图模式匹配，如 </font>`**<font style="color:rgb(13, 13, 13);">(a:Person)-[:KNOWS]->(b:Person)</font>**`<font style="color:rgb(13, 13, 13);">。</font>
+ <font style="color:rgb(13, 13, 13);">适合多跳遍历、路径查询、图算法（PageRank、社区发现等）</font><font style="color:rgb(13, 13, 13);background-color:rgb(248, 248, 248);">。</font>



+ 实践

docker 启动 docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

```plain
from neo4j import GraphDatabase
import numpy as np

# ------------------- ① 连接 Neo4j 并创建节点 -------------------
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
# 做什么：通过 Bolt 协议连接图数据库
# 为什么：Bolt 是专用的二进制协议，比 HTTP 性能更好

def create_graph(tx):
    # 清理旧数据
    tx.run("MATCH (n) DETACH DELETE n")
    
    # 创建带有 embedding 的实体节点
    tx.run("""
    CREATE (a:Entity {name: 'FAISS', embedding: [1.0,1.0,1.0,1.0]})
    CREATE (b:Entity {name: 'HNSW', embedding: [1.1,1.1,1.1,1.1]})
    CREATE (c:Entity {name: 'Elasticsearch', embedding: [5.0,5.0,5.0,5.0]})
    CREATE (a)-[:DEPENDS_ON]->(b)  // FAISS 依赖 HNSW 算法
    CREATE (c)-[:INTEGRATES]->(b)  // ES 集成了 HNSW 算法
    """)
    # 做什么：创建节点并建立 DEPENDS_ON 和 INTEGRATES 关系
    # 为什么：这是 GraphRAG 的核心优势，关系蕴含了逻辑推理路径
    # 改变：如果不加关系，退化为纯向量检索，丧失多跳能力

with driver.session() as session:
    session.execute_write(create_graph)

# ------------------- ② 创建向量索引 -------------------
def create_vector_index(tx):
    tx.run("""
    CREATE VECTOR INDEX entity_embedding_index FOR (n:Entity) ON (n.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 4,
        `vector.similarity_function`: 'cosine'
    }}
    """)
    # 做什么：为 Entity 节点的 embedding 属性创建向量索引
    # 为什么：必须创建索引，后续的向量相似度查询才能走索引加速
    # 改变：如果 dimensions 填错，创建会失败；如果不建索引，只能全库扫描计算

with driver.session() as session:
    session.execute_write(create_vector_index)

# ------------------- ③ 执行 Cypher 组合查询 -------------------
query_embedding = [1.05, 1.05, 1.05, 1.05] # 模拟 FAISS 的查询向量

def graph_rag_query(tx, query_vec):
    result = tx.run("""
    // 第一步：通过向量索引找到最相似的起始节点
    CALL db.index.vector.queryNodes('entity_embedding_index', 1, $queryVec)
    YIELD node AS start_node, score
    
    // 第二步：基于起始节点进行图遍历，获取关联上下文
    MATCH path = (start_node)-[r*1..2]-(connected_node)
    RETURN start_node.name AS core_entity, 
           score AS similarity,
           connected_node.name AS context_entity,
           [rel in relationships(path) | type(rel)] AS relations
    """, queryVec=query_vec)
    # 做什么：先执行向量检索拿到 start_node，再执行图 MATCH 拿到多跳关系
    # 为什么：单靠向量找不到“ES 和 FAISS 都依赖 HNSW”这种关联逻辑
    # 改变：如果去掉 CALL 部分，只能做全局图遍历，失去语义聚焦能力
    
    for record in result:
        print(f"核心实体: {record['core_entity']} (相似度: {record['similarity']:.2f})")
        print(f"关联上下文: {record['context_entity']}, 关系路径: {record['relations']}")

with driver.session() as session:
    session.execute_read(graph_rag_query, query_embedding)

driver.close()
```

输出

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777174850023-c751ed62-c37d-4334-88fb-a948dabdd05c.png" width="508" title="" crop="0,0,1,1" id="u7c643b6b" class="ne-image">

原因：查询向量 [1.05, 1.05, 1.05, 1.05] 与 FAISS 的 embedding [1.0,1.0,1.0,1.0] 最相似（余弦相似度 1.00），所以核心实体是 FAISS。

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1777175053017-aff46183-2e33-45f3-bf32-8407ba54ff29.png" width="483" title="" crop="0,0,1,1" id="ud30a3e29" class="ne-image">

输出展示了 "FAISS 和 Elasticsearch 都依赖 HNSW" 这一关联逻辑，这正是纯向量检索无法发现的语义关系

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>

### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">查询向量化</font>**
这个步骤和离线阶段的数据embedding向量化相呼应

**<font style="color:rgb(0, 0, 0);">概念</font>**<font style="color:rgb(13, 13, 13);">： 现实世界的是文本（“苹果公司发布新手机”）、图片、音频，计算机只认数字。向量化就是用深度学习模型（Embedding 模型），把任何一段文本压缩成一个</font>**<font style="color:rgb(0, 0, 0);">高维浮点数数组</font>**<font style="color:rgb(13, 13, 13);">（比如 1024 维的向量）。</font>

**<font style="color:rgb(0, 0, 0);">人海比喻</font>**<font style="color:rgb(13, 13, 13);">： 把一个人所有的特征（身高、性格、收入、爱好）提取出来，写进一张</font>**<font style="color:rgb(0, 0, 0);">标准化的简历</font>**<font style="color:rgb(13, 13, 13);">。</font>

**<font style="color:rgb(0, 0, 0);">核心特点</font>**<font style="color:rgb(13, 13, 13);">：</font>

+ **<font style="color:rgb(0, 0, 0);">语义捕获</font>**<font style="color:rgb(13, 13, 13);">：好的模型能让意思相近的文本，变成距离相近的向量。“汽车”和“轿车”的向量很近，“汽车”和“苹果”的向量很远。</font>
+ **<font style="color:rgb(0, 0, 0);">只管质，不管速</font>**<font style="color:rgb(13, 13, 13);">：它只负责把简历写好写准，不管你后面怎么搜。</font>
+ **<font style="color:rgb(0, 0, 0);">RAG中的常见坑</font>**<font style="color:rgb(13, 13, 13);">：如果模型拉胯（比如用了个很老的 Word2Vec），简历写错了，后面再怎么精准搜索也白搭。</font>

<font style="color:rgb(13, 13, 13);">在RAG（检索增强生成）系统中，在线查询阶段是整个系统的"入口"。当用户输入一段自然语言查询时，系统必须将其转化为机器可以计算的形式，这就是</font>**<font style="color:rgb(0, 0, 0);">查询向量化</font>**<font style="color:rgb(13, 13, 13);">的核心使命。</font>

<font style="color:rgb(13, 13, 13);"></font>

### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">相似度搜索</font>**


### <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">重排序（Reranking）</font>


### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">上下文构建</font>**


### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">提示构建</font>**


### <font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">LLM生成</font>
**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);"></font>**

### **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">结果后处理</font>**







