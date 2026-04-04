# agent概念学习
### 智能体概念
智能体简单理解为=llm节点调用+工具选择与调用+循环

他会主动思考去纠正自我学习规划任务完成任务优化任务

对比传统的开发需定义多个系统，由用户去决策，在多个专家系统中来回切换



整体流程： 规划与推理->工具选择与调用->动态修正（将用户反馈作为新的约束继续规划和推理执行）



### CoT、Few-shot（以下内容来源于GLM5回答）
Cot：Chain of Thought，思维链  
case：

```plain
┌─────────────────────────────────────────────────────────────┐
│                    CoT 的本质                                │
├─────────────────────────────────────────────────────────────┤
│                                                             
│  不使用CoT（直接给答案）：                                  
│                                                             
│  问题："小明有5个苹果，给了小红2个，又买了3个，还有几个？"  
│  回答："6个"                                               
│         ↑                                                  
│         直接跳到答案，可能出错                             
│                                                             
│  ────────────────────────────────────────────────────────  
│                                                             
│  使用CoT（一步步思考）：                                    
│                                                             
│  问题："小明有5个苹果，给了小红2个，又买了3个，还有几个？"  
│                                                             
│  思考过程：                                                 
│    1. 小明最初有5个苹果                                     
│    2. 给了小红2个，剩下：5 - 2 = 3个                       
│    3. 又买了3个，现在有：3 + 3 = 6个                       
│    4. 所以答案是6个                                         
│                                                             
│  回答："6个"                                               
│         ↑                                                  
│         展示思考过程，答案更准确                           
│                                                             
└─────────────────────────────────────────────────────────────┘
```



CoT比直接输出预测更靠谱的原理

```plain
┌─────────────────────────────────────────────────────────────┐
│                    CoT 的原理                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  大模型的工作方式：                                         │
│                                                             │
│  输入 → [预测下一个token] → 输出                            │
│                                                             │
│  不使用CoT：                                               │
│  "问题：... 答案：[直接预测答案]"                           │
│           ↑                                                │
│           跨度太大，容易出错                               │
│                                                             │
│  使用CoT：                                                 │
│  "问题：... 让我们一步步思考：                              │
│   第一步：...                                              │
│   第二步：...                                              │
│   第三步：...                                              │
│   所以答案是：..."                                         │
│        ↑                                                   │
│        每一步都是小跨度预测，更准确                        │
│                                                             │
│  核心思想：                                                │
│  • 把复杂问题分解成简单步骤                                │
│  • 每一步都是小推理，降低出错概率                          │
│  • 前面的步骤为后面的步骤提供上下文                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

实际的大模型应用



```plain
# 数学问题示例
prompt = """
问题：一个长方形，长是宽的2倍，周长是36厘米，求面积？

让我们一步步思考：
1. 设宽为x，则长为2x
2. 周长公式：2(长 + 宽) = 36
3. 代入：2(2x + x) = 36
4. 化简：6x = 36
5. 解得：x = 6（宽）
6. 长 = 2x = 12
7. 面积 = 长 × 宽 = 12 × 6 = 72

答案：72平方厘米
"""

# 逻辑推理示例
prompt = """
问题：如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？

让我们一步步思考：
1. 前提1：所有的A都是B（A ⊆ B）
2. 前提2：所有的B都是C（B ⊆ C）
3. 根据集合的传递性：如果A ⊆ B且B ⊆ C，则A ⊆ C
4. 因此：所有的A都是C

答案：是的
"""

# Agent中的CoT应用
agent_prompt = """
你是一个智能助手，请按以下格式思考和行动：

问题：[用户的问题]

思考过程：
1. 理解问题：问题的核心是什么？
2. 分析需求：需要哪些信息？
3. 制定计划：分几步解决？
4. 执行步骤：
   - 步骤1：[具体操作]
   - 步骤2：[具体操作]
   - ...
5. 验证结果：结果是否合理？

最终答案：[回答]
"""
```



各类CoT

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775057976611-357c2b27-7084-4aee-b50d-c702986e2b4e.png" width="287" title="" crop="0,0,1,1" id="u493bb3ab" class="ne-image">

Few-shot（少样本学习）

<img src="https://cdn.nlark.com/yuque/0/2026/png/21570810/1775057952605-2c843463-36a7-48ed-9053-3750ef70ef56.png" width="321" title="" crop="0,0,1,1" id="u593f3287" class="ne-image">





```plain
# 情感分析任务
prompt = """
判断以下句子的情感（正面/负面/中性）：

示例1：
句子："这家餐厅的菜很好吃，服务也很周到。"
情感：正面

示例2：
句子："等了一个小时才上菜，太失望了。"
情感：负面

示例3：
句子："这家店在市中心，交通便利。"
情感：中性

现在请判断：
句子："产品质量不错，但价格有点贵。"
情感：
"""

# 实体抽取任务
prompt = """
从文本中提取人名、地点、组织：

示例1：
文本："张三在北京的阿里巴巴工作。"
提取：{"人名": ["张三"], "地点": ["北京"], "组织": ["阿里巴巴"]}

示例2：
文本："李四去了上海，拜访了腾讯公司。"
提取：{"人名": ["李四"], "地点": ["上海"], "组织": ["腾讯"]}

现在请提取：
文本："王五从深圳来到广州，加入了字节跳动。"
提取：
"""

# 代码生成任务
prompt = """
根据描述生成Python函数：

示例1：
描述："计算列表中所有数字的和"
代码：
def sum_list(numbers):
    return sum(numbers)

示例2：
描述："找出列表中的最大值"
代码：
def find_max(numbers):
    return max(numbers)

现在请生成：
描述："计算列表中数字的平均值"
代码：
"""
```



### skill
skill：<font style="color:rgb(66, 66, 66);">A simple, open format for giving agents new capabilities and expertise.（来源于</font>[https://agentskills.io/home](https://agentskills.io/home) 介绍<font style="color:rgb(66, 66, 66);">）</font>

<font style="color:rgb(66, 66, 66);">一种简单、开放的格式，用于为智能体赋予新功能和专业知识。</font>

<font style="color:rgb(66, 66, 66);">技能是一系列指令、脚本和资源的文件夹，智能体可以发现和使用这些内容，</font><font style="color:rgb(34, 34, 34);">Agent 可以按需发现和使用这些 Skill 来扩展自身能力。</font><font style="color:rgb(66, 66, 66);">以更准确、高效地完成任务</font>

<font style="color:rgb(34, 34, 34);"> Skill 的核心是一个 </font>`<font style="color:rgb(34, 34, 34);background-color:rgba(0, 0, 0, 0.05);">SKILL.md</font>`<font style="color:rgb(34, 34, 34);"> 文件，包含元数据（至少需要 name 和 description）和指导 Agent 执行特定任务的说明。</font>




### agent第一章实践
这里参考hello-agent
前置需到 https://app.tavily.com/ 注册下拿到api key 
这里采用了ollama部署，没有采用hello-agent的openAI格式
```python
import requests

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"

import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 将所有工具函数放入一个字典，方便后续调用 这里还没有调用的
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
print("✅ 工具注册完成（此时函数未被调用）")

from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


import re
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# --- 1. 配置LLM客户端 ---
# 优先使用Ollama本地模型（如果可用）
# Ollama提供OpenAI兼容API，运行在 http://localhost:11434/v1

# 检测是否使用Ollama
USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() == "true"

if USE_OLLAMA:
    # Ollama配置（本地运行，免费）
    API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")  # Ollama不需要真实密钥
    BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    MODEL_ID = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")  # 可选: deepseek-r1:8b, qwen3:1.7b
    print(f"🦙 使用Ollama本地模型: {MODEL_ID}")
else:
    # OpenAI配置（需要API密钥）
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_ID = os.environ.get("OPENAI_MODEL", "gpt-4")
    print(f"☁️  使用OpenAI模型: {MODEL_ID}")

# Tavily API密钥（景点搜索功能）
if not os.environ.get("TAVILY_API_KEY"):
    print("⚠️  警告: 未配置TAVILY_API_KEY，景点推荐功能将不可用")
    print("   获取免费密钥: https://tavily.com/")
    print("   设置方法: export TAVILY_API_KEY='your_key'")

# 检查Ollama服务是否运行
if USE_OLLAMA:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        print("✅ Ollama服务运行正常")
    except:
        print("❌ 错误: Ollama服务未运行")
        print("   请先启动Ollama服务: ollama serve")
        print("   或在另一个终端运行: ollama run qwen3:4b")
        exit(1)

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)


def main():
    """
    主函数：运行ReAct Agent主循环
    """
    # --- 2. 初始化 ---
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    # --- 3. 运行主循环 ---
    for i in range(5):  # 设置最大循环次数
        print(f"--- 循环 {i+1} ---\n")
        
        # 3.1. 构建Prompt
        full_prompt = "\n".join(prompt_history)
        
        # 3.2. 调用LLM进行思考
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        # 模型可能会输出多余的Thought-Action，需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)
        
        # 3.3. 解析并执行行动
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break
        
        # 解析工具调用
        tool_match = re.search(r"(\w+)\(", action_str)
        if not tool_match:
            observation = f"错误: 无法解析工具名称，Action格式应为: function_name(arg=\"value\")，实际为: {action_str}"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
            
        tool_name = tool_match.group(1)
        args_match = re.search(r"\((.*)\)", action_str)
        if not args_match:
            observation = f"错误: 无法解析工具参数，Action格式应为: function_name(arg=\"value\")，实际为: {action_str}"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
            
        args_str = args_match.group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        if tool_name in available_tools:
            # 这里开始调用工具函数
            print(f"🔧 正在调用工具: {tool_name}({kwargs})")
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # 3.4. 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)

# 系统提示词 这里模拟了输出哪个工具的
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""


if __name__ == "__main__":
    main()

```

##### 返回
![返回.png](https://cdn.nlark.com/yuque/0/2026/png/21570810/1775308932172-2d605ef9-662d-4bb9-a7a2-d8ec5548aa58.png?x-oss-process=image%2Fformat%2Cwebp)

核心思想就是通过系统提示词拿到需调用的工具为天气weather 则回去调用天气调用api拿到当前的天气
循环调用llm拿到工具为景点api、调用https://app.tavily.com/home 拿到天气适合旅游的景点
拿到之前的返回拼接prompt_history 当当收集到足够信息可以回答用户问题时再去调用llm 则会结束调用即observing观察过程







# langchain框架
# langgraph框架