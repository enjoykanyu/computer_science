# 手动实现了余弦相似度的计算
# import numpy as np

# # ① 定义两个模拟的嵌入向量 (假设维度为4)
# vec_query = np.array([0.5, 0.2, -0.1, 0.8])
# vec_doc = np.array([0.6, 0.1, -0.2, 0.7])

# # ② 计算点积 (分子)
# # ① 这行做了什么：计算两个向量的内积（对应元素相乘后求和）
# # ② 为什么这样写：余弦相似度公式的分子部分，衡量向量在同方向上的投影累积
# # ③ 如果改成改用减法：将变成计算差异，而非相似度累加
# dot_product = np.dot(vec_query, vec_doc)

# # ③ 计算范数乘积 (分母)
# # ① 这行做了什么：分别求两个向量的L2范数（欧氏长度）并相乘
# # ② 为什么这样写：用于归一化，消除文档长度（向量模长）对相似度的影响
# # ③ 如果改成去掉分母：等价于计算点积，长文档会因为有更多非零元素而获得不公平的高分
# norm_query = np.linalg.norm(vec_query)
# norm_doc = np.linalg.norm(vec_doc)

# # ④ 计算余弦相似度
# cosine_sim = dot_product / (norm_query * norm_doc)

# print(f"余弦相似度: {cosine_sim:.4f}")
# # 预期输出: 接近 0.96，说明两者高度相似

# langchain 实现余弦相似度
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
 
# # 初始化（使用 bge-m3，中文效果更好的嵌入模型）
# embeddings = OllamaEmbeddings(model="bge-m3")
 
# # 创建向量库
# docs = ["猫很可爱", "狗爱玩球", "鸟会飞翔"]
# vectorstore = Chroma.from_texts(texts=docs, embedding=embeddings)
 
# # 搜索
# results = vectorstore.similarity_search_with_relevance_scores("计算机", k=2)
 
# # 输出
# for doc, score in results:
#     print(f"{score:.3f} - {doc.page_content}")

#langgraph 实现余弦相似度
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    query: str
    documents: Annotated[list, operator.add]
    similarity_score: float
    route: str
    answer: str

# ✅ 初始化 Embedding
embeddings = OllamaEmbeddings(model="bge-m3")

knowledge_base = [
    "猫是独立的动物，喜欢晒太阳和睡觉",
    "狗是忠诚的伙伴，需要每天遛弯",
    "仓鼠是夜行动物，白天基本在睡觉",
    "Python 是一种编程语言，广泛用于数据科学",
    "JavaScript 用于网页开发",
    "量子力学是一种物理理论，用于描述微小粒子（如电子、原子等）的行为"
]

# 🚀 关键修改：使用内存模式，且不指定 persist_directory
# 这样每次运行都是全新实例，绝对保证使用的是 cosine 度量
vectorstore = Chroma.from_texts(
    texts=knowledge_base,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"}  # ✅ 强制余弦距离
)

# ✅ 修正后的检索节点
def retrieve_node(state: AgentState):
    query = state["query"]
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        return {"documents": [], "similarity_score": 0.0}
    
    # 🚨 核心知识点：Chroma 返回的 score 到底是什么？
    # 当 collection_metadata={"hnsw:space": "cosine"} 时，
    # Chroma 返回的 score 是【余弦距离】 = 1 - 余弦相似度
    # 余弦距离越小，说明越相似（0表示完全一样）
    
    top_distance = results[0][1]
    # 转换为余弦相似度：1 - 距离
    # 使用 max(0, min(1, ...)) 做截断保护，防止浮点数溢出
    top_similarity = max(0, min(1, 1 - top_distance))
    
    # 打印所有召回文档的距离和相似度，方便观察分布
    print(f"--- 检索详情 ---")
    for doc, dist in results:
        sim = max(0, min(1, 1 - dist))
        print(f"  文档: {doc.page_content[:15]}... | 距离: {dist:.4f} | 相似度: {sim:.4f}")
    
    docs = [doc.page_content for doc, _ in results]
    
    return {
        "documents": docs,
        "similarity_score": top_similarity
    }

def route_decision(state: AgentState):
    score = state["similarity_score"]
    if score > 0.7:
        route = "high_confidence"
    elif score > 0.4:
        route = "medium_confidence"
    else:
        route = "low_confidence"
    print(f"📍 路由: {route} (相似度: {score:.3f})")
    return {"route": route}

def high_confidence_answer(state: AgentState):
    doc = state["documents"][0] if state["documents"] else "无相关信息"
    return {"answer": f"✅ 高置信：{doc}"}

def medium_confidence_answer(state: AgentState):
    docs = " | ".join(state["documents"][:2])
    return {"answer": f"⚠️ 中等置信：{docs}"}

def low_confidence_answer(state: AgentState):
    return {"answer": "❌ 抱歉，知识库中没有相关信息"}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("route", route_decision)
workflow.add_node("high_answer", high_confidence_answer)
workflow.add_node("medium_answer", medium_confidence_answer)
workflow.add_node("low_answer", low_confidence_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "route")
workflow.add_conditional_edges(
    "route",
    lambda state: state["route"],
    {
        "high_confidence": "high_answer",
        "medium_confidence": "medium_answer",
        "low_confidence": "low_answer"
    }
)
workflow.add_edge("high_answer", END)
workflow.add_edge("medium_answer", END)
workflow.add_edge("low_answer", END)

app = workflow.compile()

# 测试
if __name__ == "__main__":
    queries = [
        "猫的特点是什么？",
        "编程语言有哪些？",
        "量子力学的原理是什么？"
    ]
    
    for query in queries:
        print(f"\n{'='*60}\n❓ {query}\n{'='*60}")
        result = app.invoke({
            "query": query,
            "documents": [],
            "similarity_score": 0.0,
            "route": "",
            "answer": ""
        })
        print(f"💬 {result['answer']}\n")