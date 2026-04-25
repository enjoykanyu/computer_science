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