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

# chunks = fixed_size_chunking(text, chunk_size=50, overlap=10)
# for i, chunk in enumerate(chunks):
#     print(f"[块{i+1}] {chunk}")
#     print()


# # 递归字符分块
# # 
# from langchain_text_splitters import RecursiveCharacterTextSplitter  
  
# def recursive_chunking(text: str, chunk_size: int = 100, overlap: int = 20):  
#     """  
#     递归字符分块  
#     分隔符优先级：段落 > 换行 > 句号 > 逗号 > 空格 > 单字符  
#     """  
#     splitter = RecursiveCharacterTextSplitter(  
#         chunk_size=chunk_size,  
#         chunk_overlap=overlap,  
#         separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],  
#     )  
#     chunks = splitter.split_text(text)  
#     return chunks  
  
# # 测试  
# text = """  
# 人工智能是计算机科学的一个分支。  
# 它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。  
  
# 该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。  
# 深度学习是其中最重要的技术之一。  
# """  
  
# chunks = recursive_chunking(text, chunk_size=60, overlap=10)  
# for i, chunk in enumerate(chunks):  
#     print(f"[块{i+1}] {chunk}")  
#     print()    


#语义分块修改后
import re  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_ollama import OllamaEmbeddings  
from langchain_huggingface import HuggingFaceEmbeddings
def preprocess_chinese(text: str) -> str:  
    # 关键修复：把中文标点替换成英文句号+空格  
    # SemanticChunker 的分句器只认英文标点 .?!  
    text = re.sub(r'[。！？]', '. ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text  
  
def semantic_chunking(text: str):  
    text = preprocess_chinese(text)  
      
    # 调试：打印预处理后的文本，确认分句正确  
    print("预处理后文本：", text[:100], "...")  
      
    # embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")  
     # 使用 BAAI/bge-m3 中文 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    ) 
    splitter = SemanticChunker(  
        embeddings=embeddings,  
        breakpoint_threshold_type="percentile",  
        breakpoint_threshold_amount=60,  # 必须低于75才能切出3块  
    )  
      
    chunks = splitter.split_text(text)  
    return chunks  
  
text = """  
苹果是一种水果，富含维生素C。苹果有红色、绿色等多种颜色。苹果可以直接吃，也可以榨汁。量子计算机利用量子叠加和纠缠原理工作。它比传统计算机快得多。量子比特可以同时表示0和1。今天天气很好，阳光明媚。适合出去散步。公园里的花都开了。天气相当好，可以出去玩。 人工智能相当强大。苹果很好吃。  
"""  
  
chunks = semantic_chunking(text)  
for i, chunk in enumerate(chunks):  
    print(f"[块{i+1}] {chunk}")  
    print()

# import re
# import numpy as np
# from langchain_ollama import OllamaEmbeddings


# def split_chinese_sentences(text: str) -> list[str]:
#     sentences = re.split(r'(?<=[。！？])', text)
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return sentences


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def semantic_chunking(text: str, similarity_threshold: float = 0.49) -> list[str]:
#     """
#     自定义中文语义分块
#     1. 按中文标点分句
#     2. 计算相邻句子的余弦相似度
#     3. 相似度低于阈值处切分
#     """
#     sentences = split_chinese_sentences(text)
#     if len(sentences) <= 1:
#         return [text]

#     embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

#     sentence_embeddings = np.array(embeddings.embed_documents(sentences))

#     similarities = []
#     for i in range(len(sentence_embeddings) - 1):
#         sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
#         similarities.append(sim)

#     print("句子间相似度:")
#     for i, sim in enumerate(similarities):
#         print(f"  句子{i+1} <-> 句子{i+2}: {sim:.4f}")

#     chunks = []
#     current_chunk = sentences[0]
#     for i, sim in enumerate(similarities):
#         if sim < similarity_threshold:
#             chunks.append(current_chunk)
#             current_chunk = sentences[i + 1]
#         else:
#             current_chunk += sentences[i + 1]
#     chunks.append(current_chunk)

#     return chunks

# text = """  
# 苹果是一种水果，富含维生素C。苹果有红色、绿色等多种颜色。苹果可以直接吃，也可以榨汁。量子计算机利用量子叠加和纠缠原理工作。它比传统计算机快得多。量子比特可以同时表示0和1。今天天气很好，阳光明媚。适合出去散步。公园里的花都开了。天气相当好，可以出去玩。 人工智能相当强大。苹果很好吃。  
# """  
  
# chunks = semantic_chunking(text)  
# for i, chunk in enumerate(chunks):  
#     print(f"[块{i+1}] {chunk}")  
#     print()


# 文档结构分块

# import re  
# from typing import List, Dict  
  
# def structure_chunking(markdown_text: str) -> List[Dict]:  
#     """  
#     文档结构分块：按 Markdown 标题层级切分  
#     适合有明确结构的文档（技术文档、报告、教材等）  
#     """  
#     chunks = []  
#     current_chunk = {"title": "前言", "level": 0, "content": ""}  
      
#     for line in markdown_text.split("\n"):  
#         # 检测标题行（# 一级标题，## 二级标题，### 三级标题）  
#         header_match = re.match(r'^(#{1,3})\s+(.+)', line)  
          
#         if header_match:  
#             # 遇到新标题，保存当前块  
#             if current_chunk["content"].strip():  
#                 chunks.append(current_chunk.copy())  
              
#             # 开始新块  
#             level = len(header_match.group(1))  # 几个#就是几级  
#             title = header_match.group(2)  
#             current_chunk = {  
#                 "title": title,  
#                 "level": level,  
#                 "content": ""  
#             }  
#         else:  
#             # 普通内容行，追加到当前块  
#             current_chunk["content"] += line + "\n"  
      
#     # 别忘了最后一块  
#     if current_chunk["content"].strip():  
#         chunks.append(current_chunk)  
      
#     return chunks  
  
  
# # 测试  
# markdown_doc = """  
# # 人工智能简介  
  
# 人工智能（AI）是计算机科学的重要分支。  
  
# ## 机器学习  
  
# 机器学习是AI的核心技术。  
# 通过大量数据训练模型，让机器自动学习规律。  
  
# ### 深度学习  
  
# 深度学习使用多层神经网络。  
# 在图像识别、语音识别领域表现优异。  
  
# ## 自然语言处理  
  
# NLP让计算机理解人类语言。  
# ChatGPT就是NLP技术的典型应用。  
# """  
  
# chunks = structure_chunking(markdown_doc)  
# for chunk in chunks:  
#     print(f"[{'#'*chunk['level']} {chunk['title']}]")  
#     print(chunk['content'].strip())  
#     print()