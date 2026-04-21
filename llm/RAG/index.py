import faiss 
import numpy as np  
  
d = 1536  # 向量维度  
N = 10000  # 文档数量  
  
# 构建索引  
index = faiss.IndexFlatL2(d)  # L2 距离（欧氏距离）  
# 或者：faiss.IndexFlatIP(d)  # 内积（余弦相似度需先归一化） 内积=余弦相似度， 只看方向 
  
# 添加向量  
vectors = np.random.rand(N, d).astype('float32')  
index.add(vectors)  
  
# 查询  
query = np.random.rand(1, d).astype('float32') # 查询数量=1个 生成查询向量相当于查询1这个在哪里类似于用户的输入问题的向量     
distances, indices = index.search(query, k=5)  # 返回最近的5个  在索引中搜索最近邻
print(f"最近邻索引: {indices[0]}")  
print(f"距离: {distances[0]}")