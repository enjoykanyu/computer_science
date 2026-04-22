# 暴力flat索引
# import faiss 
# import numpy as np  
  
# d = 1536  # 向量维度  
# N = 10000  # 文档数量  
  
# # 构建索引  
# index = faiss.IndexFlatL2(d)  # L2 距离（欧氏距离）  
# # 或者：faiss.IndexFlatIP(d)  # 内积（余弦相似度需先归一化） 内积=余弦相似度， 只看方向 
  
# # 添加向量  
# vectors = np.random.rand(N, d).astype('float32')  
# index.add(vectors)  
  
# # 查询  
# query = np.random.rand(1, d).astype('float32') # 查询数量=1个 生成查询向量相当于查询1这个在哪里类似于用户的输入问题的向量     
# distances, indices = index.search(query, k=5)  # 返回最近的5个  在索引中搜索最近邻
# print(f"最近邻索引: {indices[0]}")  
# print(f"距离: {distances[0]}")
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
index_ivf.nprobe = 70 

start_time = time.time()
distances_ivf, indices_ivf = index_ivf.search(query_vectors, k)
ivf_time = (time.time() - start_time) * 1000

# 现在这行就不会报错了！
recall_ivf = len(set(indices_flat[0]) & set(indices_ivf[0])) / k

print(f"耗时: {ivf_time:.2f} 毫秒")
print(f"Flat标准答案 ID: {indices_flat[0]}")
print(f"IVF查出来的  ID: {indices_ivf[0]}")
print(f"⚠️ 召回率: {recall_ivf * 100}%")