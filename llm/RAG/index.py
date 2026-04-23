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
# import numpy as np
# import faiss
# import time

# # ------- 1. 准备数据 -------
# d = 64
# nb = 100000
# nq = 1
# k = 5
# np.random.seed(1234)
# db_vectors = np.random.random((nb, d)).astype('float32')
# query_vectors = np.random.random((nq, d)).astype('float32')

# # ------- 2. 先跑一次 Flat，拿到标准答案 indices_flat -------
# index_flat = faiss.IndexFlatIP(d)
# index_flat.add(db_vectors)
# distances_flat, indices_flat = index_flat.search(query_vectors, k) # 这里生成了 indices_flat！

# # ------- 3. 开始跑 IVF -------
# print("="*50 + " IVF 聚类索引 " + "="*50)
# nlist = 100 
# quantizer = faiss.IndexFlatIP(d)
# index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# # IVF 必须先 train
# index_ivf.train(db_vectors)
# index_ivf.add(db_vectors)

# # nprobe 控制查几个桶
# index_ivf.nprobe = 70 

# start_time = time.time()
# distances_ivf, indices_ivf = index_ivf.search(query_vectors, k)
# ivf_time = (time.time() - start_time) * 1000

# # 现在这行就不会报错了！
# recall_ivf = len(set(indices_flat[0]) & set(indices_ivf[0])) / k

# print(f"耗时: {ivf_time:.2f} 毫秒")
# print(f"Flat标准答案 ID: {indices_flat[0]}")
# print(f"IVF查出来的  ID: {indices_ivf[0]}")
# print(f"⚠️ 召回率: {recall_ivf * 100}%")


# ==========================================
# 3. HNSW 图索引
# ==========================================
import numpy as np
import faiss
import time

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
# print("="*50 + " 1. Flat 暴力检索 " + "="*50)
# index_flat = faiss.IndexFlatIP(d) 
# index_flat.add(db_vectors)

# start_time = time.time()
# distances_flat, indices_flat = index_flat.search(query_vectors, k) # 这里生成了标准答案 indices_flat
# flat_time = (time.time() - start_time) * 1000

# print(f"耗时: {flat_time:.2f} 毫秒")
# print(f"最相似的 Top-{k} ID: {indices_flat[0]}")
# print("👉 这便是【标准答案】，后面都要跟它比！\n")


# ==========================================
# 3. HNSW 图索引
# ==========================================
# print("="*50 + " 2. HNSW 图索引 " + "="*50)
# M = 32
# index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
# index_hnsw.hnsw.efConstruction = 200
# index_hnsw.add(db_vectors)
# index_hnsw.hnsw.efSearch = 64

# start_time = time.time()
# distances_hnsw, indices_hnsw = index_hnsw.search(query_vectors, k)
# hnsw_time = (time.time() - start_time) * 1000

# recall_hnsw = len(set(indices_flat[0]) & set(indices_hnsw[0])) / k

# print(f"耗时: {hnsw_time:.2f} 毫秒")
# print(f"最相似的 Top-{k} ID: {indices_hnsw[0]}")
# print(f"✅ 召回率: {recall_hnsw * 100}%\n")



# ==========================================
# 4. LSH 局部敏感哈希
# ==========================================
# print("="*50 + " 4. LSH 局部敏感哈希 (修正版) " + "="*50)

# # 1. 先算 L2 距离的【真实标准答案】
# index_flat_l2 = faiss.IndexFlatL2(d)  # 👈 改用 L2！
# index_flat_l2.add(db_vectors)
# distances_flat_l2, indices_flat_l2 = index_flat_l2.search(query_vectors, k)
# print(f"L2标准答案 Top-{k} ID: {indices_flat_l2[0]}")

# # 2. 算 LSH (它底层就是 L2)
# nbits = 1  # 只有2个超级大桶
# index_lsh = faiss.IndexLSH(d, nbits)
# index_lsh.add(db_vectors)

# start_time = time.time()
# distances_lsh, indices_lsh = index_lsh.search(query_vectors, k)
# lsh_time = (time.time() - start_time) * 1000

# # 3. 用 L2 的标准答案去对比
# recall_lsh = len(set(indices_flat_l2[0]) & set(indices_lsh[0])) / k

# print(f"耗时: {lsh_time:.2f} 毫秒")
# print(f"最相似的 Top-{k} ID: {indices_lsh[0]}")
# print(f"⚠️ 真实召回率: {recall_lsh * 100}% (现在才是真正的 LSH 水平！)\n")

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