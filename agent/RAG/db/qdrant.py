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