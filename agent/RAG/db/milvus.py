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

