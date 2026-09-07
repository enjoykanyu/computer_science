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