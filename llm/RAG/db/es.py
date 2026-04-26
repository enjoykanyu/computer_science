from elasticsearch import Elasticsearch
import numpy as np

# ------------------- ① 连接 ES 并创建索引 -------------------
es = Elasticsearch("http://localhost:9200") # 连接本地 ES
# 做什么：实例化 ES 客户端
# 为什么：通过 REST API 与 ES 交互
# 改变：如果开启了安全认证，这里需配置 http_auth 或 api_key

index_name = "rag_hybrid_index"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name) # 清理旧索引

mapping = {
    "mappings": {
        "properties": {
            "text_content": {             # 文本字段
                "type": "text",           # 做什么：定义为全文检索类型
                "analyzer": "standard"    # 为什么：使用标准分词器（IK 需要额外安装）
            },
            "embedding": {                # 向量字段
                "type": "dense_vector",   # 做什么：ES 8.x 使用 dense_vector 类型
                "dims": 4,                # 为什么：必须与后续灌入的向量维度严格一致
                "index": True,            # 做什么：启用向量索引（默认使用 HNSW）
                "similarity": "l2_norm",  # 为什么：使用 L2 距离计算相似度
                "index_options": {        # ✅ 显式配置 HNSW 参数
                    "type": "hnsw",       # 索引类型：HNSW
                    "m": 16,              # 每层最大连接数（默认16，越大越准越慢）
                    "ef_construction": 100 # 构建时搜索宽度（默认100）
                }
            }
        }
    }
    # 注意：ES 8.12 不需要 index.knn 设置，KNN 功能默认内置
}
es.indices.create(index=index_name, body=mapping)

# ------------------- ② 灌入模拟文档 -------------------
docs = [
    {"text_content": "FAISS 是一个向量检索库", "embedding": [1.0, 1.0, 1.0, 1.0]},
    {"text_content": "Elasticsearch 支持全文搜索", "embedding": [2.0, 2.0, 2.0, 2.0]},
    {"text_content": "大模型经常会发生幻觉", "embedding": [3.0, 3.0, 3.0, 3.0]}
]

for i, doc in enumerate(docs):
    es.index(index=index_name, id=i, document=doc)
    # 做什么：将文档以指定 ID 写入 ES
    # 为什么：固定 ID 便于测试，否则 ES 自动生成随机 ID
    # 改变：如果不指定 ID，相同文档重复 index 会生成多个副本

es.indices.refresh(index=index_name) # 强制刷新，确保数据可被立刻搜索
# 做什么：让刚写入的数据从内存 Buffer 刷入 Segment 变为可搜索状态
# 为什么：ES 默认 1 秒刷新一次，测试时需立刻可见
# 改变：生产环境中频繁 refresh 会严重影响写入性能

# ------------------- ③ 执行混合检索 -------------------
query_embedding = [1.1, 1.1, 1.1, 1.1]
query_text = "向量"

# ES 8.12 KNN 查询语法：使用顶层 knn 参数，不是 bool.should
# 参考：https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html

# 方式1：纯 KNN 搜索（使用 knn 顶层参数）
try:
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 2,
            "num_candidates": 10
        },
        size=2
    )
    print("✅ KNN 检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")
except Exception as e:
    print(f"❌ KNN 查询失败: {e}")

# 方式2：混合搜索（KNN + 文本过滤）
print("\n" + "="*50)
print("尝试混合搜索（KNN + 文本过滤）...")
try:
    # ES 8.12 支持在 knn 查询中添加 filter
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 2,
            "num_candidates": 10,
            "filter": {
                "match": {"text_content": query_text}
            }
        },
        size=2
    )
    print("✅ 混合检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")
except Exception as e:
    print(f"❌ KNN+Filter 查询失败: {e}")


# 方式3：手动实现混合搜索（免费版可用）
print("\n" + "="*50)
print("方式3：手动实现混合搜索（加权合并得分）...")
try:
    # 分别获取向量搜索结果和文本搜索结果
    knn_response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 5,
            "num_candidates": 10
        },
        size=5
    )
    
    text_response = es.search(
        index=index_name,
        query={"match": {"text_content": query_text}},
        size=5
    )
    
    # 手动合并得分（加权平均）
    from collections import defaultdict
    
    scores = defaultdict(lambda: {"knn": 0, "text": 0, "doc": None})
    
    # 归一化 KNN 得分（转为 0-1 范围）
    knn_hits = knn_response['hits']['hits']
    if knn_hits:
        max_knn = max(hit['_score'] for hit in knn_hits)
        min_knn = min(hit['_score'] for hit in knn_hits)
        knn_range = max_knn - min_knn if max_knn != min_knn else 1
        
        for hit in knn_hits:
            doc_id = hit['_id']
            normalized_score = (hit['_score'] - min_knn) / knn_range
            scores[doc_id]["knn"] = normalized_score
            scores[doc_id]["doc"] = hit['_source']
    
    # 归一化文本得分
    text_hits = text_response['hits']['hits']
    if text_hits:
        max_text = max(hit['_score'] for hit in text_hits)
        min_text = min(hit['_score'] for hit in text_hits)
        text_range = max_text - min_text if max_text != min_text else 1
        
        for hit in text_hits:
            doc_id = hit['_id']
            normalized_score = (hit['_score'] - min_text) / text_range
            scores[doc_id]["text"] = normalized_score
            if scores[doc_id]["doc"] is None:
                scores[doc_id]["doc"] = hit['_source']
    
    # 加权合并 (alpha 控制权重: 0=纯文本, 1=纯向量)
    alpha = 0.5  # 向量权重
    beta = 0.5   # 文本权重
    
    final_scores = []
    for doc_id, data in scores.items():
        combined_score = alpha * data["knn"] + beta * data["text"]
        final_scores.append((doc_id, combined_score, data["doc"]))
    
    # 排序取 Top-2
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✅ 手动混合检索结果 (向量权重={alpha}, 文本权重={beta}):")
    for doc_id, score, doc in final_scores[:2]:
        print(f"  ID: {doc_id}, 合并得分: {score:.4f}, Text: {doc['text_content']}")
        print(f"      (向量得分: {scores[doc_id]['knn']:.4f}, 文本得分: {scores[doc_id]['text']:.4f})")
        
except Exception as e:
    print(f"❌ 手动混合查询失败: {e}")
    print("\n尝试仅文本搜索...")
    # 降级为纯文本搜索
    text_query = {"match": {"text_content": query_text}}
    response = es.search(index=index_name, query=text_query, size=2)
    print("✅ 文本检索结果:")
    for hit in response['hits']['hits']:
        print(f"  ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text_content']}")

# 查看索引配置
print("\n" + "="*50)
print("【索引配置详情】")

# 查看 settings 中的索引参数
settings = es.indices.get_settings(index=index_name)
print("\n索引设置 (Settings):")
print(f"  分片数: {settings[index_name]['settings']['index']['number_of_shards']}")
print(f"  副本数: {settings[index_name]['settings']['index']['number_of_replicas']}")

# 查看 mapping 详情
mapping = es.indices.get_mapping(index=index_name)
embedding_config = mapping[index_name]['mappings']['properties']['embedding']
print("\n向量字段配置 (Mapping):")
print(f"  类型: {embedding_config['type']}")
print(f"  维度: {embedding_config['dims']}")
print(f"  索引: {embedding_config['index']}")
print(f"  相似度: {embedding_config['similarity']}")

# 检查是否有 index_options（自定义 HNSW 参数）
if 'index_options' in embedding_config:
    print(f"  索引选项: {embedding_config['index_options']}")
else:
    print("  索引选项: 使用默认值 (HNSW, m=16, ef_construction=100)")

# 验证 HNSW 是否生效 - 查看段信息
print("\n" + "="*50)
print("【验证 HNSW 索引】")
segments = es.indices.segments(index=index_name)
      