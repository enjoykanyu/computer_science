from neo4j import GraphDatabase
import numpy as np

# ------------------- ① 连接 Neo4j 并创建节点 -------------------
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
# 做什么：通过 Bolt 协议连接图数据库
# 为什么：Bolt 是专用的二进制协议，比 HTTP 性能更好

def create_graph(tx):
    # 清理旧数据
    tx.run("MATCH (n) DETACH DELETE n")
    
    # 创建带有 embedding 的实体节点
    tx.run("""
    CREATE (a:Entity {name: 'FAISS', embedding: [1.0,1.0,1.0,1.0]})
    CREATE (b:Entity {name: 'HNSW', embedding: [1.1,1.1,1.1,1.1]})
    CREATE (c:Entity {name: 'Elasticsearch', embedding: [5.0,5.0,5.0,5.0]})
    CREATE (a)-[:DEPENDS_ON]->(b)  // FAISS 依赖 HNSW 算法
    CREATE (c)-[:INTEGRATES]->(b)  // ES 集成了 HNSW 算法
    """)
    # 做什么：创建节点并建立 DEPENDS_ON 和 INTEGRATES 关系
    # 为什么：这是 GraphRAG 的核心优势，关系蕴含了逻辑推理路径
    # 改变：如果不加关系，退化为纯向量检索，丧失多跳能力

with driver.session() as session:
    session.execute_write(create_graph)

# ------------------- ② 创建向量索引 -------------------
def create_vector_index(tx):
    tx.run("""
    CREATE VECTOR INDEX entity_embedding_index FOR (n:Entity) ON (n.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 4,
        `vector.similarity_function`: 'cosine'
    }}
    """)
    # 做什么：为 Entity 节点的 embedding 属性创建向量索引
    # 为什么：必须创建索引，后续的向量相似度查询才能走索引加速
    # 改变：如果 dimensions 填错，创建会失败；如果不建索引，只能全库扫描计算

with driver.session() as session:
    session.execute_write(create_vector_index)

# ------------------- ③ 执行 Cypher 组合查询 -------------------
query_embedding = [1.05, 1.05, 1.05, 1.05] # 模拟 FAISS 的查询向量

def graph_rag_query(tx, query_vec):
    result = tx.run("""
    // 第一步：通过向量索引找到最相似的起始节点
    CALL db.index.vector.queryNodes('entity_embedding_index', 1, $queryVec)
    YIELD node AS start_node, score
    
    // 第二步：基于起始节点进行图遍历，获取关联上下文
    MATCH path = (start_node)-[r*1..2]-(connected_node)
    RETURN start_node.name AS core_entity, 
           score AS similarity,
           connected_node.name AS context_entity,
           [rel in relationships(path) | type(rel)] AS relations
    """, queryVec=query_vec)
    # 做什么：先执行向量检索拿到 start_node，再执行图 MATCH 拿到多跳关系
    # 为什么：单靠向量找不到“ES 和 FAISS 都依赖 HNSW”这种关联逻辑
    # 改变：如果去掉 CALL 部分，只能做全局图遍历，失去语义聚焦能力
    
    for record in result:
        print(f"核心实体: {record['core_entity']} (相似度: {record['similarity']:.2f})")
        print(f"关联上下文: {record['context_entity']}, 关系路径: {record['relations']}")

with driver.session() as session:
    session.execute_read(graph_rag_query, query_embedding)

driver.close()