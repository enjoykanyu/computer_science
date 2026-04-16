#RAG

###概念
RAG（Retrieval-Augmented Generation）= 先从知识库检索“证据”，再把证据喂给大模型生成答案。
核心目标：减少幻觉、提高答案准确率

### RAG 过程
一次问答流程通常是：
- 文档切分（Chunking）
- 把每个 chunk 编码成向量（Embedding）
- 用户提问 → 也向量化
- 向量检索（召回 topK）
- 把 topK chunk 作为上下文（Context）喂给 LLM
- LLM 生成答案（可附引用）
