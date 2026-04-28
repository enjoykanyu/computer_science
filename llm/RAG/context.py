from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Sequence

# ════════════════════════════════════════════════════
# 第一段：初始化模型
# ════════════════════════════════════════════════════

# Bi-Encoder：使用 OllamaEmbeddings
bi_encoder = OllamaEmbeddings(model="bge-m3")


# ── 核心：自定义 Cross-Encoder，直接用 transformers ──
class CrossEncoderReranker:
    """
    直接使用 transformers 加载 Cross-Encoder 模型
    无需 sentence_transformers 依赖
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_length = max_length
        print(f"Cross-Encoder 已加载: {model_name} (device={self.device})")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 3,
    ) -> list[Document]:
        """
        对 query-documents 对进行重排序，返回 Top-N
        """
        if not documents:
            return []

        # 构造 (query, doc) 对
        pairs = [[query, doc.page_content] for doc in documents]

        # Tokenize + 推理
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1).cpu().tolist()

        # 单条文档时 squeeze 会变成标量
        if isinstance(scores, (int, float)):
            scores = [scores]

        # 按分数降序排列
        scored_docs = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for doc, score in scored_docs[:top_n]:
            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": score},
                )
            )
        return results


cross_encoder = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")

# ════════════════════════════════════════════════════
# 第二段：构建向量索引
# ════════════════════════════════════════════════════

knowledge_base = [
    "iPhone截图：同时按侧边键和音量上键，松开后截图成功",
    "iPhone截图：在屏幕左下角会出现缩略图，点击可编辑",
    "苹果手机AssistiveTouch截图：设置→辅助功能→触控→AssistiveTouch",
    "iOS截图快捷方式：可在设置中将截图添加到后台轻点手势",
    "苹果手机录屏方法：从控制中心点击录屏按钮",
    "Android手机截图：音量下键+电源键同时按",
    "Mac截图快捷键：Command+Shift+3全屏，Command+Shift+4区域截图",
    "苹果公司2024年营收：服务业务首次超过Mac",
    "苹果M3芯片性能评测：CPU性能提升15%",
    "iPhone相机使用技巧：夜间模式、人像模式详解",
    "App Store上架指南：开发者证书配置步骤",
    "iOS开发入门：Swift语言基础教程",
    "苹果手机屏幕镜像：AirPlay投屏方法",
    "iPhone无法截图的解决方法：重启设备或检查按键",
    "微信朋友圈截图不显示对方头像的原因",
]

print("正在构建向量索引...")
vectorstore = Chroma.from_texts(
    texts=knowledge_base,
    embedding=bi_encoder,
    collection_name="rag_demo",
)
print(f"索引构建完成，共 {len(knowledge_base)} 个文档")

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20},
)

# ════════════════════════════════════════════════════
# 第三段：两阶段检索（手动实现，无需 ContextualCompressionRetriever）
# ════════════════════════════════════════════════════


def two_stage_retrieve(
    query: str,
    stage1_k: int = 20,
    stage2_n: int = 3,
) -> list[Document]:
    """
    Stage 1: Bi-Encoder 粗召回 → Chroma 向量检索 Top-K
    Stage 2: Cross-Encoder 精排 → Top-N
    """
    # ── Stage 1 ──
    base_retriever.search_kwargs["k"] = stage1_k
    stage1_docs = base_retriever.invoke(query)
    print(f"\n[Stage 1] 粗召回 {len(stage1_docs)} 个候选文档")

    # ── Stage 2 ──
    stage2_docs = cross_encoder.rerank(query, stage1_docs, top_n=stage2_n)
    print(f"[Stage 2] 重排序完成，返回 Top-{stage2_n}")

    for i, doc in enumerate(stage2_docs):
        score = doc.metadata.get("relevance_score", 0.0)
        print(f"  Top-{i+1} [score: {score:.4f}] {doc.page_content}")

    return stage2_docs


# ════════════════════════════════════════════════════
# 第四段：构建 Prompt 并调用 LLM (Ollama qwen3)
# ════════════════════════════════════════════════════

# 初始化 Ollama LLM (qwen3 模型)
llm = OllamaLLM(model="qwen3:0.6b")


def rag_generate(query: str, contexts: list[Document]) -> str:
    context_text = "\n".join(
        [f"[文档{i+1}]: {doc.page_content}" for i, doc in enumerate(contexts)]
    )

    prompt = f"""根据以下参考资料回答用户问题。
如果参考资料中没有足够信息，请明确说明"参考资料中没有相关信息"，不要编造答案。

参考资料：
{context_text}

用户问题：{query}

请根据参考资料给出准确、简洁的答案："""

    # 调用 Ollama qwen3 生成答案
    print(f"[正在调用 Ollama qwen3 生成答案...]")
    answer = llm.invoke(prompt)
    return answer


# ════════════════════════════════════════════════════
# 第五段：对比演示
# ════════════════════════════════════════════════════


def demo_comparison(query: str):
    print(f"\n{'='*60}")
    print(f"查询：{query}")
    print("=" * 60)

    # ── 方案A：仅向量检索（无重排序）──
    print("\n【方案A：仅向量检索（无重排序）】")
    base_retriever.search_kwargs["k"] = 3
    results_no_rerank = base_retriever.invoke(query)
    for i, doc in enumerate(results_no_rerank):
        print(f"  Top-{i+1} {doc.page_content}")

    # ── 方案B：向量检索 + Cross-Encoder 重排序 ──
    print("\n【方案B：向量检索 + Cross-Encoder 重排序】")
    results_reranked = two_stage_retrieve(query, stage1_k=20, stage2_n=3)

    # ── 生成答案 ──
    print("\n【RAG 生成结果】")
    answer = rag_generate(query, results_reranked)
    print(answer)


# 运行
demo_comparison("苹果手机截屏方法有哪些")