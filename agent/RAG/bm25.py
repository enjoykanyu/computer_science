import math  
from collections import defaultdict  
  
# ============================================================  
# Step 1: 文本预处理  
# ============================================================  
  
def tokenize(text: str) -> list[str]:  
    """  
    将文本分词并小写化。  
    ① 这行做了什么：把字符串按空格切分，并统一转小写  
    ② 为什么这样写：BM25 是词袋模型，大小写不影响语义，统一小写避免重复计数  
    ③ 如果改成不 lower()：'Python' 和 'python' 会被视为不同词，召回率下降  
    """  
    return text.lower().split()  
  
# ============================================================  
# Step 2: 构建倒排索引  
# ============================================================  
  
class BM25:  
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):  
        """  
        corpus: 文档列表，每个元素是一段文本  
        k1: 词频饱和参数，控制 TF 的影响上限，推荐 1.2~2.0  
        b:  长度归一化参数，0=不归一化，1=完全归一化，推荐 0.75  
        """  
        self.k1 = k1  
        self.b = b  
        self.corpus = corpus  
  
        # 对每篇文档分词，得到词列表  
        # ① 这行做了什么：将所有文档转为词列表的列表  
        # ② 为什么用列表推导：简洁且高效，避免显式 for 循环  
        self.tokenized_corpus = [tokenize(doc) for doc in corpus]  
  
        # 文档数量 N  
        self.N = len(corpus)  
  
        # 每篇文档的长度（词数）  
        # ① 这行做了什么：计算每篇文档的词数  
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]  
  
        # 平均文档长度 avgdl  
        # ③ 如果 corpus 为空：会触发 ZeroDivisionError，生产中需加保护  
        self.avgdl = sum(self.doc_lengths) / self.N  
  
        # 构建倒排索引：term → {doc_id: tf}  
        # defaultdict(dict) 避免 KeyError，自动初始化  
        self.inverted_index = defaultdict(dict)  
        self._build_index()  
  
        # 计算每个词的 IDF  
        self.idf = {}  
        self._compute_idf()  
  
    def _build_index(self):  
        """  
        构建倒排索引。  
        ① 这行做了什么：遍历每篇文档的每个词，统计词频  
        ② 为什么用 defaultdict(int)：避免判断 key 是否存在，直接 += 1  
        """  
        for doc_id, tokens in enumerate(self.tokenized_corpus):  
            # 统计当前文档中每个词的词频  
            term_freq = defaultdict(int)  
            for token in tokens:  
                term_freq[token] += 1  # 词频 +1  
  
            # 将词频写入倒排索引  
            for term, tf in term_freq.items():  
                self.inverted_index[term][doc_id] = tf  
  
    def _compute_idf(self):  
        """  
        计算每个词的 IDF（BM25 改进版公式）。  
        公式：IDF = log((N - n + 0.5) / (n + 0.5) + 1)  
        ① +0.5 平滑：避免 n=0 时除零，也避免 n=N 时 IDF 为负  
        ② +1 偏移：确保 IDF 始终为正（即使某词出现在所有文档中）  
        ③ 如果用原始 TF-IDF 的 IDF：log(N/n)，当 n=N 时 IDF=0，  
           某些实现会出现负值，BM25 的改进版更稳健  
        """  
        for term, postings in self.inverted_index.items():  
            n = len(postings)  # 包含该词的文档数  
            # BM25 IDF 公式（Robertson & Sparck Jones 改进版）  
            self.idf[term] = math.log(  
                (self.N - n + 0.5) / (n + 0.5) + 1  
            )  
  
    def score(self, query: str, doc_id: int) -> float:  
        """  
        计算查询 query 对文档 doc_id 的 BM25 分数。  
        ① 这行做了什么：对查询中每个词，计算其对该文档的贡献并求和  
        ② 为什么分词查询：BM25 是词袋模型，每个词独立计分后求和  
        """  
        tokens = tokenize(query)  
        score = 0.0  
        doc_len = self.doc_lengths[doc_id]  
  
        for term in tokens:  
            if term not in self.inverted_index:  
                continue  # 词不在语料库中，贡献为 0（OOV 问题）  
  
            # 获取该词在该文档中的词频，不存在则为 0  
            tf = self.inverted_index[term].get(doc_id, 0)  
  
            # BM25 核心公式分子：tf × (k1 + 1)  
            numerator = tf * (self.k1 + 1)  
  
            # BM25 核心公式分母：tf + k1 × (1 - b + b × doc_len/avgdl)  
            # ① 长度归一化项：b × doc_len/avgdl  
            #    当 doc_len > avgdl（长文档），分母变大，分数降低  
            #    当 doc_len < avgdl（短文档），分母变小，分数升高  
            # ② 如果 b=0：长度归一化消失，退化为纯词频饱和  
            denominator = tf + self.k1 * (  
                1 - self.b + self.b * doc_len / self.avgdl  
            )  
  
            # 该词的贡献 = IDF × TF饱和值  
            score += self.idf[term] * (numerator / denominator)  
  
        return score  
  
    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:  
        """  
        返回 Top-K 最相关文档的 (doc_id, score) 列表。  
        ① sorted(..., reverse=True)：按分数从高到低排序  
        ② [:top_k]：只取前 K 个结果  
        """  
        scores = [  
            (doc_id, self.score(query, doc_id))  
            for doc_id in range(self.N)  
        ]  
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]  
  
  
# ============================================================  
# Step 5: 运行示例  
# ============================================================  
  
corpus = [  
    "Python is a great programming language for data science",  
    "Java is widely used in enterprise software development",  
    "Python and machine learning go hand in hand",  
    "Data science requires statistics and programming skills",  
    "Enterprise software often uses Java Spring framework",  
]  
  
bm25 = BM25(corpus, k1=1.5, b=0.75)  
results = bm25.search("Python data science", top_k=3)  
  
for rank, (doc_id, score) in enumerate(results, 1):  
    print(f"Rank {rank}: [Score={score:.4f}] {corpus[doc_id]}")