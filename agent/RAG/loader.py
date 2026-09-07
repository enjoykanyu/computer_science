# ============================================================  
# 1. PDF 加载（文字层 PDF）  
# ============================================================  
# from langchain_community.document_loaders import PyPDFLoader  
  
# loader = PyPDFLoader("report.pdf")  
# docs = loader.load()  
# # print(f"PDF 共 {len(docs)} 页")  
# # print(docs[0].page_content[:200])  
# # print(docs[0].metadata)  # {'source': 'report.pdf', 'page': 0}  
# # 惰性加载（大文件推荐）  
# for doc in loader.lazy_load():  
#     print(doc.metadata["page"], doc.page_content[:50])  

# # ============================================================  
# # 2. Word 文档  
# # ============================================================  
# from langchain_community.document_loaders import Docx2txtLoader  
  
# loader = Docx2txtLoader("contract.docx")  
# docs = loader.load()  # 整个文档作为一个 Document  

# ============================================================  
# 5. 网页  
# ============================================================  
# from langchain_community.document_loaders import WebBaseLoader
# import bs4
# # 设置请求头，模拟真实浏览器访问
# headers = {
#     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
# }

# # 示例1：爬取有文章内容的页面（如技术博客）
# loader = WebBaseLoader(
#     web_paths=["https://docs.python.org/3/tutorial/index.html"],  # Python 官方文档
#     header_template=headers,
#     bs_kwargs={"parse_only": bs4.SoupStrainer("div")},  # 只解析 <div> 标签
# )

# docs = loader.load()  
# print(docs[0].page_content[:200])  # 打印文章内容

# ============================================================  
# 6. Markdown（保留标题层级）  
# ============================================================  
# from langchain_text_splitters import MarkdownHeaderTextSplitter  
  
# md_text = """  
# # 第一章  
# ## 1.1 背景  
# 这是背景内容。  
# ## 1.2 目标  
# 这是目标内容。  
# # 第二章  
# 正文内容。  
# """  
  
# splitter = MarkdownHeaderTextSplitter(  
#     headers_to_split_on=[  
#         ("#", "H1"),  
#         ("##", "H2"),  
#         ("###", "H3"),  
#     ]  
# )  
# docs = splitter.split_text(md_text)  
# print(docs)
# # 每个 Document 的 metadata 包含 {"H1": "第一章", "H2": "1.1 背景"}  
  
# ============================================================  
# 7. 代码文件  
# ============================================================  
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import TextLoader  
  
loader = TextLoader("chunk.py")  
docs = loader.load()  
  
splitter = RecursiveCharacterTextSplitter.from_language(  
    language=Language.PYTHON,  
    chunk_size=500,  
    chunk_overlap=50,  
)  
chunks = splitter.split_documents(docs)  
print(chunks)
# 每个 Document 的 metadata 包含 {"source": "chunk.py"}
  