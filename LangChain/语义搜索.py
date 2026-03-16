"""
LangChain 语义搜索引擎 - 完整示例
"""

# ==================== 1. 导入依赖 ====================
import os
import getpass
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


# ==================== 2. 设置环境变量 ====================
def setup_environment():
    """设置必要的环境变量"""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API Key: ")
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("环境变量设置完成")


# ==================== 3. 加载文档 ====================
def load_documents(file_path: str) -> List[Document]:
    """加载 PDF 文档"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"加载文档数量：{len(docs)}")
    return docs


# ==================== 4. 分割文档 ====================
def split_documents(docs: List[Document]) -> List[Document]:
    """分割文档为块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"分割后文档块数：{len(all_splits)}")
    return all_splits


# ==================== 5. 创建嵌入模型 ====================
def create_embeddings():
    """创建嵌入模型"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("嵌入模型创建完成")
    return embeddings


# ==================== 6. 创建向量存储 ====================
def create_vector_store(docs: List[Document], embeddings):
    """创建向量存储并索引文档"""
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=docs)
    print(f"已索引文档数量：{len(ids)}")
    return vector_store


# ==================== 7. 创建检索器 ====================
def create_retriever(vector_store, k: int = 3):
    """创建检索器"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print(f"检索器创建完成 (k={k})")
    return retriever


# ==================== 8. 自定义检索器 ====================
@chain
def custom_retriever(query: str, vector_store) -> List[Document]:
    """自定义检索器"""
    return vector_store.similarity_search(query, k=1)


# ==================== 9. 查询文档 ====================
def query_documents(retriever, query: str):
    """查询文档"""
    results = retriever.invoke(query)
    print(f"\n=== 查询结果 ===")
    print(f"查询：{query}")
    for i, doc in enumerate(results):
        print(f"\n结果 {i + 1}:")
        print(f"内容：{doc.page_content[:200]}...")
        print(f"元数据：{doc.metadata}")
    return results


# ==================== 10. 主函数 ====================
def main():
    """主执行流程"""
    print("=" * 60)
    print("LangChain 语义搜索引擎")
    print("=" * 60)

    # 设置环境
    # setup_environment()

    # 加载文档
    # docs = load_documents("../example_data/nke-10k-2023.pdf")

    # 分割文档
    # all_splits = split_documents(docs)

    # 创建嵌入模型
    # embeddings = create_embeddings()

    # 创建向量存储
    # vector_store = create_vector_store(all_splits, embeddings)

    # 创建检索器
    # retriever = create_retriever(vector_store, k=3)

    # 查询文档
    # query = "How many distribution centers does Nike have in the US?"
    # results = query_documents(retriever, query)

    print("\n完成！")

# 运行
# if __name__ == "__main__":
#     main()