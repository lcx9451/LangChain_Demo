"""
LangChain RAG 代理 - 完整示例
"""

# ==================== 1. 导入依赖 ====================
import os
import getpass
import bs4
from typing import Any, List

from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


# ==================== 2. 设置环境变量 ====================
def setup_environment():
    """设置必要的环境变量"""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API Key: ")
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    print("环境变量设置完成")


# ==================== 3. 初始化模型 ====================
def setup_model():
    """初始化聊天模型"""
    model = init_chat_model("gpt-4o")
    print("模型初始化完成")
    return model


# ==================== 4. 初始化嵌入 ====================
def setup_embeddings():
    """初始化嵌入模型"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("嵌入模型初始化完成")
    return embeddings


# ==================== 5. 加载文档 ====================
def load_documents(url: str) -> List[Document]:
    """从网页加载文档"""
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )

    docs = loader.load()
    print(f"加载文档数量：{len(docs)}")
    print(f"总字符数：{len(docs[0].page_content)}")
    return docs


# ==================== 6. 分割文档 ====================
def split_documents(docs: List[Document]) -> List[Document]:
    """分割文档为块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)
    print(f"分割后文档块数：{len(all_splits)}")
    return all_splits


# ==================== 7. 创建向量存储 ====================
def create_vector_store(docs: List[Document], embeddings):
    """创建向量存储并索引文档"""
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(documents=docs)
    print(f"已索引文档数量：{len(document_ids)}")
    return vector_store


# ==================== 8. 定义检索工具 ====================
def create_retrieve_tool(vector_store):
    """创建检索工具"""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """检索信息以帮助回答查询。"""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


# ==================== 9. 创建 RAG 代理 ====================
def create_rag_agent(model, tools, system_prompt: str):
    """创建 RAG 代理"""
    agent = create_agent(model, tools, system_prompt=system_prompt)
    print("RAG 代理创建完成")
    return agent


# ==================== 10. 创建 RAG 链 ====================
def create_rag_chain(model, vector_store):
    """创建 RAG 链（使用中间件）"""
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        system_message = (
            "您是一个有用的助手。在响应中使用以下上下文："
            f"\n\n{docs_content}"
        )
        return system_message

    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    print("RAG 链创建完成")
    return agent


# ==================== 11. 查询代理 ====================
def query_agent(agent, query: str):
    """查询代理"""
    print(f"\n=== 查询 ===")
    print(f"问题：{query}")
    print("\n=== 代理响应 ===")

    # for event in agent.stream(
    #     {"messages": [{"role": "user", "content": query}]},
    #     stream_mode="values",
    # ):
    #     event["messages"][-1].pretty_print()


# ==================== 12. 主函数 ====================
def main():
    """主执行流程"""
    print("=" * 60)
    print("LangChain RAG 代理")
    print("=" * 60)

    # 设置环境
    # setup_environment()

    # 初始化组件
    # model = setup_model()
    # embeddings = setup_embeddings()

    # 加载和处理文档
    # url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    # docs = load_documents(url)
    # all_splits = split_documents(docs)
    # vector_store = create_vector_store(all_splits, embeddings)

    # 创建 RAG 代理
    # retrieve_tool = create_retrieve_tool(vector_store)
    # prompt = "您可以访问从博客文章检索上下文的工具。使用该工具帮助回答用户查询。"
    # rag_agent = create_rag_agent(model, [retrieve_tool], prompt)

    # 创建 RAG 链
    # rag_chain = create_rag_chain(model, vector_store)

    # 查询
    # query = "What is task decomposition?"
    # query_agent(rag_agent, query)

    print("\n完成！")

# 运行
# if __name__ == "__main__":
#     main()