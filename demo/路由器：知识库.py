"""
多源知识路由器示例
此示例演示多代理系统的路由器模式。
路由器对查询进行分类，并行路由到专业化代理，并将结果综合为组合响应。
"""

# ==================== 1. 导入依赖 ====================
import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field


# ==================== 2. 定义状态 ====================
class AgentInput(TypedDict):
    """每个子代理的简单输入状态。"""
    query: str


class AgentOutput(TypedDict):
    """每个子代理的输出。"""
    source: str
    result: str


class Classification(TypedDict):
    """单个路由决策：调用哪个代理以及使用什么查询。"""
    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    """路由器主状态。"""
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


# ==================== 3. 定义分类器输出模式 ====================
class ClassificationResult(BaseModel):
    """将用户查询分类为代理特定子问题的结果。"""
    classifications: list[Classification] = Field(
        description="要调用的代理列表及其定向子问题"
    )


# ==================== 4. 定义工具 ====================
@tool
def search_code(query: str, repo: str = "main") -> str:
    """在 GitHub 仓库中搜索代码。"""
    return f"Found code matching '{query}' in {repo}: authentication middleware in src/auth.py"


@tool
def search_issues(query: str) -> str:
    """搜索 GitHub 问题和拉取请求。"""
    return f"Found 3 issues matching '{query}': #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool
def search_prs(query: str) -> str:
    """搜索拉取请求获取实现细节。"""
    return f"PR #156 added JWT authentication, PR #178 updated OAuth scopes"


@tool
def search_notion(query: str) -> str:
    """在 Notion 工作区搜索文档。"""
    return f"Found documentation: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool
def get_page(page_id: str) -> str:
    """按 ID 获取特定 Notion 页面。"""
    return f"Page content: Step-by-step authentication setup instructions"


@tool
def search_slack(query: str) -> str:
    """搜索 Slack 消息和线程。"""
    return f"Found discussion in #engineering: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool
def get_thread(thread_id: str) -> str:
    """获取特定 Slack 线程。"""
    return f"Thread discusses best practices for API key rotation"


# ==================== 5. 初始化模型和代理 ====================
model = init_chat_model("openai:gpt-4.1")
router_llm = init_chat_model("openai:gpt-4.1-mini")

github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details by searching "
        "repositories, issues, and pull requests."
    ),
)

notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "processes, policies, and team documentation by searching "
        "the organization's Notion workspace."
    ),
)

slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant threads and discussions where team members have "
        "shared knowledge and solutions."
    ),
)


# ==================== 6. 定义工作流节点 ====================
def classify_query(state: RouterState) -> dict:
    """分类查询并确定调用哪些代理。"""
    structured_llm = router_llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """分析此查询并确定要咨询哪些知识库。
            对于每个相关源，生成针对该源优化的定向子问题。

            可用源：
            - github: 代码、API 参考、实现细节、问题、拉取请求
            - notion: 内部文档、流程、政策、团队维基
            - slack: 团队讨论、非正式知识共享、最近对话

            仅返回与查询相关的源。
            """
        },
        {"role": "user", "content": state["query"]}
    ])
    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类分发到代理。"""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]


def query_github(state: AgentInput) -> dict:
    """查询 GitHub 代理。"""
    result = github_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}


def query_notion(state: AgentInput) -> dict:
    """查询 Notion 代理。"""
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput) -> dict:
    """查询 Slack 代理。"""
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}


def synthesize_results(state: RouterState) -> dict:
    """将所有代理的结果组合为连贯的答案。"""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    formatted = [
        f"**From {r['source'].title()}:** \n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""综合这些搜索结果以回答原始问题：{state['query']}
            - 组合来自多个源的信息而不冗余
            - 突出最相关和可操作的信息
            - 注意源之间的任何差异
            - 保持响应简洁且组织良好
            """
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}


# ==================== 7. 构建工作流 ====================
workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)

# ==================== 8. 运行路由器 ====================
if __name__ == "__main__":
    result = workflow.invoke({"query": "How do I authenticate API requests?"})
    print("Original query:", result["query"])
    print("\nClassifications:")
    for c in result["classifications"]:
        print(f"   {c['source']}: {c['query']}")
    print("\n" + "=" * 60 + "\n")
    print("Final Answer:")
    print(result["final_answer"])