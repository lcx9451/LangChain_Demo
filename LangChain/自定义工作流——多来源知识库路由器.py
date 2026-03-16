# 完整示例：多来源知识库路由器
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# 1. 定义状态
class RouterState(TypedDict):
    query: str
    messages: List
    github_results: List
    notion_results: List
    slack_results: List
    final_answer: str


# 2. 定义专业化代理
github_agent = create_agent(
    model="openai:gpt-4.1",
    system_prompt=(
        "You are a GitHub documentation specialist. "
        "Search and analyze GitHub repositories, issues, and documentation."
    ),
)

notion_agent = create_agent(
    model="openai:gpt-4.1",
    system_prompt=(
        "You are a Notion documentation specialist. "
        "Search and analyze Notion pages and databases."
    ),
)

slack_agent = create_agent(
    model="openai:gpt-4.1",
    system_prompt=(
        "You are a Slack conversation specialist. "
        "Search and analyze Slack channels and messages."
    ),
)


# 3. 查询分类函数
def classify_query(query: str) -> List[dict]:
    """分类查询并确定要调用哪些代理。"""
    results = []
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ["code", "repository", "github", "api"]):
        results.append({"query": query, "agent": "github_agent"})

    if any(keyword in query_lower for keyword in ["doc", "note", "notion", "guide"]):
        results.append({"query": query, "agent": "notion_agent"})

    if any(keyword in query_lower for keyword in ["chat", "discussion", "slack", "team"]):
        results.append({"query": query, "agent": "slack_agent"})

    if not results:
        results = [
            {"query": query, "agent": "github_agent"},
            {"query": query, "agent": "notion_agent"},
            {"query": query, "agent": "slack_agent"}
        ]

    return results


# 4. 路由节点
def route_query(state: RouterState):
    """根据查询分类路由到相关代理。"""
    classifications = classify_query(state["query"])
    return [
        Send(c["agent"], {"query": c["query"], "messages": state["messages"]})
        for c in classifications
    ]


# 5. 代理调用节点
def call_github_agent(state: RouterState):
    result = github_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"github_results": [result["messages"][-1].content]}


def call_notion_agent(state: RouterState):
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"notion_results": [result["messages"][-1].content]}


def call_slack_agent(state: RouterState):
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"slack_results": [result["messages"][-1].content]}


# 6. 结果综合节点
def synthesize_results(state: RouterState) -> RouterState:
    """综合所有代理的结果为最终答案。"""
    all_results = []
    sources = []

    if state.get("github_results"):
        all_results.append(f"GitHub: {state['github_results'][0]}")
        sources.append("GitHub")

    if state.get("notion_results"):
        all_results.append(f"Notion: {state['notion_results'][0]}")
        sources.append("Notion")

    if state.get("slack_results"):
        all_results.append(f"Slack: {state['slack_results'][0]}")
        sources.append("Slack")

    if len(all_results) == 1:
        final_answer = all_results[0]
    else:
        final_answer = f"""Based on your inquiry, here's what I found:

        {' | '.join(all_results)}

        Sources consulted: {', '.join(sources)}"""

    return {"final_answer": final_answer}


# 7. 构建图
builder = StateGraph(RouterState)

builder.add_node("github_agent", call_github_agent)
builder.add_node("notion_agent", call_notion_agent)
builder.add_node("slack_agent", call_slack_agent)
builder.add_node("synthesize", synthesize_results)

builder.add_conditional_edges(START, route_query,
                              ["github_agent", "notion_agent", "slack_agent"])

builder.add_edge("github_agent", "synthesize")
builder.add_edge("notion_agent", "synthesize")
builder.add_edge("slack_agent", "synthesize")
builder.add_edge("synthesize", END)

# 带持久化编译
workflow = builder.compile(checkpointer=InMemorySaver())

# 8. 运行示例
config = {"configurable": {"thread_id": "router_001"}}
result = workflow.invoke(
    {"query": "How do I set up the API authentication?", "messages": []},
    config=config
)

print(f"Final Answer: {result['final_answer']}")