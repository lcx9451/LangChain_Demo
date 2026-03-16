# 完整示例：多领域创意助手
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import List


# 1. 定义状态
class CreativeAssistantState(AgentState):
    loaded_skills: List[str] = []
    current_project: str = ""
    content_history: List[str] = []


# 2. 定义创意技能
CREATIVE_SKILLS = {
    "blog_writer": {
        "prompt": """You are a blog writing specialist.

        Expertise:
        - SEO-optimized content
        - Engaging headlines
        - Clear structure with headings
        - Call-to-action integration

        Format:
        - Introduction (hook + thesis)
        - Body paragraphs (3-5 sections)
        - Conclusion (summary + CTA)

        Tone: Professional yet conversational"""
    },
    "social_media": {
        "prompt": """You are a social media content specialist.

        Platforms:
        - Twitter/X: Short, punchy, hashtags
        - LinkedIn: Professional, longer form
        - Instagram: Visual-focused, captions

        Best practices:
        - Hook in first line
        - Use emojis appropriately
        - Include relevant hashtags
        - Engage with questions"""
    },
    "email_marketing": {
        "prompt": """You are an email marketing specialist.

        Email types:
        - Welcome series
        - Promotional campaigns
        - Newsletter updates
        - Re-engagement sequences

        Best practices:
        - Compelling subject lines
        - Clear value proposition
        - Single clear CTA
        - Mobile-friendly formatting"""
    },
    "technical_writer": {
        "prompt": """You are a technical writing specialist.

        Document types:
        - API documentation
        - User guides
        - Technical tutorials
        - Release notes

        Best practices:
        - Clear, concise language
        - Code examples
        - Step-by-step instructions
        - Screenshots when helpful"""
    }
}


# 3. 技能加载工具
@tool
def load_creative_skill(
        skill_name: str,
        runtime
) -> str:
    """Load a creative writing skill.

    Available skills:
    - blog_writer: Blog post creation
    - social_media: Social media content
    - email_marketing: Email campaigns
    - technical_writer: Technical documentation

    Returns the skill's prompt and expertise.
    """
    if skill_name not in CREATIVE_SKILLS:
        available = ", ".join(CREATIVE_SKILLS.keys())
        return f"Skill '{skill_name}' not found. Available: {available}"

    # 跟踪加载的技能
    current_skills = runtime.state.get("loaded_skills", [])
    if skill_name not in current_skills:
        current_skills.append(skill_name)

    return CREATIVE_SKILLS[skill_name]["prompt"]


# 4. 创建创意助手代理
creative_assistant = create_agent(
    model="gpt-4.1",
    tools=[load_creative_skill],
    state_schema=CreativeAssistantState,
    system_prompt=(
        "You are a creative writing assistant with on-demand skills.\n"
        "\n"
        "Available skills:\n"
        "- blog_writer: Blog posts and articles\n"
        "- social_media: Social media content\n"
        "- email_marketing: Email campaigns\n"
        "- technical_writer: Technical documentation\n"
        "\n"
        "Workflow:\n"
        "1. Understand the content type needed\n"
        "2. Load the appropriate skill\n"
        "3. Create content following the skill's guidelines\n"
        "\n"
        "Always ask clarifying questions about audience and goals."
    ),
    checkpointer=InMemorySaver()
)

# 5. 使用示例
config = {"configurable": {"thread_id": "creative_001"}}

result = creative_assistant.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I need to write a blog post about AI trends in 2024"
            }
        ]
    },
    config=config
)

for msg in result["messages"]:
    msg.pretty_print()