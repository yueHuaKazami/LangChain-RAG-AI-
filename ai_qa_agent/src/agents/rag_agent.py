"""RAG Agent 调度逻辑。

Agent（智能体）= LLM+ 工具+ 系统提示词。
与 RAG Chain 的区别：Agent 由 LLM 自主决定是否检索，更灵活但行为不确定；
RAG Chain 则是每次必检索的确定性流水线。

系统提示词从 prompts/rag_agent_prompt.txt 加载，与代码解耦。
"""
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from config import settings


def build_rag_agent(
    llm: BaseChatModel,
    retriever_tool: BaseTool,
) -> Any:
    """组装 RAG 智能体。

    通过依赖注入接收 LLM 和工具，支持灵活替换底层实现。

    Args:
        llm: 对话模型实例。
        retriever_tool: 检索工具实例（由 tools/knowledge_retriever 构建）。

    Returns:
        一个 LangGraph 编译后的智能体，调用方式：
        agent.invoke({"messages": [...]})
    """
    # 系统提示词从外部文件加载，不在代码中硬编码
    system_prompt = settings.load_prompt("rag_agent_prompt.txt")

    # create_agent 把"模型 + 工具 + 提示词"组装成一个智能体
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
    )
    print("[agents] RAG 智能体构建完成")
    return agent
