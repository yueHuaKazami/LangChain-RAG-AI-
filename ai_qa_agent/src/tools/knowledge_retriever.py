"""知识库检索工具。

把向量数据库包装成 Agent 可调用的工具。
工具的 name 和 description 会被传给 LLM，LLM 据此判断
"什么情况下该调用这个工具"，因此描述要清晰说明工具用途。
"""
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool
from langchain_core.tools.retriever import create_retriever_tool

from config import settings

def build_knowledge_retriever(
    vectorstore: Chroma,
    *,
    k: int = settings.RETRIEVER_K,
    tool_name: str = "ai_knowledge_search",
    tool_description: str = (
        "在人工智能知识库中检索相关资料。"
        "当用户询问 AI/机器学习/深度学习/大模型等专业问题时使用此工具。"
    ),
) -> BaseTool:
    """把向量数据库包装成智能体可调用的检索工具。

    Args:
        vectorstore: 已构建好的向量数据库。
        k: 每次检索返回最相关的文档块数量。
        tool_name: 工具名称，会传给 LLM，需唯一且具描述性。
        tool_description: 工具描述，会传给 LLM，决定 LLM 何时调用此工具。

    Returns:
        一个 StructuredTool，可传给 create_agent。
    """
    # as_retriever 把向量库转成检索器，k 表示每次返回最相关的 k 个文档块
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # create_retriever_tool 把检索器包装成"工具"
    retriever_tool = create_retriever_tool(
        retriever,
        name=tool_name,
        description=tool_description,
    )
    print(f"[tools] 检索工具已创建：{tool_name}（k={k}）")
    return retriever_tool
