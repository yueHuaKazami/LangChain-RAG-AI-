"""MCP Server — 将 RAG 知识检索能力暴露为标准 MCP 工具。

客户端配置示例（MCP settings.json）：
{
  "mcpServers": {
    "rag-knowledge-search": {
      "command": "uv",
      "args": ["run", "ai_qa_agent/src/mcp_server.py"],
      "cwd": "/path/to/RAG_Langchain"
    }
  }
}
"""
import os
import sys

# 路径设置：确保 src/ 内的模块可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from models.embeddings import create_embeddings
from retrieval.document_loader import load_documents_from_dir
from retrieval.vectorstore import build_vectorstore

from mcp.server.fastmcp import FastMCP

# =============================================================================
# MCP Server 初始化
# =============================================================================

mcp = FastMCP("rag-knowledge-search")

# =============================================================================
# 全局资源：启动时加载知识库并构建向量索引，之后所有请求复用
# =============================================================================

_vectorstore = None


def _get_vectorstore():
    """延迟初始化向量库（首次调用时加载，之后复用）。"""
    global _vectorstore
    if _vectorstore is None:
        print("[mcp] Loading knowledge base...", file=sys.stderr)
        try:
            embeddings = create_embeddings()
            documents = load_documents_from_dir()
            _vectorstore = build_vectorstore(documents, embeddings)
            print(f"[mcp] 知识库就绪：{len(documents)} 篇文档", file=sys.stderr)
        except Exception as e:
            print(f"[mcp] 知识库加载失败：{e}", file=sys.stderr)
            raise
    return _vectorstore


# =============================================================================
# MCP 工具定义
# =============================================================================

@mcp.tool()
def ai_knowledge_search(query: str, k: int = 5) -> str:
    """在 AI 知识库中检索与查询相关的文档内容。

    覆盖领域：Transformer、LLM、RAG、机器学习、神经网络、深度学习等。
    当用户提出 AI 专业问题或需要查阅知识库时调用此工具。

    Args:
        query: 自然语言检索查询
        k: 返回的最相关文档块数量，默认 5

    Returns:
        格式化后的检索结果，包含文档来源和内容片段。
    """
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        return "未找到相关文档。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        results.append(f"[{i}] 来源：{source}\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@mcp.tool()
def ai_list_documents() -> str:
    """列出知识库中已加载的所有文档及其概览信息。"""
    vectorstore = _get_vectorstore()
    # 通过 Chroma 的 get 方法获取唯一来源
    try:
        metadatas = vectorstore.get()["metadatas"]
        sources = sorted(set(m.get("source", "未知") for m in metadatas if m))
        if not sources:
            return "知识库为空，请将文档放入 data/ 目录后重启服务。"
        lines = [f"- {s}" for s in sources]
        return f"知识库包含 {len(sources)} 篇文档：\n" + "\n".join(lines)
    except Exception:
        return "无法获取文档列表，请检查知识库是否正常加载。"


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    print("[mcp] RAG Knowledge Search MCP Server starting...", file=sys.stderr)
    mcp.run(transport="stdio")
