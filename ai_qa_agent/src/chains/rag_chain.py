"""RAG Chain 确定性流水线。

与 RAG Agent 不同，RAG Chain 是固定流程的确定性流水线：
  问题 → 检索 → 拼接上下文 → 填充提示词 → LLM 生成 → 输出

使用 LCEL（LangChain Expression Language）构建，天然支持流式输出和异步。
"""
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from config import settings


def _format_docs(docs: list[Document]) -> str:
    """把检索到的文档块拼接成纯文本上下文。

    Args:
        docs: 检索器返回的文档块列表。

    Returns:
        用空行分隔的文档正文。
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    llm: BaseChatModel,
    retriever,
) -> RunnableSerializable:
    """构建 RAG Chain 确定性流水线。

    流程：{"question", "history"} → 检索 → 拼接上下文 → 填充提示词 → LLM → 文本输出

    Args:
        llm: 对话模型实例。
        retriever: 检索器（由 vectorstore.as_retriever() 创建）。

    Returns:
        一个 LCEL 链，调用方式：
        chain.invoke({"question": "问题", "history": [消息列表]})
        返回值为字符串（最终回答）。
    """
    # 系统提示词从外部文件加载，含 {context} 占位符
    system_template = settings.load_prompt("rag_chain_prompt.txt")

    # 提示词模板：系统提示（含检索上下文）+ 历史对话 + 当前问题
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    # LCEL 链：并行检索上下文 + 透传问题和历史，再拼接提示词交给 LLM
    chain = (
        {
            "context": itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("[chains] RAG Chain 确定性流水线构建完成")
    return chain
