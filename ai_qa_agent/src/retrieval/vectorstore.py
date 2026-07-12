"""向量库构建。

将原始文档切分为文本块，向量化后存入 Chroma 向量数据库。
Chroma 是嵌入式向量数据库，无需单独部署服务。

切分策略（混合切分，解决小主题被大块稀释导致检索不到的问题）：
  - Markdown 文件：先用 MarkdownHeaderTextSplitter 按标题切分，让每个 ##
    / ### 主题成为独立块（如"卷积神经网络（CNN）""支持向量机（SVM）"自成一块），
    再对过长的块用 RecursiveCharacterTextSplitter 二次切分
  - 其他格式（txt/pdf/docx/doc）：直接用 RecursiveCharacterTextSplitter 按字符切分

标题拼接：MarkdownHeaderTextSplitter 会把标题移到 metadata，导致 chunk 内容
缺少主题关键词（如 SVM 块只剩"寻找最优分离超平面..."正文，检索"支持向量机"时
匹配不到）。因此切分后将标题以 "h1 / h2 / h3" 格式拼接到 chunk 内容前，既保留
主题切分的好处，又增强关键词检索匹配。
"""
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import settings

# Markdown 按标题层级切分：每个 ## / ### 主题成为独立块
_MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _split_single_document(
    doc: Document,
    md_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """对单个文档按格式选择切分策略。

    Args:
        doc: 原始文档。
        md_splitter: Markdown 标题切分器。
        text_splitter: 字符递归切分器。

    Returns:
        切分后的文档块列表。
    """
    source = doc.metadata.get("source", "")
    ext = os.path.splitext(source)[1].lower()

    if ext == ".md":
        # Markdown：按标题切分，保留主题完整性
        md_chunks = md_splitter.split_text(doc.page_content)

        # 将标题拼接到内容前，增强关键词匹配
        for chunk in md_chunks:
            chunk.metadata["source"] = source
            header_parts = [
                chunk.metadata.get(key, "")
                for key in ("h1", "h2", "h3")
            ]
            header_parts = [h for h in header_parts if h]
            if header_parts:
                chunk.page_content = " / ".join(header_parts) + "\n" + chunk.page_content
        # 对过长的标题块二次切分
        chunks = text_splitter.split_documents(md_chunks)
        return chunks
    else:
        # 其他格式：直接按字符切分
        return text_splitter.split_documents([doc])


def build_vectorstore(
    documents: list[Document],
    embeddings: Embeddings,
    *,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> Chroma:
    """将文档切分、向量化并构建 Chroma 向量数据库。

    流程：原始文档 → 按格式选择切分策略 → 每块转向量 → 存入向量数据库。

    Args:
        documents: 待入库的文档列表。
        embeddings: 向量化模型实例。
        chunk_size: 每块最大字符数（用于二次切分和非 Markdown 文件）。
        chunk_overlap: 相邻块重叠字符数，防止句子被截断。

    Returns:
        填充好文档向量的 Chroma 向量数据库实例。
    """
    md_splitter = MarkdownHeaderTextSplitter(_MARKDOWN_HEADERS)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 逐个文档按格式切分后汇总
    all_chunks: list[Document] = []
    for doc in documents:
        chunks = _split_single_document(doc, md_splitter, text_splitter)
        all_chunks.extend(chunks)
    print(f"[vectorstore] 文档已切分为 {len(all_chunks)} 个块")

    # 向量化存入 Chroma：自动用 embeddings 把每个 chunk 转成向量，
    # 把"文本块 + 向量 + 元数据"一起存入数据库
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
    )
    print(f"[vectorstore] 向量数据库构建完成，共 {len(all_chunks)} 条向量")
    return vectorstore
