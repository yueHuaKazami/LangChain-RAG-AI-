"""Embedding 向量化模型的封装与实例化。

使用本地 HuggingFace 中文 embedding 模型（BAAI/bge-small-zh-v1.5），
替代云端 OpenAIEmbeddings，避免 tiktoken 不兼容非 OpenAI 模型名的问题。
首次运行会下载模型（约 100MB），之后离线可用。
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def create_embeddings(
    *,
    model_name: str = settings.EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """创建向量化模型实例。

    通过依赖注入支持灵活替换 embedding 模型。

    Args:
        model_name: HuggingFace 模型名，默认为 settings.EMBEDDING_MODEL。

    Returns:
        HuggingFaceEmbeddings 实例。
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"[models] Embedding 已创建：{model_name}")
    return embeddings
