"""LLM 对话模型的封装与实例化。

通过 OpenAI 兼容接口调用阿里云 MaaS 部署的 GLM-5.2，
不依赖任何厂商专属 SDK，方便切换底层模型。
"""
from langchain_openai import ChatOpenAI

from config import settings


def create_llm(
    *,
    model: str = settings.LLM_MODEL,
    base_url: str = settings.BASE_URL,
    temperature: float = settings.LLM_TEMPERATURE,
) -> ChatOpenAI:
    """创建对话模型实例。

    通过依赖注入支持灵活替换底层 LLM：调用方可覆盖 model / base_url / temperature。

    Args:
        model: 模型名称，默认为 settings.LLM_MODEL。
        base_url: API 端点，默认为 settings.BASE_URL。
        temperature: 采样温度，默认为 settings.LLM_TEMPERATURE。

    Returns:
        ChatOpenAI 实例。

    Raises:
        EnvironmentError: 如果未检测到 API Key。
    """
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=settings.get_api_key(),
        temperature=temperature,
    )
    print(f"[models] LLM 已创建：{model} @ {base_url}")
    return llm
