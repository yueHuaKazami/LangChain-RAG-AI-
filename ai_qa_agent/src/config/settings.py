"""项目全局配置。

所有 API 密钥、模型参数、路径配置集中在此模块管理。

"""
import os
from dotenv import load_dotenv

# === 环境变量加载===
# 1. 加载 .env 文件中的环境变量
# 2. 设置 HuggingFace 离线模式，用本地缓存的模型
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# === 路径配置 ===
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_qa_agent/src
PROJECT_ROOT = os.path.dirname(_SRC_DIR)  # ai_qa_agent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROMPTS_DIR = os.path.join(_SRC_DIR, "prompts")

# === LLM 配置 ===
BASE_URL = "https://llm-pqh5yejiodhtvajz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "glm-5.2"
LLM_TEMPERATURE = 0

# === Embedding 配置 ===
# 智源发布的中文 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# === 检索配置 ===
CHUNK_SIZE = 500       # 文本切分每块最多字符数
CHUNK_OVERLAP = 50     # 相邻块重叠字符数，防止句子被截断
RETRIEVER_K = 5        # 每次检索返回最相关的文档块数量

# === 支持的文档格式 ===
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc"}


def get_api_key() -> str:
    """从环境变量获取 API Key。

    Returns:
        API Key 字符串。

    Raises:
        EnvironmentError: 如果未检测到 OPENAI_API_KEY 环境变量。
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到 OPENAI_API_KEY 环境变量。请在项目根目录的 .env 文件中设置。"
        )
    return api_key


def load_prompt(filename: str) -> str:
    """从 prompts 目录加载提示词文本文件。

    Args:
        filename: prompts 目录下的文件名，如 "rag_agent_prompt.txt"。

    Returns:
        提示词文本内容。
    """
    file_path = os.path.join(PROMPTS_DIR, filename)
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()
