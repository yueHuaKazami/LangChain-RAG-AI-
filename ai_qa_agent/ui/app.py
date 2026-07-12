"""
AI领域专家问答系统 - Streamlit 交互界面。

运行方式（在项目根目录 D:\TraeCode\RAG_Langchain 下执行）：
    uv run streamlit run ai_qa_agent/ui/app.py

功能：
  - 左侧边栏：知识库管理（上传、查看、删除文档；重建索引），展示 data/ 已有文档
  - 主区域：多轮对话问答，对接真实 LangChain RAG Chain，支持 Markdown 渲染 + 打字机效果
"""
import os
import sys
import time
import io
import streamlit as st

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# =============================================================================
# 路径与环境初始化（必须在导入 src 模块之前完成）
# =============================================================================

# 找到项目根目录（.env 所在位置）
# app.py 位于 ai_qa_agent/ui/app.py，项目根在再往上两层
_UI_DIR = os.path.dirname(os.path.abspath(__file__))           # .../ai_qa_agent/ui
_AGENT_DIR = os.path.dirname(_UI_DIR)                           # .../ai_qa_agent
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)                     # .../RAG_Langchain

# 将 src 目录加入 Python 导入路径
sys.path.insert(0, os.path.join(_AGENT_DIR, "src"))

# 在 settings 模块级 load_dotenv() 执行前，从项目根加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 设置 HuggingFace 离线模式（与 settings.py 保持一致）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import settings
from models.llm import create_llm
from models.embeddings import create_embeddings
from retrieval.document_loader import load_documents_from_dir
from retrieval.vectorstore import build_vectorstore
from chains.rag_chain import build_rag_chain
from agents.rag_agent import build_rag_agent
from tools.knowledge_retriever import build_knowledge_retriever
from chains.rag_chain import build_rag_chain

# 页面配置

st.set_page_config(
    page_title="AI领域专家问答系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 缓存资源：LLM 和 Embeddings 模型（使用 st.cache_resource 避免重复加载）
@st.cache_resource
def get_llm():
    """获取 LLM 实例（全局缓存，仅首次加载）。"""
    return create_llm()


@st.cache_resource
def get_embeddings():
    """获取 Embeddings 实例（全局缓存，仅首次加载）。"""
    return create_embeddings()

# 工具函数：文档解析
def parse_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str | None:
    """解析上传的文档，提取纯文本内容。

    支持格式：PDF、TXT、MD。
    对编码错误做容错处理，解析失败时返回 None 并提示用户。
    """
    file_name = uploaded_file.name.lower()
    try:
        # PDF：使用 pypdf 逐页提取
        if file_name.endswith(".pdf"):
            from pypdf import PdfReader
            raw_bytes = uploaded_file.read()
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        # TXT / MD：纯文本，处理编码容错
        elif file_name.endswith((".txt", ".md")):
            raw_bytes = uploaded_file.read()
            # 优先 UTF-8，失败回退 GBK（Windows 中文环境常见编码）
            for encoding in ("utf-8", "gbk", "latin-1"):
                try:
                    return raw_bytes.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    continue
            return raw_bytes.decode("utf-8", errors="replace")

        else:
            st.warning(f"不支持的文件格式：{uploaded_file.name}")
            return None

    except Exception as e:
        st.error(f"解析文件失败（{uploaded_file.name}）：{e}")
        return None
        st.error(f"解析文件失败（{uploaded_file.name}）：{e}")
        return None


# =============================================================================
# RAG 后端：真实 LangChain RAG Chain 调用
# =============================================================================

def _session_to_lc_messages(messages: list[dict]) -> list:
    """将 session_state 消息列表转换为 LangChain BaseMessage 对象。

    RAG Chain 的 invoke 需要 history: list[BaseMessage]。
    session_state.messages 存储的是 {"role": ..., "content": ...} 字典，需要转换。
    """
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def _rag_chain_query(question: str, history: list[dict]) -> str:
    """RAG Chain 模式：确定性流水线，每次提问强制检索知识库。

    流程：检索 -> 拼接上下文 -> LLM 生成。
    每次提问都强制检索知识库，确保回答基于实际文档内容。
    """
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return "知识库索引尚未构建。请先在左侧边栏点击「重建知识库索引」。"

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})
    chain = build_rag_chain(llm, retriever)

    lc_history = _session_to_lc_messages(history)
    answer = chain.invoke({
        "question": question,
        "history": lc_history,
    })
    return answer


def _rag_agent_query(question: str, history: list[dict]) -> str:
    """RAG Agent 模式：LLM 自主决策是否检索知识库。

    LLM 自主判断问题是否需要检索知识库，更灵活但行为不完全确定。
    适用于综合对话场景，可处理闲聊与专业问题的混合输入。
    """
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return "知识库索引尚未构建。请先在左侧边栏点击「重建知识库索引」。"

    llm = get_llm()
    retriever_tool = build_knowledge_retriever(vectorstore)
    agent = build_rag_agent(llm, retriever_tool)

    # 将 session 历史转为 LangChain 消息，并追加当前用户问题
    lc_messages = _session_to_lc_messages(history)
    lc_messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": lc_messages})
    # Agent 返回 {"messages": [...]}，最后一条 AIMessage 即为最终回答
    all_msgs = result["messages"]
    for msg in reversed(all_msgs):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "无法生成回答，请重试。"


def rag_query(question: str, history: list[dict]) -> str:
    """RAG 问答分发器：根据当前 chat_mode 调用对应的后端模式。"""
    mode = st.session_state.get("chat_mode", "chain")
    if mode == "agent":
        return _rag_agent_query(question, history)
    else:
        return _rag_chain_query(question, history)


# =============================================================================
# 知识库索引重建：真实文档加载 + 向量化 + 存入 Chroma
# =============================================================================

def rebuild_index(doc_names: list[str], doc_contents: dict[str, str]) -> bool:
    """重建知识库向量索引。

    加载 data/ 目录已有文档 + 用户上传文档 -> 切分 -> 向量化 -> 存入 Chroma。
    结果保存在 st.session_state.vectorstore 中。
    """
    all_docs: list[Document] = []

    # 1. 加载 data/ 目录中的已有文档
    data_dir = settings.DATA_DIR
    progress_text = st.empty()
    progress_text.text("正在加载知识库文档...")
    try:
        data_docs = load_documents_from_dir(data_dir)
        all_docs.extend(data_docs)
    except FileNotFoundError:
        st.warning(f"data/ 目录为空或不存在：{data_dir}")

    # 2. 加载用户上传的文档（从 session_state 中读取，构造 Document 对象）
    for name in doc_names:
        content = doc_contents.get(name, "")
        if content.strip():
            all_docs.append(Document(page_content=content, metadata={"source": name}))

    if not all_docs:
        progress_text.text("")
        st.warning("没有可用的文档，索引为空。")
        st.session_state.vectorstore = None
        return True

    progress_text.text(f"共 {len(all_docs)} 篇文档，正在切分并向量化...")

    # 3. 构建向量数据库（内部完成切分 + 向量化 + 存入 Chroma）
    embeddings = get_embeddings()
    st.session_state.vectorstore = build_vectorstore(all_docs, embeddings)

    progress_text.text("")
    return True
    progress_text.text("")
    return True


# =============================================================================
# Session State 初始化
# =============================================================================

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "chain"    # 默认 Chain 模式（"agent" / "chain"）

if "messages" not in st.session_state:
    st.session_state.messages = []          # 聊天记录：[{"role": ..., "content": ...}]

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []         # 用户上传的文档名称列表

if "doc_contents" not in st.session_state:
    st.session_state.doc_contents = {}      # {文件名: 文本内容}

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False  # 是否展开上传文件选择器

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False    # 索引是否已构建

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None     # 向量数据库实例


# =============================================================================
# 侧边栏：知识库管理
# =============================================================================

with st.sidebar:
    # ---- 对话模式选择器 ----
    st.subheader("对话模式")
    mode_option = st.radio(
        "选择问答模式",
        options=["chain", "agent"],
        format_func=lambda x: "🔗 RAG Chain — 确定性检索" if x == "chain" else "🤖 RAG Agent — 智能决策",
        index=0 if st.session_state.chat_mode == "chain" else 1,
        key="sidebar_mode_radio",
    )
    if mode_option != st.session_state.chat_mode:
        st.session_state.chat_mode = mode_option
        st.session_state.messages = []  # 切换模式时清空历史，避免格式不兼容
        st.rerun()

    mode_label = "Agent 智能决策" if st.session_state.chat_mode == "agent" else "🔗 Chain 确定性检索"
    st.caption(f"当前模式：{mode_label}")

    st.divider()

    st.title("📚 知识库管理")

    # ---- data/ 目录已有文档（只读展示） ----
    st.subheader("📁 知识库已有文档")
    data_files: list[str] = []
    if os.path.isdir(settings.DATA_DIR):
        for f in sorted(os.listdir(settings.DATA_DIR)):
            ext_lower = os.path.splitext(f)[1].lower()
            if ext_lower in settings.SUPPORTED_EXTENSIONS:
                data_files.append(f)

    if data_files:
        for f in data_files:
            st.caption(f"📄 {f}")
    else:
        st.caption("（暂无文档）")

    st.divider()

    # ---- 上传文档区域 ----
    st.subheader("📤 上传文档")
    if st.button("上传文档", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader

    if st.session_state.show_uploader:
        uploaded_files = st.file_uploader(
            "选择文档（支持 PDF、TXT、MD）",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files:
            newly_added = False
            for uploaded_file in uploaded_files:
                name = uploaded_file.name
                # 避免重复添加同名文件
                if name in st.session_state.doc_contents:
                    st.warning(f"文件已存在，跳过：{name}")
                    continue
                text = parse_uploaded_file(uploaded_file)
                if text is not None:
                    st.session_state.doc_names.append(name)
                    st.session_state.doc_contents[name] = text
                    newly_added = True
                    st.success(f"已解析：{name}（{len(text)} 字符）")
            if newly_added:
                st.session_state.index_ready = False
                st.info("💡 文档已上传，请点击底部「重建知识库索引」使新文档生效。")

    st.divider()

    # ---- 用户上传文档列表 ----
    st.subheader("📋 已上传文档")
    if not st.session_state.doc_names:
        st.caption("（暂无上传文档）")
    else:
        for i, name in enumerate(st.session_state.doc_names):
            col1, col2 = st.columns([4, 1])
            with col1:
                char_count = len(st.session_state.doc_contents.get(name, ""))
                st.caption(f"{name}  ({char_count} 字符)")
            with col2:
                if st.button("🗑", key=f"del_{i}", help=f"删除 {name}"):
                    del st.session_state.doc_contents[name]
                    st.session_state.doc_names.remove(name)
                    st.session_state.index_ready = False
                    st.info(f"已删除 {name}，请点击底部「重建知识库索引」更新。")
                    st.rerun()

    st.divider()

    # ---- 重建索引按钮 ----
    st.subheader("⚙️ 索引管理")
    index_status = "✅ 索引就绪" if st.session_state.index_ready else "⚠️ 索引待重建"
    st.caption(index_status)

    if st.button("重建知识库索引", use_container_width=True):
        with st.spinner("正在重建知识库索引，请稍候..."):
            success = rebuild_index(
                st.session_state.doc_names,
                st.session_state.doc_contents,
            )
        if success:
            st.session_state.index_ready = True
            st.success("知识库索引重建完成！现在可以开始提问。")
            st.rerun()


# 主区域：聊天界面
# 顶部标题栏
st.title("AI领域专家问答系统")
st.caption("基于 RAG 的知识检索增强问答，覆盖人工智能、机器学习、深度学习等专业领域。")
st.divider()

# 聊天历史渲染
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# 底部输入框
if prompt := st.chat_input("输入你的问题，按 Enter 发送..."):
    # 1. 将用户消息加入历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 在界面中展示用户消息
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. 调用真实 RAG Chain 获取回答，并用打字机效果逐字显示
        with st.chat_message("assistant"):
            # 获取完整回答（排除刚加入的用户消息，只传历史上下文）
            history = st.session_state.messages[:-1]
            full_answer = rag_query(prompt, history)

            # 打字机效果：逐字渲染 Markdown
            placeholder = st.empty()
            displayed = ""
            for char in full_answer:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.015)  # 控制打字速度
            placeholder.markdown(full_answer)

    # 4. 将助手消息保存到历史
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
