"""主入口，负责实例化 Agent/Chain 并启动对话。

支持两种问答模式：
  - agent: RAG 智能体，LLM 自主决定是否检索，更灵活
  - chain: RAG 确定性流水线，每次必检索，延迟低可控性强

运行方式：
  uv run ai_qa_agent/src/main.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings

from langchain_core.messages import HumanMessage, AIMessage

from models.llm import create_llm
from models.embeddings import create_embeddings
from retrieval.document_loader import load_documents_from_dir
from retrieval.vectorstore import build_vectorstore
from tools.knowledge_retriever import build_knowledge_retriever
from memory.chat_memory import ChatMemory
from agents.rag_agent import build_rag_agent
from chains.rag_chain import build_rag_chain


def print_banner() -> None:
    """打印启动横幅。"""
    print("\n" + "=" * 60)
    print("AI 知识问答系统（输入 'quit' 退出）")
    print("=" * 60)


def choose_mode() -> str:
    """让用户选择问答模式。

    Returns:
        "agent" 或 "chain"。
    """
    print("\n请选择问答模式：")
    print("  1. agent  —— RAG 智能体：LLM 自主决定是否检索，更灵活")
    print("  2. chain  —— RAG 流水线：每次必检索，延迟低可控性强")
    while True:
        choice = input("\n请输入 1 或 2: ").strip()
        if choice in ("1", "agent"):
            return "agent"
        if choice in ("2", "chain"):
            return "chain"
        print("输入无效，请重新输入。")


def chat_loop_agent(agent) -> None:
    """RAG Agent 多轮对话循环。

    Agent 模式：agent.invoke 返回完整消息列表（含工具调用过程），
    用它替换历史以保证下一轮看到本轮的工具交互记录。

    Args:
        agent: build_rag_agent 构建的智能体实例。
    """
    print("\n[模式] RAG Agent 已启动\n")
    memory = ChatMemory()

    while True:
        user_input = input("用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 把用户消息加入历史，传入完整 messages 供智能体决策
        memory.add(HumanMessage(content=user_input))
        result = agent.invoke({"messages": memory.messages})

        # result["messages"] 含本轮新增的所有消息（工具调用/返回 + 最终回复）
        # 用它替换历史，保证下一轮能看到本轮的工具交互过程
        memory.replace(result["messages"])

        # 最后一条消息是智能体的最终回复
        reply = memory.messages[-1].content
        print(f"\nAI: {reply}\n")


def chat_loop_chain(chain) -> None:
    """RAG Chain 多轮对话循环。

    Chain 模式：chain.invoke 返回字符串（最终回答），
    需手动包装成 AIMessage 存入历史。
    每次提问必检索，流程固定为：检索 → 拼接上下文 → LLM 生成。

    Args:
        chain: build_rag_chain 构建的 LCEL 链。
    """
    print("\n[模式] RAG Chain 已启动\n")
    memory = ChatMemory()

    while True:
        user_input = input("用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 确定性流水线：每次必检索，LLM 无决策权
        answer = chain.invoke({
            "question": user_input,
            "history": memory.messages,
        })
        memory.add(HumanMessage(content=user_input))
        memory.add(AIMessage(content=answer))

        print(f"\nAI: {answer}\n")


def main() -> None:
    """程序主入口，按阶段顺序串联执行。"""
    mode = choose_mode()

    # 阶段 0：初始化模型（LLM + Embedding）
    print("\n[阶段0] 正在初始化模型...")
    llm = create_llm()
    embeddings = create_embeddings()

    # 阶段 1：加载文档并构建向量数据库
    print("\n[阶段1] 正在构建知识库...")
    documents = load_documents_from_dir()
    vectorstore = build_vectorstore(documents, embeddings)

    # 阶段 2：根据模式构建调度器并启动对话
    print(f"\n[阶段2] 正在构建 {mode.upper()} 调度器...")
    if mode == "agent":
        # Agent 模式：把检索器包装成工具，交给智能体自主调用
        retriever_tool = build_knowledge_retriever(vectorstore)
        agent = build_rag_agent(llm, retriever_tool)
        print_banner()
        chat_loop_agent(agent)
    else:
        # Chain 模式：直接用检索器构建确定性流水线
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVER_K}
        )
        chain = build_rag_chain(llm, retriever)
        print_banner()
        chat_loop_chain(chain)


if __name__ == "__main__":
    main()
