"""领域本体 Schema —— 为 Chroma 向量库提供概念级元数据过滤。

FUTUREWORK.md 第 11 点主题一：在入库阶段让 LLM 根据本体 Schema 为每个 chunk
打上概念标签（如 Architecture、LLM_Model），作为 metadata 存入 Chroma。
检索阶段通过 where={"concept_type": "Architecture"} 将全局检索降维到概念子空间，
从源头阻断"跨概念语义污染"。

使用方式：
    from retrieval.ontology import AI_ONTOLOGY, label_chunks_with_ontology

    labeled = label_chunks_with_ontology(chunks, llm, ontology=AI_ONTOLOGY)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# AI 领域本体 Schema
AI_ONTOLOGY = OrderedDict({
    "Architecture": {
        "label": "Architecture",
        "description": (
            "神经网络与模型架构设计原理，包括 Transformer 结构、CNN、RNN、"
            "自注意力机制、编码器-解码器等内部结构"
        ),
    },
    "LLM_Model": {
        "label": "LLM_Model",
        "description": (
            "大语言模型及具体模型实例，如 GPT 系列、BERT、LLaMA、GLM，"
            "包括预训练、微调、RLHF、指令调优等"
        ),
    },
    "Training": {
        "label": "Training",
        "description": (
            "模型训练与优化技术，包括反向传播、梯度下降、优化器、"
            "损失函数、正则化、学习率调度等"
        ),
    },
    "RAG": {
        "label": "RAG",
        "description": (
            "检索增强生成技术，包括向量数据库、embedding、文档切分、"
            "检索策略、上下文增强、知识库构建等"
        ),
    },
    "ML_Basics": {
        "label": "ML_Basics",
        "description": (
            "机器学习基础概念，包括监督/无监督学习、分类回归、"
            "评估指标、过拟合、交叉验证、特征工程等"
        ),
    },
    "Neural_Network": {
        "label": "Neural_Network",
        "description": (
            "神经网络基础理论，包括神经元、激活函数、前向/反向传播、"
            "全连接层、感知机、深度神经网络结构等"
        ),
    },
    "Application": {
        "label": "Application",
        "description": (
            "AI 工程应用与实践，包括 LangChain 框架、模型部署、"
            "API 调用、Streamlit、工具集成、实际案例等"
        ),
    },
    "General": {
        "label": "General",
        "description": (
            "通用 AI 概念，不属于以上任何特定类别时使用"
        ),
    },
})

# 从 Ontology 中提取的纯概念标签列表（供 LLM prompt 使用）
_CONCEPT_LABELS = [v["label"] for v in AI_ONTOLOGY.values()]

# 分类 Prompt 构建
def _build_classification_prompt(chunks: list[Document]) -> str:
    """构建批量分类 prompt：将多个 chunk 和概念列表打包为一个 LLM 请求。

    输出格式要求每行一个概念标签，按 chunk 顺序对应，
    方便解析为 `chunks[i].metadata["concept_type"] = label`。
    """
    # 构建概念描述列表
    concepts_desc = "\n".join(
        f"- {v['label']}: {v['description']}"
        for v in AI_ONTOLOGY.values()
    )

    # 构建文档块列表
    chunks_text = "\n".join(
        f"[{i}] {chunk.page_content[:300]}"
        for i, chunk in enumerate(chunks)
    )

    return (
        "你是AI领域文档分类专家。请为以下每个文档块从给定概念列表中选择最匹配的一个。\n\n"
        f"概念列表：\n{concepts_desc}\n\n"
        "文档块（每个以 [序号] 开头）：\n"
        f"{chunks_text}\n\n"
        "对每个文档块，只输出概念名称（如 Architecture），每行一个，按序号顺序："
    )

# LLM 批量分类
def label_chunks_with_ontology(
    chunks: list[Document],
    llm: Optional[BaseChatModel] = None,
    *,
    ontology: OrderedDict = AI_ONTOLOGY,
    batch_size: int = 20,
) -> list[Document]:
    """使用 LLM 为文档块打上本体概念标签，存入 metadata["concept_type"]。

    采用批量分类以减少 LLM 调用次数：每 batch_size 个 chunk 打包为一个请求。
    如果 LLM 未传入或调用失败，降级为 "General" 兜底标签。

    Args:
        chunks: 待标签的文档块列表。
        llm: LLM 实例（可选）。未提供时所有 chunk 标记为 General。
        ontology: 本体 Schema 字典。
        batch_size: 每次 LLM 调用处理的 chunk 数量。

    Returns:
        带有 "concept_type" metadata 的文档块列表（原地修改，同时也返回）。
    """
    valid_labels = [v["label"] for v in ontology.values()]

    if llm is None:
        print("[ontology] LLM 未提供，所有 chunk 标记为 General")
        for chunk in chunks:
            chunk.metadata["concept_type"] = "General"
        return chunks

    total = len(chunks)
    labeled = 0
    print(f"[ontology] 开始 LLM 批量分类，共 {total} 个 chunk，batch_size={batch_size}")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = chunks[batch_start:batch_end]

        try:
            prompt_text = _build_classification_prompt(batch)
            response = llm.invoke([
                SystemMessage(content="你是一个精准的文档分类器，只输出概念名称，每行一个。"),
                HumanMessage(content=prompt_text),
            ])

            # 解析 LLM 输出：每行一个概念标签
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
            ]

            for i, line in enumerate(lines):
                if i >= len(batch):
                    break
                # 清理可能的序号、空格等前缀
                label = line.lstrip("0123456789. []-").strip()
                if label in valid_labels:
                    batch[i].metadata["concept_type"] = label
                else:
                    # 尝试模糊匹配
                    matched = _fuzzy_match_label(label, valid_labels)
                    batch[i].metadata["concept_type"] = matched
                labeled += 1

        except Exception as e:
            print(f"[ontology] LLM 分类失败 (batch {batch_start}-{batch_end}): {e}")
            for chunk in batch:
                if "concept_type" not in chunk.metadata:
                    chunk.metadata["concept_type"] = "General"

    # 兜底：未被打标签的 chunk 标记为 General
    for chunk in chunks:
        if "concept_type" not in chunk.metadata:
            chunk.metadata["concept_type"] = "General"
            labeled += 1

    print(f"[ontology] 分类完成：{labeled}/{total} 个 chunk 已标签")
    return chunks


def _fuzzy_match_label(raw: str, valid_labels: list[str]) -> str:
    """对 LLM 输出的非标准标签做模糊匹配，退回最接近的有效标签。"""
    raw_lower = raw.lower()
    for label in valid_labels:
        if label.lower() in raw_lower or raw_lower in label.lower():
            return label
    return "General"

# 查询意图识别（检索阶段使用）
def identify_query_concept(
    query: str,
    llm: BaseChatModel,
    *,
    ontology: OrderedDict = AI_ONTOLOGY,
) -> str:
    """识别用户查询对应的本体概念，用于检索时的子空间过滤。

    Args:
        query: 用户自然语言查询。
        llm: LLM 实例。
        ontology: 本体 Schema 字典。

    Returns:
        概念标签字符串（如 "Architecture"）。
    """
    concepts_desc = "\n".join(
        f"- {v['label']}: {v['description']}"
        for v in ontology.values()
    )

    prompt = (
        "你是AI领域意图分类专家。请判断以下用户问题最匹配哪个概念类别。\n\n"
        f"概念类别：\n{concepts_desc}\n\n"
        f"用户问题：{query}\n\n"
        "只输出概念名称（如 Architecture），不要解释。"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_label = response.content.strip().lstrip("0123456789. []-").strip()

    valid_labels = [v["label"] for v in ontology.values()]
    if raw_label in valid_labels:
        return raw_label
    return _fuzzy_match_label(raw_label, valid_labels)
