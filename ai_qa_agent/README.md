# AI 知识问答系统

基于 LangChain 框架的 RAG 检索增强生成问答系统。支持 **Agent**（LLM 自主决策检索）与 **Chain**（确定性检索管线）两种模式。通过 OpenAI 兼容接口调用 GLM-5.2，本地运行 BAAI/bge-small-zh-v1.5 中文 embedding。提供 Streamlit Web 界面与 CLI 终端两种交互方式。

## 功能特性


| 类别       | 说明                                                           |
| ---------- | -------------------------------------------------------------- |
| 双模式问答 | `agent` — LLM 自主决定是否检索；`chain` — 每次提问必检索     |
| 多格式文档 | `.md` `.txt` `.pdf` `.docx` `.doc`                             |
| 混合切分   | Markdown 按标题切分 + 过长块二次递归切分；其他格式直接递归切分 |
| 本地向量化 | BAAI/bge-small-zh-v1.5，支持离线运行                           |
| Web 界面   | Streamlit，支持打字机效果、文档上传与索引管理                  |
| CLI 终端   | 命令行对话，适合快速调试                                       |
| 依赖注入   | LLM、Embeddings、工具均可通过工厂函数灵活替换                  |

## 技术架构

### 技术栈


| 层级       | 选型                                           |
| ---------- | ---------------------------------------------- |
| LLM        | GLM-5.2 (阿里云 MaaS, OpenAI 兼容接口)         |
| Embedding  | BAAI/bge-small-zh-v1.5 (HuggingFace, 本地运行) |
| 向量数据库 | Chroma (嵌入式)                                |
| 框架       | LangChain (LCEL + Agent 编排)                  |
| UI         | Streamlit                                      |
| 包管理     | uv                                             |

### 目录结构

```
ai_qa_agent/
├── src/
│   ├── config/settings.py         # API 密钥、模型参数、路径
│   ├── prompts/                   # 提示词模板（与代码解耦）
│   │   ├── rag_agent_prompt.txt
│   │   └── rag_chain_prompt.txt
│   ├── models/                    # LLM & Embedding 工厂
│   │   ├── llm.py                 # ChatOpenAI 实例化
│   │   └── embeddings.py          # HuggingFaceEmbeddings 实例化
│   ├── retrieval/                 # 文档加载 & 向量库构建
│   │   ├── document_loader.py
│   │   └── vectorstore.py
│   ├── tools/                     # Agent 可调用工具
│   │   ├── base_tool.py
│   │   └── knowledge_retriever.py
│   ├── memory/chat_memory.py      # 对话历史管理
│   ├── agents/rag_agent.py        # ReAct Agent（LLM 自主决策）
│   ├── chains/rag_chain.py        # LCEL Chain（确定性地检索）
│   └── main.py                    # CLI 入口
├── ui/app.py                      # Streamlit Web UI
├── data/                          # 知识库文档
│   ├── ai_knowledge.md
│   ├── AI_master.docx
│   ├── machine_learning.md
│   └── neural_network.md
├── RULE.md                        # 架构规范
├── Streamlit.md                   # UI 需求规格
├── qa_notes.md                    # 开发问答记录
└── README.md
```

### 核心工作流

**Agent 模式** — LLM 自主决定是否调用检索工具：

```
用户提问 → Agent (LLM + 工具集)
              │
              ├── 决策：是否需要检索？
              │      ├── 是 → ai_knowledge_search → 向量库
              │      └── 否 → 直接回答
              │
              └── 最终回答
```

**Chain 模式** — 每次提问必检索的确定性管线：

```
用户提问 → 检索器 → Top-k 文档块 → 提示词组装 → LLM → 回答
```

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- (仅 Windows) Microsoft Word — `.doc` 格式解析依赖；如未安装可先转换为 `.docx`

### 安装

```bash
git clone <repo-url>
cd RAG_Langchain
uv python pin 3.11
uv sync
```

<details>
<summary>手动安装依赖</summary>

```bash
uv add langchain langchain-openai langchain-chroma langchain-text-splitters \
       langchain-huggingface sentence-transformers python-dotenv \
       pypdf python-docx pywin32 streamlit torchvision
```

</details>

### 配置

在项目根目录创建 `.env` 文件（该文件已在 `.gitignore` 中忽略，不会被提交到 GitHub）：

```bash
# 请替换为你自己的 API Key
OPENAI_API_KEY=sk-your-api-key-here

# HuggingFace 离线模式（可选，使用本地缓存的 embedding 模型）
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

首次运行时自动下载 embedding 模型（约 100MB），之后使用本地缓存。

### 运行

Streamlit Web 界面（推荐）：

```bash
uv run streamlit run ai_qa_agent/ui/app.py
```

CLI 终端：

```bash
uv run ai_qa_agent/src/main.py
```

## 文档支持

### 格式


| 格式         | 解析引擎           | 平台       |
| ------------ | ------------------ | ---------- |
| `.md` `.txt` | Python 内置        | 全平台     |
| `.pdf`       | pypdf              | 全平台     |
| `.docx`      | python-docx        | 全平台     |
| `.doc`       | pywin32 (Word COM) | 仅 Windows |

### 切分策略

- **Markdown 文件**：先用 `MarkdownHeaderTextSplitter` 按 `#`/`##`/`###` 标题切分，每个主题成块；过长块再用 `RecursiveCharacterTextSplitter` 二次切分。切分后将标题以 `h1 / h2 / h3` 格式拼接到 chunk 内容前，防止小主题因缺少父级关键词导致检索不到。
- **其他格式**：直接使用 `RecursiveCharacterTextSplitter`，分隔符 `["\n\n", "\n", " ", ""]`。

### 知识库

`data/` 目录包含以下 AI 领域参考文档：


| 文件                  | 内容                                        |
| --------------------- | ------------------------------------------- |
| `ai_knowledge.md`     | Transformer, LLM, RAG, 向量数据库等核心概念 |
| `AI_master.docx`      | AI 综合知识概览                             |
| `machine_learning.md` | 学习范式、模型评估、过拟合、常见算法        |
| `neural_network.md`   | 神经元、前向/反向传播、CNN/RNN/注意力机制   |

向 `data/` 添加新文档后，在 Streamlit 侧边栏点击 **重建知识库索引** 或重启 CLI 即可生效。
