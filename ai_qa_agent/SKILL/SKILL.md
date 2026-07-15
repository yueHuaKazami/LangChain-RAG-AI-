---
name: "rag-qa-system"
description: "RAG-powered AI knowledge Q&A system built on LangChain, Chroma, and GLM-5.2. Supports dual-mode Q&A (Agent and Chain), multi-format document ingestion, Streamlit web UI, and local embeddings. Use when the user needs to (1) set up or configure a RAG-based Q&A system, (2) run the Streamlit web UI or CLI interface, (3) add new knowledge documents to the vector database, (4) debug RAG retrieval or generation issues, (5) switch between Agent and Chain modes, or (6) understand the project's modular architecture."
---

# RAG QA System

AI knowledge Q&A system. RAG (Retrieval-Augmented Generation) with LangChain + Chroma + GLM-5.2. Dual-mode: Agent (LLM autonomous retrieval decisions) and Chain (deterministic retrieve-then-generate). Local BAAI/bge-small-zh-v1.5 embeddings. Streamlit web UI with document upload and conversation history.

## Project layout

```
ai_qa_agent/
├── src/                          # Core agent logic
│   ├── config/settings.py        # API keys, model params, chunking params
│   ├── models/                   # LLM & embedding factories
│   ├── retrieval/                # Document loading + vectorstore
│   │   ├── document_loader.py    # .md .txt .pdf .docx .doc
│   │   └── vectorstore.py        # Chroma + mixed chunking
│   ├── tools/                    # Agent-callable tools
│   ├── agents/rag_agent.py       # ReAct Agent (LLM decides retrieve)
│   ├── chains/rag_chain.py       # LCEL Chain (always retrieves)
│   └── main.py                   # CLI entry
├── ui/app.py                     # Streamlit web UI
├── data/                         # Knowledge base (.md .docx .pdf)
├── RULE.md                       # Architecture spec
├── Streamlit.md                  # UI spec
└── README.md
```

## Quick start

### Run Streamlit UI

```bash
cd <project-root>
uv run streamlit run ai_qa_agent/ui/app.py
```

### Run CLI

```bash
cd <project-root>
uv run ai_qa_agent/src/main.py
```

### Prerequisites

- Python 3.11+, uv package manager
- `.env` at project root with `OPENAI_API_KEY=sk-your-key`
- First run downloads BAAI/bge-small-zh-v1.5 embedding model (~100 MB)

## Core workflows

### Adding knowledge documents

1. Place `.md` / `.txt` / `.pdf` / `.docx` / `.doc` files in `ai_qa_agent/data/`
2. Streamlit: click "重建知识库索引" in sidebar
3. CLI: restart the program (auto-loads on startup)

### Switching Q&A modes

Use the mode selector in the Streamlit sidebar or choose at CLI startup:

- **Chain mode**: Deterministic. Every question triggers vector DB retrieval first, then LLM generation. Lower latency, suitable for strict knowledge-base Q&A.
- **Agent mode**: Flexible. LLM autonomously decides whether to retrieve via the `ai_knowledge_search` tool. Handles mixed casual + professional questions.

### Conversation history

History is persisted to `ai_qa_agent/conversations.json` (gitignored). Each session is identified by mode + first question. The sidebar lists all past conversations; click to resume any session. Switching modes saves the current session before starting a new one.

### Document chunking strategy

- Markdown files: `MarkdownHeaderTextSplitter` by `#`/`##`/`###`, then `RecursiveCharacterTextSplitter` for overflow. Headers are prepended to chunk content to prevent keyword dilution.
- Other formats: `RecursiveCharacterTextSplitter` with `["\n\n", "\n", " ", ""]`.

Key parameters in `src/config/settings.py`: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`, `RETRIEVER_K=5`.

## Architecture

See [references/architecture.md](references/architecture.md) for detailed module responsibilities and data flow.

## Configuration

See [references/configuration.md](references/configuration.md) for full parameter reference.

## Environment validation

Run `scripts/validate_env.py` to check Python version, dependencies, .env, and data directory.
