# Architecture Reference

## Module Map

| Module | Path | Role |
|--------|------|------|
| config | `src/config/settings.py` | API keys, model params, paths, chunking params |
| models | `src/models/` | `llm.py`: ChatOpenAI factory; `embeddings.py`: HuggingFaceEmbeddings factory |
| retrieval | `src/retrieval/` | `document_loader.py`: multi-format loading; `vectorstore.py`: Chroma + mixed chunking |
| tools | `src/tools/` | `knowledge_retriever.py`: wraps vectorstore as Agent-callable tool |
| memory | `src/memory/chat_memory.py` | Conversation history (CLI) |
| agents | `src/agents/rag_agent.py` | ReAct Agent, uses `create_agent(langgraph)` |
| chains | `src/chains/rag_chain.py` | LCEL pipeline: retrieve -> format -> LLM -> parse |
| UI | `ui/app.py` | Streamlit: sidebar (docs, index), chat (typewriter, history) |

## Data Flow

### Chain Mode

```
User Question -> Retriever (Chroma) -> Top-k Chunks -> Prompt Assembly -> LLM (GLM-5.2) -> Answer
```

1. `retriever.invoke(question)` queries Chroma with embedding similarity
2. `_format_docs()` concatenates retrieved chunks
3. `ChatPromptTemplate` fills `{context}` and `{question}` into the system prompt
4. LLM generates answer based on retrieved context

### Agent Mode

```
User Question -> Agent (LLM + Tools)
                  |
                  Decision: call ai_knowledge_search?
                  |
                  YES -> Retrieve from Chroma -> Observe context
                  NO  -> Direct answer
                  |
                  Final Answer
```

1. Agent receives system prompt instructing it to use `ai_knowledge_search` tool
2. LLM decides whether the question requires retrieval
3. If tool is called, results are appended to message history
4. Agent produces final answer based on full context

## Key Design Decisions

- **Mixed chunking**: Markdown files split by headers first (preserving topic integrity), then by character for overflow chunks
- **Header injection**: After splitting, headers are prepended to chunk content to prevent keyword dilution (e.g., SVM chunk includes parent topic in searchable text)
- **Local embeddings**: BAAI/bge-small-zh-v1.5 runs locally for offline capability and cost savings
- **Dependency injection**: LLM and tools are factory functions, swappable without touching orchestration code
- **Prompt decoupling**: All system prompts live in `prompts/*.txt`, not hardcoded in Python

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| LLM | GLM-5.2 | Alibaba MaaS, OpenAI-compatible API |
| Embedding | BAAI/bge-small-zh-v1.5 | HuggingFace, local, ~100MB |
| Vector DB | Chroma | Embedded, no separate deployment |
| Framework | LangChain 1.3+ | LCEL + Agent orchestration |
| UI | Streamlit 1.50+ | Typewriter effect, session state |
| Package Manager | uv | Fast Python package manager |
