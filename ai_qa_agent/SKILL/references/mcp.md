# MCP Integration

The RAG QA System exposes its knowledge retrieval as an MCP (Model Context Protocol) server, allowing any MCP-compatible client (Claude Desktop, Codex, etc.) to search the AI knowledge base.

## Architecture

```
MCP Client (Claude/Codex)
    │
    │ stdio (JSON-RPC)
    ▼
mcp_server.py
    │
    ├── ai_knowledge_search(query, k) -> str
    │       │
    │       ▼
    │   Chroma VectorStore
    │       │
    │       ▼
    │   BAAI/bge-small-zh-v1.5 Embeddings
    │
    └── ai_list_documents() -> str
```

## Installation

```bash
uv add mcp
```

## Run

```bash
# From project root
uv run ai_qa_agent/src/mcp_server.py
```

The server starts on stdio transport. It initializes the Chroma vectorstore on first tool call (lazy loading).

## Exposed Tools

### `ai_knowledge_search`

Search AI knowledge base for relevant document chunks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query |
| `k` | `int` | `5` | Number of results |

Returns formatted results with document source and content.

### `ai_list_documents`

List all loaded documents in the knowledge base.

No parameters. Returns a list of document filenames.

## Client Configuration

### Codex / Claude Desktop

Add to MCP settings (`codex_mcp_settings.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-knowledge-search": {
      "command": "uv",
      "args": ["run", "ai_qa_agent/src/mcp_server.py"],
      "cwd": "/absolute/path/to/RAG_Langchain"
    }
  }
}
```

### Debugging

Use the MCP Inspector to test tools interactively:

```bash
uv run mcp dev ai_qa_agent/src/mcp_server.py
```

## Adding New Documents

1. Place files in `ai_qa_agent/data/`
2. Restart the MCP server (the vectorstore is rebuilt on startup)
3. Or restart the Streamlit app and click "重建知识库索引"
