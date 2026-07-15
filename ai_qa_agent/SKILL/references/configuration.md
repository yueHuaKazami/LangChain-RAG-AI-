# Configuration Reference

## .env file

Located at project root. Not committed to git.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for GLM-5.2 (OpenAI-compatible) |
| `HF_HUB_OFFLINE` | No | `1` | Skip HuggingFace network checks |
| `TRANSFORMERS_OFFLINE` | No | `1` | Use local model cache only |

## Chunking Parameters

In `src/config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Max characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `RETRIEVER_K` | 5 | Top-k chunks returned per query |

## LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `glm-5.2` | Model name passed to API |
| `BASE_URL` | (MaaS endpoint) | OpenAI-compatible base URL |
| `LLM_TEMPERATURE` | `0` | Sampling temperature (0 = deterministic) |

## Embedding Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | HuggingFace model name |

## Supported Document Formats

| Extension | Engine | Platform |
|-----------|--------|----------|
| `.md` `.txt` | Python built-in | All |
| `.pdf` | pypdf | All |
| `.docx` | python-docx | All |
| `.doc` | pywin32 (Word COM) | Windows only |

## Dependencies

Managed by `uv` in `pyproject.toml`:

```
langchain, langchain-openai, langchain-chroma, langchain-text-splitters,
langchain-huggingface, sentence-transformers, python-dotenv,
pypdf, python-docx, pywin32, streamlit, torchvision
```
