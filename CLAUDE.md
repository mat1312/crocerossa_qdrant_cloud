# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval Augmented Generation) system for Italian Red Cross (Croce Rossa Italiana) documentation. The system parses PDF/Office documents, chunks them semantically, generates embeddings, and stores them in Qdrant Cloud for semantic search.

**Technology Stack:**
- **Package Manager**: uv (by Astral)
- **Linter/Formatter**: Ruff
- **Document Parsing**: LlamaParse via `llama-cloud`
- **Vector Database**: Qdrant Cloud
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Sparse Vectors**: FastEmbed BM25
- **Text Processing**: LangChain (chunking, document loaders)
- **Language**: Python 3.11+

## Environment Setup

Required environment variables in `.env`:
```bash
LLAMA_CLOUD_API_KEY=...     # LlamaParse API key
OPENAI_API_KEY=...          # OpenAI API key (for embeddings)
QDRANT_URL=...              # Qdrant Cloud cluster URL
QDRANT_API_KEY=...          # Qdrant Cloud API key
QDRANT_COLLECTION=...       # Collection name (e.g., crocerossa_docs)
```

**Install dependencies with uv:**
```bash
uv sync          # Install all dependencies (creates .venv automatically)
uv sync --no-dev # Production only (skip ruff)
```

**Run scripts:**
```bash
uv run python src/ingest/smart_ingest.py new_cri_docs/
uv run python src/ingest/smart_ingest_hybrid.py new_docs/
uv run python src/ingest/ingest_volontariato.py docs_onlyvolontario/
uv run python src/dashboard/qdrant_dashboard.py
```

**Linting & formatting:**
```bash
uv run ruff check src/         # Lint
uv run ruff check src/ --fix   # Lint + autofix
uv run ruff format src/        # Format
```

## Core Architecture

### Document Processing Pipeline

```
Source Documents (PDF/DOCX/XLSX)
    ↓
[1] LlamaParse → Markdown
    ↓
[2] RecursiveCharacterTextSplitter → Chunks (800 chars, 100 overlap)
    ↓
[3] OpenAI Embeddings (dense) + FastEmbed BM25 (sparse)
    ↓
[4] Qdrant Cloud → Indexed & Searchable (hybrid search)
```

### Key Configuration Parameters

**Chunking:**
- `CHUNK_SIZE = 800` characters
- `CHUNK_OVERLAP = 100` characters
- Separators prioritize Markdown headers (`\n## `, `\n### `), then paragraphs, then sentences

**Embedding Model:**
- Dense: `text-embedding-3-large` (3072 dimensions, cosine similarity)
- Sparse: `Qdrant/bm25` via FastEmbed

**Batch Processing:**
- Qdrant upsert batch size: 50 chunks
- Timeout: 120 seconds for Qdrant operations

## Project Structure

```
crocerossa_qdrant_cloud/
├── pyproject.toml               # Dependencies & project config (uv)
├── ruff.toml                    # Ruff linter/formatter config
├── .env                         # Environment variables (gitignored)
├── .gitignore
├── CLAUDE.md
│
├── src/
│   ├── ingest/                  # Document ingestion pipelines
│   │   ├── smart_ingest.py      # v3.1 - Hybrid search + safe Excel (pandas)
│   │   ├── smart_ingest_hybrid.py # v3.2 - Full hybrid with Italian stemmer
│   │   └── ingest_volontariato.py # Ingest for docs_onlyvolontariato collection
│   │
│   ├── tools/                   # Utility scripts
│   │   ├── purge_excel.py       # Remove Excel chunks from Qdrant
│   │   ├── setup_text_index.py  # One-shot Italian text index setup
│   │   └── search_keyword.py    # BM25 keyword search tool
│   │
│   ├── evaluation/              # RAG quality evaluation
│   │   ├── evaluate_rag.py      # Ragas-based evaluation metrics
│   │   └── generate_dataset.py  # Synthetic golden dataset generation
│   │
│   └── dashboard/               # Streamlit management UI
│       └── qdrant_dashboard.py  # Collection explorer & analytics
│
├── docs_onlyvolontario/         # [INPUT] Volontariato docs (gitignored)
├── new_cri_docs/                # [INPUT] New documents to process (gitignored)
├── cri_docs/                    # [INPUT] Source documents (gitignored)
├── parsed/                      # [OUTPUT] Parsed markdown (gitignored)
└── full_parsed/                 # [OUTPUT] Full parsed documents (gitignored)
```

## Primary Scripts

### 1. smart_ingest.py (v3.1)

Pipeline unificata parsing + chunking + indexing. Supporta sostituzione automatica documenti e formati multipli. Usa Pandas per Excel (no hallucinations).

```bash
uv run python src/ingest/smart_ingest.py new_cri_docs/
uv run python src/ingest/smart_ingest.py new_cri_docs/ --dry-run
uv run python src/ingest/smart_ingest.py new_cri_docs/ --mode=add-only
uv run python src/ingest/smart_ingest.py new_cri_docs/ --chunk-size=1000 --chunk-overlap=150
```

### 2. smart_ingest_hybrid.py (v3.2 - RECOMMENDED)

Versione evoluta con ricerca ibrida completa: dense + sparse vectors, text index italiano con stemmer Snowball, ASCII folding, stopwords.

```bash
uv run python src/ingest/smart_ingest_hybrid.py new_docs/
uv run python src/ingest/smart_ingest_hybrid.py doc.pdf --dry-run
uv run python src/ingest/smart_ingest_hybrid.py new_docs/ --no-sparse
```

### 3. ingest_volontariato.py

Pipeline per la collection `docs_onlyvolontariato`. Tutti i documenti passano da LlamaParse cost_effective.

```bash
uv run python src/ingest/ingest_volontariato.py docs_onlyvolontario/
```

### 4. Dashboard

```bash
uv run streamlit run src/dashboard/qdrant_dashboard.py
```

### 5. Evaluation

```bash
uv run python src/evaluation/generate_dataset.py --pdf path/to/file.pdf --size 50
uv run python src/evaluation/evaluate_rag.py --dataset golden_dataset.csv
```

## Metadata Structure

Each chunk stored in Qdrant:
```python
{
    "page_content": "chunk text...",
    "metadata": {
        "filename": "documento.pdf",
        "source": "/full/path/to/documento.pdf",
        "document_type": "PDF",
        "chunk_id": 0,
        "total_chunks": 42,
        "chunk_title": "First 50 chars of chunk...",
        "processed_date": "2025-12-12T10:30:00",
        "file_hash": "md5_hash_of_file"
    }
}
```

**Backward Compatibility**: Older chunks may have nested `metadata.metadata` structure. Search logic handles both formats.

## Common Development Tasks

### Adding/Updating a Document

```bash
# smart_ingest auto-replaces existing documents (matches by filename without extension)
uv run python src/ingest/smart_ingest_hybrid.py new_cri_docs/documento.pdf
```

### Checking Collection Stats

```python
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
info = client.get_collection("crocerossa_docs")
print(f"Points: {info.points_count}, Status: {info.status}")
```

## Code Style & Patterns

- **Linter/Formatter**: Ruff (config in `ruff.toml`)
- **Environment config**: Always load from `.env` using `python-dotenv`
- **Logging**: Use `logging` module, not `print()` for operational messages
- **Error handling**: Wrap external API calls (LlamaParse, OpenAI, Qdrant) in try-except with logging
- **Type hints**: dataclasses + typing module

## API Rate Limits & Costs

| Service | Detail |
|---------|--------|
| LlamaParse | Free tier: 1000 pages/day |
| OpenAI Embeddings | text-embedding-3-large, ~$0.13/1M tokens |
| Qdrant Cloud | 3072-dim vectors, ~12KB per point |
