# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval Augmented Generation) system for Italian Red Cross (Croce Rossa Italiana) documentation. The system parses PDF/Office documents, chunks them semantically, generates embeddings, and stores them in Qdrant Cloud for semantic search.

**Technology Stack:**
- **Document Parsing**: LlamaParse (supports PDF, DOCX, XLSX, XLS, PPTX)
- **Vector Database**: Qdrant Cloud
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Text Processing**: LangChain (chunking, document loaders)
- **Language**: Python 3.x

## Environment Setup

Required environment variables in `.env`:
```bash
LLAMA_CLOUD_API_KEY=...     # LlamaParse API key
OPENAI_API_KEY=...          # OpenAI API key (for embeddings)
QDRANT_URL=...              # Qdrant Cloud cluster URL
QDRANT_API_KEY=...          # Qdrant Cloud API key
QDRANT_COLLECTION=...       # Collection name (e.g., crocerossa_docs)
```

**Install dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix
pip install -r requirements.txt
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
[3] OpenAI Embeddings → Vectors (3072-dim)
    ↓
[4] Qdrant Cloud → Indexed & Searchable
```

### Key Configuration Parameters

**Chunking** ([ingest_qdrant_new.py:37-39](ingest_qdrant_new.py#L37-L39), [smart_ingest.py:61-62](smart_ingest.py#L61-L62)):
- `CHUNK_SIZE = 800` characters
- `CHUNK_OVERLAP = 100` characters
- Separators prioritize Markdown headers (`\n## `, `\n### `), then paragraphs, then sentences

**Embedding Model** ([ingest_qdrant_new.py:39](ingest_qdrant_new.py#L39)):
- Model: `text-embedding-3-large`
- Dimensions: 3072
- Distance metric: Cosine similarity

**Batch Processing**:
- Qdrant upsert batch size: 50 chunks ([ingest_qdrant_new.py:170](ingest_qdrant_new.py#L170))
- Timeout: 120 seconds for Qdrant operations

## Primary Scripts

### 1. smart_ingest.py (RECOMMENDED)

**Purpose**: Unified pipeline for parsing, chunking, and indexing. Supports automatic document replacement and handles multiple file formats.

**Usage:**
```bash
# Process entire folder
python smart_ingest.py new_cri_docs/

# Process single file
python smart_ingest.py new_cri_docs/documento.pdf

# Dry run (show what would happen)
python smart_ingest.py new_cri_docs/ --dry-run

# Add-only mode (skip if exists)
python smart_ingest.py new_cri_docs/ --mode=add-only

# Custom chunk size
python smart_ingest.py new_cri_docs/ --chunk-size=1000 --chunk-overlap=150
```

**Key Features:**
- Automatically detects and replaces existing documents (searches by filename without extension)
- Generates deterministic point IDs using MD5 hash of `filename:chunk_id`
- Supports nested metadata format for backward compatibility
- Batch processing with progress logging

**Architecture Notes:**
- Entry point: `main()` at [smart_ingest.py:725](smart_ingest.py#L725)
- Three main classes: `DocumentParser`, `DocumentChunker`, `QdrantManager`
- Config dataclass centralizes all parameters: [smart_ingest.py:51-79](smart_ingest.py#L51-L79)

### 2. ingest_qdrant_new.py

**Purpose**: Full reindex of all documents in the `parsed/` folder. **DESTRUCTIVE** - deletes entire collection and rebuilds from scratch.

**Usage:**
```bash
python ingest_qdrant_new.py
```

**⚠️ Warning**: This script deletes the entire Qdrant collection before re-ingesting. Use only when you need to rebuild the entire index.

**Process:**
1. Delete collection if exists ([ingest_qdrant_new.py:105-121](ingest_qdrant_new.py#L105-L121))
2. Load all `.md` files from `parsed/` folder
3. Chunk documents
4. Generate embeddings and upsert to Qdrant

### 3. ingest_single_file.py

**Purpose**: Add a single markdown file to existing collection without deleting anything.

**Usage:**
```bash
# Absolute path
python ingest_single_file.py /path/to/file.md

# Relative to parsed/ folder
python ingest_single_file.py documento.md
```

**Key Difference**: Only appends; does NOT check for or remove existing versions of the document.

### 4. delete_single_document.py

**Purpose**: Remove all chunks belonging to a specific document from Qdrant.

**Usage:**
```bash
# Dry run (preview)
python delete_single_document.py documento.pdf --dry-run

# Actually delete (requires confirmation)
python delete_single_document.py documento.pdf

# Force delete (no confirmation)
python delete_single_document.py documento.pdf --force
```

**How it works**: Searches for chunks matching `metadata.filename` or `metadata.source` fields.

### 5. parse_async_withdoc.py

**Purpose**: Parallel PDF/DOC parsing using LlamaParse with multiprocessing.

**Usage:**
```bash
# Place files in todo_parsing/ folder, then run:
python parse_async_withdoc.py
```

**Output**: Markdown files in `parsed/` folder.

**Architecture**: Uses `ProcessPoolExecutor` with `max_workers = min(os.cpu_count(), total_files)` for parallel processing.

### 6. converter_mdtotxt.py

**Purpose**: Convert markdown files to plain text (utility script).

**Usage:**
```bash
python converter_mdtotxt.py
```

Reads from `full_parsed/`, writes to `full_parsed_txt/`.

## Folder Structure

```
crocerossa_qdrant_cloud/
├── .env                      # Environment variables (API keys)
├── requirements.txt          # Python dependencies
├── smart_ingest.py          # [PRIMARY] Unified ingest pipeline
├── ingest_qdrant_new.py     # Full reindex (destructive)
├── ingest_single_file.py    # Add single file (append only)
├── delete_single_document.py # Remove document by filename
├── parse_async_withdoc.py   # Parallel PDF parsing
├── converter_mdtotxt.py     # MD to TXT converter
│
├── new_cri_docs/            # [INPUT] New documents to process
├── cri_docs/                # [INPUT] Source documents (older)
├── todo_parsing/            # [INPUT] Files queued for parsing
│
├── parsed/                  # [OUTPUT] Parsed markdown (gitignored)
├── full_parsed/             # [OUTPUT] Full parsed documents
└── full_parsed_txt/         # [OUTPUT] Plain text conversions
```

**Note**: `parsed/`, `todo_parsing/`, and `.env` are gitignored.

## Metadata Structure

Each chunk stored in Qdrant has the following metadata (flat structure in latest version):

```python
{
    "page_content": "chunk text...",
    "metadata": {
        "filename": "documento.pdf",
        "source": "/full/path/to/documento.pdf",
        "document_type": "PDF",  # or "Word", "Excel", "PowerPoint"
        "chunk_id": 0,
        "total_chunks": 42,
        "chunk_title": "First 50 chars of chunk...",
        "processed_date": "2025-12-12T10:30:00",
        "file_hash": "md5_hash_of_file"
    }
}
```

**Backward Compatibility**: Older chunks may have nested `metadata.metadata` structure. The search logic in `smart_ingest.py` handles both formats ([smart_ingest.py:309-325](smart_ingest.py#L309-L325)).

## Common Development Tasks

### Adding a New Document

**Recommended workflow:**
```bash
# 1. Place document in new_cri_docs/
# 2. Run smart ingest
python smart_ingest.py new_cri_docs/nuovo_documento.pdf

# The script will:
# - Parse PDF to Markdown (LlamaParse)
# - Chunk the content
# - Generate embeddings
# - Upsert to Qdrant (auto-replaces if exists)
```

### Updating an Existing Document

**smart_ingest.py automatically handles updates:**
```bash
# Just process the new version - old chunks are auto-deleted
python smart_ingest.py new_cri_docs/documento_aggiornato.pdf
```

The script searches for any existing chunks with matching filename (ignoring extension) and deletes them before adding new ones ([smart_ingest.py:541-553](smart_ingest.py#L541-L553)).

### Full Reindex

**When needed**: Changed chunk size, embedding model, or need to clean up inconsistencies.

```bash
# 1. Ensure all source markdown files are in parsed/
# 2. Run full reindex (DESTRUCTIVE!)
python ingest_qdrant_new.py
```

### Checking Collection Stats

```python
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

info = client.get_collection("crocerossa_docs")
print(f"Points: {info.points_count}")
print(f"Status: {info.status}")
```

### Modifying Chunk Size

**In smart_ingest.py:**
```bash
python smart_ingest.py new_cri_docs/ --chunk-size=1200 --chunk-overlap=150
```

**In other scripts:**
Edit the constants at the top of the file:
- [ingest_qdrant_new.py:37-39](ingest_qdrant_new.py#L37-L39)
- [ingest_single_file.py:36-38](ingest_single_file.py#L36-L38)

**⚠️ Important**: Changing chunk size requires a full reindex for consistency.

## Document Filename Handling

**Critical Design Decision**: The system searches for documents **without extension** to handle format changes.

Example: If `Codice-etico.pdf` was previously parsed as `Codice-etico.md`, the system will still find and replace it when you upload a new `Codice-etico.pdf`.

Implementation: [smart_ingest.py:295-301](smart_ingest.py#L295-L301) generates filename variants with different extensions for matching.

## Error Handling & Troubleshooting

### Common Issues

**1. "Collection does not exist"**
- Run any ingest script with a valid file to auto-create the collection
- Or use `smart_ingest.py` which calls `ensure_collection_exists()`

**2. Timeout errors during upsert**
- Reduce batch size: edit `batch_size` parameter (default: 50)
- Increase timeout: edit `QdrantClient(timeout=...)` (default: 120s)

**3. "File already exists" with add-only mode**
- Use `--mode=replace` (default) to overwrite
- Or delete manually: `python delete_single_document.py filename.pdf`

**4. Empty content after parsing**
- Check if LlamaParse API key is valid
- Verify file is not corrupted
- Some PDFs may be image-only (requires OCR)

### Logging

All scripts use Python's `logging` module at INFO level. To enable DEBUG:
```bash
python smart_ingest.py new_cri_docs/ --verbose
```

Or programmatically:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Testing Changes

When modifying the ingestion pipeline:

1. **Use dry-run mode first:**
   ```bash
   python smart_ingest.py new_cri_docs/ --dry-run
   ```

2. **Test with a single small document:**
   ```bash
   python smart_ingest.py new_cri_docs/small_test.pdf
   ```

3. **Verify in Qdrant:**
   ```python
   # Check that chunks were created with correct metadata
   client.scroll(
       collection_name="crocerossa_docs",
       scroll_filter=models.Filter(
           must=[models.FieldCondition(
               key="metadata.filename",
               match=models.MatchValue(value="small_test.pdf")
           )]
       ),
       limit=5
   )
   ```

4. **Test deletion:**
   ```bash
   python delete_single_document.py small_test.pdf --dry-run
   ```

## Code Style & Patterns

- **Environment config**: Always load from `.env` using `python-dotenv`
- **Logging**: Use `logging` module, not `print()` for operational messages
- **Error handling**: Wrap external API calls (LlamaParse, OpenAI, Qdrant) in try-except with logging
- **Async operations**: Use `asyncio` for I/O-bound operations (see [ingest_qdrant_new.py:230-260](ingest_qdrant_new.py#L230-L260))
- **Type hints**: Used in `smart_ingest.py` with dataclasses and typing module
- **Docstrings**: Italian comments in older scripts, English in `smart_ingest.py`

## API Rate Limits & Costs

**LlamaParse:**
- Free tier: 1000 pages/day
- Rate limits: Check LlamaCloud dashboard

**OpenAI Embeddings:**
- Model: text-embedding-3-large
- Cost: ~$0.13 per 1M tokens
- A 800-char chunk ≈ 200 tokens
- 1000 chunks ≈ 200K tokens ≈ $0.026

**Qdrant Cloud:**
- Depends on plan (check Qdrant dashboard)
- Each chunk = 1 point
- Vector size: 3072 floats (12KB per vector)

## Migration Notes

If updating from older ingestion scripts to `smart_ingest.py`:

1. Old chunks have nested `metadata.metadata` structure
2. New chunks have flat `metadata` structure
3. Both formats are supported in queries ([smart_ingest.py:309-330](smart_ingest.py#L309-L330))
4. No migration needed - old and new chunks coexist
5. When documents are updated via `smart_ingest.py`, old chunks are replaced with new format
