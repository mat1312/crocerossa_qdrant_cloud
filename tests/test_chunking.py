"""Tests for PageAwareChunker chunk optimization and doc type detection."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingest"))

from langchain_core.documents import Document


def make_doc(text: str, **meta) -> Document:
    """Helper to create a Document with given text and metadata."""
    return Document(page_content=text, metadata=meta)


def make_chunker():
    """Create a PageAwareChunker with real config values."""
    with patch.dict(
        "os.environ",
        {
            "LLAMA_CLOUD_API_KEY": "test",
            "OPENAI_API_KEY": "test",
            "QDRANT_URL": "https://test.example.com",
            "QDRANT_API_KEY": "test",
            "QDRANT_COLLECTION": "test",
        },
    ):
        from smart_ingest_hybrid import HybridConfig, PageAwareChunker

        config = HybridConfig()
        return PageAwareChunker(config)


class TestOptimizeChunks:
    def setup_method(self):
        self.chunker = make_chunker()

    def test_empty_input(self):
        assert self.chunker._optimize_chunks([]) == []

    def test_all_large_chunks_unchanged(self):
        chunks = [make_doc("x" * 500), make_doc("y" * 600)]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        assert len(result) == 2
        assert result[0].page_content == "x" * 500
        assert result[1].page_content == "y" * 600

    def test_small_chunk_merged_with_next(self):
        chunks = [make_doc("small"), make_doc("x" * 500)]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        assert len(result) == 1
        assert "small" in result[0].page_content
        assert "x" * 500 in result[0].page_content

    def test_multiple_small_chunks_merged(self):
        chunks = [make_doc("a" * 100), make_doc("b" * 100), make_doc("c" * 500)]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        # a+b < 400, so buffer continues; a+b+c >= 400, so flushed
        assert len(result) == 1

    def test_residual_buffer_appended_to_last(self):
        chunks = [make_doc("x" * 500), make_doc("tiny")]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        assert len(result) == 1
        assert "tiny" in result[0].page_content
        assert "x" * 500 in result[0].page_content

    def test_single_small_chunk_kept(self):
        chunks = [make_doc("tiny")]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        assert len(result) == 1
        assert result[0].page_content == "tiny"

    def test_separator_between_merged(self):
        chunks = [make_doc("first"), make_doc("second" * 100)]
        result = self.chunker._optimize_chunks(chunks, min_chars=400)
        assert "\n\n" in result[0].page_content


class TestGetDocType:
    def setup_method(self):
        self.chunker = make_chunker()

    def test_pdf(self):
        assert self.chunker._get_doc_type("documento.pdf") == "PDF"

    def test_xlsx(self):
        assert self.chunker._get_doc_type("dati.xlsx") == "Excel"

    def test_xls(self):
        assert self.chunker._get_doc_type("vecchio.xls") == "Excel"

    def test_docx(self):
        assert self.chunker._get_doc_type("testo.docx") == "Word"

    def test_doc(self):
        assert self.chunker._get_doc_type("vecchio.doc") == "Word"

    def test_unknown(self):
        assert self.chunker._get_doc_type("file.txt") == "Unknown"

    def test_case_insensitive(self):
        assert self.chunker._get_doc_type("FILE.PDF") == "PDF"
