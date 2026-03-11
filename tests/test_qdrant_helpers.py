"""Tests for QdrantHybridManager helper methods (point ID, filename variants)."""

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingest"))


class TestGeneratePointId:
    def setup_method(self):
        with patch.dict(
            "os.environ",
            {
                "QDRANT_URL": "https://test.example.com",
                "QDRANT_API_KEY": "test-key",
                "QDRANT_COLLECTION": "test",
                "OPENAI_API_KEY": "test",
                "LLAMA_CLOUD_API_KEY": "test",
            },
        ):
            # Import here to avoid env issues
            from smart_ingest_hybrid import HybridConfig, QdrantHybridManager

            self.config = HybridConfig()
            # Mock the Qdrant client to avoid connection
            with patch("smart_ingest_hybrid.QdrantClient"), patch("smart_ingest_hybrid.OpenAIEmbeddings"):
                self.manager = QdrantHybridManager(self.config)

    def test_deterministic(self):
        id1 = self.manager._generate_point_id("test.pdf", 0)
        id2 = self.manager._generate_point_id("test.pdf", 0)
        assert id1 == id2

    def test_different_chunks_different_ids(self):
        id1 = self.manager._generate_point_id("test.pdf", 0)
        id2 = self.manager._generate_point_id("test.pdf", 1)
        assert id1 != id2

    def test_different_files_different_ids(self):
        id1 = self.manager._generate_point_id("doc_a.pdf", 0)
        id2 = self.manager._generate_point_id("doc_b.pdf", 0)
        assert id1 != id2

    def test_returns_positive_integer(self):
        result = self.manager._generate_point_id("test.pdf", 42)
        assert isinstance(result, int)
        assert result > 0

    def test_matches_manual_md5(self):
        content = "test.pdf:5"
        expected = int.from_bytes(hashlib.md5(content.encode()).digest()[:8], byteorder="big")
        result = self.manager._generate_point_id("test.pdf", 5)
        assert result == expected


class TestGetFilenameVariants:
    def setup_method(self):
        with patch.dict(
            "os.environ",
            {
                "QDRANT_URL": "https://test.example.com",
                "QDRANT_API_KEY": "test-key",
                "QDRANT_COLLECTION": "test",
                "OPENAI_API_KEY": "test",
                "LLAMA_CLOUD_API_KEY": "test",
            },
        ):
            from smart_ingest_hybrid import HybridConfig, QdrantHybridManager

            self.config = HybridConfig()
            with patch("smart_ingest_hybrid.QdrantClient"), patch("smart_ingest_hybrid.OpenAIEmbeddings"):
                self.manager = QdrantHybridManager(self.config)

    def test_includes_original(self):
        variants = self.manager._get_filename_variants("documento.pdf")
        assert "documento.pdf" in variants

    def test_includes_md_variant(self):
        variants = self.manager._get_filename_variants("documento.pdf")
        assert "documento.md" in variants

    def test_includes_multiple_extensions(self):
        variants = self.manager._get_filename_variants("test.pdf")
        assert "test.xlsx" in variants
        assert "test.docx" in variants
        assert "test.txt" in variants
        assert "test.pptx" in variants

    def test_no_duplicates(self):
        variants = self.manager._get_filename_variants("test.pdf")
        assert len(variants) == len(set(variants))

    def test_md_file_input(self):
        variants = self.manager._get_filename_variants("test.md")
        assert "test.pdf" in variants
        assert "test.md" in variants
