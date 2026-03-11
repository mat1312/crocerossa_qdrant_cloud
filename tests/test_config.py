"""Tests for HybridConfig dataclass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingest"))

from smart_ingest_hybrid import HybridConfig


class TestHybridConfig:
    def test_defaults(self, mock_env):
        config = HybridConfig()
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 300
        assert config.dense_model == "text-embedding-3-large"
        assert config.dense_dimensions == 3072
        assert config.sparse_model == "Qdrant/bm25"
        assert config.enable_sparse is True
        assert config.batch_size == 32

    def test_env_loading(self, mock_env):
        config = HybridConfig()
        assert config.llama_api_key == "test-llama-key"
        assert config.openai_api_key == "test-openai-key"
        assert config.qdrant_url == "https://test-qdrant.example.com"
        assert config.qdrant_api_key == "test-qdrant-key"
        assert config.qdrant_collection == "test_collection"

    def test_custom_values(self, mock_env):
        config = HybridConfig(chunk_size=1500, chunk_overlap=200, enable_sparse=False)
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 200
        assert config.enable_sparse is False

    def test_supported_extensions(self, mock_env):
        config = HybridConfig()
        assert ".pdf" in config.supported_extensions
        assert ".xlsx" in config.supported_extensions
        assert ".docx" in config.supported_extensions
        assert ".pptx" in config.supported_extensions
        assert ".txt" not in config.supported_extensions
