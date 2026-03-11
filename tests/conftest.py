"""Shared test fixtures."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src/ to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingest"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "tools"))


@pytest.fixture
def mock_env():
    """Mock environment variables for tests."""
    env = {
        "LLAMA_CLOUD_API_KEY": "test-llama-key",
        "OPENAI_API_KEY": "test-openai-key",
        "QDRANT_URL": "https://test-qdrant.example.com",
        "QDRANT_API_KEY": "test-qdrant-key",
        "QDRANT_COLLECTION": "test_collection",
    }
    with patch.dict("os.environ", env):
        yield env
