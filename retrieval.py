"""Utility per costruire un retriever similarity + Cohere Rerank v3.5
usando il wrapper ufficiale `ContextualCompressionRetriever`.
Restituisce un oggetto `BaseRetriever` compatibile con tutte le catene
LangChain.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank  # document compressor

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")
RERANK_K: int = int(os.getenv("RERANK_K", "20"))  # chunk finali
FETCH_K: int = int(os.getenv("FETCH_K", "100"))   # chunk iniziali


def _check_config() -> None:
    if not COHERE_API_KEY:
        raise EnvironmentError(
            "COHERE_API_KEY non trovata. Aggiungila a .env o variabili d'ambiente."
        )


def build_rerank_retriever(vector_store):
    """Restituisce un `ContextualCompressionRetriever` (BaseRetriever)."""

    _check_config()

    # 1️⃣ Primo step: similarità dal vector store
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FETCH_K},
    )

    # 2️⃣ Compressor Cohere (filtra/riordina)
    compressor = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        top_n=RERANK_K,
        model="rerank-v3.5",
    )

    # 3️⃣ Retriever combinato
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
