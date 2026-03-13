"""
Ingest Pipeline - COLLECTION: docs_onlyvolontariato
Basato su ingest1162.py v3.2 con LlamaParse v2 tier=cost_effective.

Tutti i documenti (PDF, DOCX, XLSX, PPTX) passano da LlamaParse cost_effective.

Usage:
    python ingest_volontariato.py new_docs/              # Processa cartella
    python ingest_volontariato.py doc.pdf                # Singolo file
    python ingest_volontariato.py new_docs/ --dry-run    # Preview
    python ingest_volontariato.py new_docs/ --no-sparse  # Solo dense
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document processing - LlamaParse imported lazily in _parse_with_llamaparse()
import pandas as pd

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    Language,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SnowballLanguage,
    SnowballParams,
    SparseVector,
    SparseVectorParams,
    StopwordsSet,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# COLLECTION NAME (HARDCODED)
# ============================================================================
COLLECTION_NAME = "docs_onlyvolontariato"


# ============================================================================
# CONFIGURAZIONE
# ============================================================================


@dataclass
class HybridConfig:
    """Configurazione centralizzata."""

    # === LlamaParse ===
    llama_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY", ""))
    llama_result_type: str = "markdown"
    llama_language: str = "it"
    llama_num_workers: int = 4
    # LlamaParse v2: tier system
    llama_tier: str = "cost_effective"
    llama_version: str = "latest"

    # === Chunking ===
    chunk_size: int = 2000
    chunk_overlap: int = 300

    # === Dense Embeddings ===
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    dense_model: str = "text-embedding-3-large"
    dense_dimensions: int = 3072
    dense_vector_name: str = "dense"

    # === Sparse Embeddings (BM25) ===
    enable_sparse: bool = True
    sparse_model: str = "Qdrant/bm25"
    sparse_vector_name: str = "sparse"

    # === Qdrant ===
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = COLLECTION_NAME

    # === Processing ===
    batch_size: int = 32

    # === Estensioni supportate ===
    supported_extensions: tuple = (".pdf", ".xlsx", ".xls", ".doc", ".docx", ".pptx")


# ============================================================================
# DOCUMENT PARSER - LlamaParse v2 cost_effective
# ============================================================================


class DocumentParser:
    """Parser documenti: Pandas per Excel, LlamaParse v2 per il resto."""

    def __init__(self, config: HybridConfig):
        self.config = config

    def parse_file(self, filepath: str) -> str | list[str]:
        """Parsa un file. Per Excel restituisce list[str] (un testo per riga).
        Per altri formati restituisce str (markdown da LlamaParse)."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in self.config.supported_extensions:
            raise ValueError(f"Estensione non supportata: {ext}")

        if ext in (".xlsx", ".xls"):
            return self._parse_excel_pandas(filepath)
        return self._parse_with_llamaparse(filepath)

    def _parse_excel_pandas(self, filepath: Path) -> list[str]:
        """Parsa Excel con Pandas: ogni riga diventa un testo in linguaggio naturale."""
        logger.info(f"📊 Parsing Excel (Pandas): {filepath.name}")
        start_time = time.time()

        df = pd.read_excel(filepath, sheet_name=0, dtype=str)
        df = df.fillna("")

        # Pulizia colonne vuote
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, (df != "").any(axis=0)]

        # Rileva tipo di file dai nomi colonne
        col_lower = [str(c).lower() for c in df.columns]
        col_str = " ".join(col_lower)

        if "comitato territoriale" in col_str or "zona sismica" in col_str:
            rows = self._format_competenze_territoriali(df)
        elif "delegato" in col_str or "presidente" in col_str:
            rows = self._format_delegati_territoriali(df)
        else:
            rows = self._format_generic(df)

        # Filtra righe vuote
        rows = [r for r in rows if len(r.strip()) > 20]

        elapsed = time.time() - start_time
        logger.info(f"✅ Excel parsed: {filepath.name} → {len(rows)} righe ({elapsed:.2f}s)")
        return rows

    def _format_competenze_territoriali(self, df: pd.DataFrame) -> list[str]:
        """Formatta file Competenze Territoriali: 1 riga = 1 comune."""
        rows = []
        for _, row in df.iterrows():
            citta = str(row.iloc[0]).strip()
            provincia = str(row.iloc[1]).strip()
            comitato = str(row.iloc[2]).strip()
            zona_meteo = str(row.iloc[3]).strip()
            zona_sismica = str(row.iloc[4]).strip()
            zona_aib = str(row.iloc[6]).strip() if len(row) > 6 else ""
            toponomastica = str(row.iloc[7]).strip() if len(row) > 7 else ""

            if not citta or not comitato:
                continue

            parts = [
                f"Il comune di {citta} (provincia di {provincia}) "
                f"è sotto la competenza territoriale del Comitato CRI di {comitato}.",
            ]
            if zona_meteo:
                parts.append(f"Zona di allerta meteo: {zona_meteo}.")
            if zona_sismica:
                parts.append(f"Zona sismica: {zona_sismica}.")
            if zona_aib:
                parts.append(f"Zona AIB (antincendio boschivo): {zona_aib}.")
            if toponomastica:
                parts.append(f"Riferimento normativo: {toponomastica}.")

            rows.append(" ".join(parts))
        return rows

    def _format_delegati_territoriali(self, df: pd.DataFrame) -> list[str]:
        """Formatta file Delegati Territoriali: 1 riga = 1 comitato con tutti i delegati."""
        rows = []

        # Le colonne hanno merged headers, mappiamo manualmente
        # Col 0: Comitati, 1: Presidente, 2: cellulare, 3: Atto
        # Poi gruppi di 3: (Nome, mail, telefono) per ogni delegato
        delegati_nomi = [
            "Delegato Salute",
            "Delegato Inclusione Sociale e Migrazioni",
            "Delegato Operazioni, Emergenza e Soccorsi",
            "Delegato Principi e Valori",
            "Delegato Cooperazione Internazionale",
            "Delegato Organizzazione, Innovazione, Sviluppo e Volontariato",
            "Referente Formazione",
        ]

        for _, row in df.iterrows():
            vals = [str(v).strip() if str(v).strip() != "nan" else "" for v in row.values]

            comitato = vals[0] if len(vals) > 0 else ""
            presidente = vals[1] if len(vals) > 1 else ""
            cellulare = vals[2] if len(vals) > 2 else ""

            if not comitato:
                continue

            # Pulisci numeri telefono (rimuovi formule Excel tipo =+39...)
            cellulare = cellulare.replace("=+", "+").replace("=", "")

            parts = [f"Il Comitato CRI di {comitato} ha come presidente {presidente}."]
            if cellulare:
                parts.append(f"Telefono presidente: {cellulare}.")

            # Delegati: partono dalla colonna 4, gruppi di 3 (nome, mail, tel)
            col_start = 4
            for i, deleg_nome in enumerate(delegati_nomi):
                base = col_start + (i * 3)
                nome = vals[base] if len(vals) > base else ""
                mail = vals[base + 1] if len(vals) > base + 1 else ""
                tel = vals[base + 2] if len(vals) > base + 2 else ""

                if nome:
                    tel = tel.replace("=+", "+").replace("=", "")
                    deleg_parts = [f"{deleg_nome}: {nome}"]
                    if mail:
                        deleg_parts.append(f"email {mail}")
                    if tel:
                        deleg_parts.append(f"tel {tel}")
                    parts.append(", ".join(deleg_parts) + ".")

            rows.append(" ".join(parts))
        return rows

    def _format_generic(self, df: pd.DataFrame) -> list[str]:
        """Fallback generico: ogni riga come elenco colonna: valore."""
        rows = []
        columns = list(df.columns)
        for _, row in df.iterrows():
            parts = []
            for col in columns:
                val = str(row[col]).strip()
                if val and val != "nan":
                    parts.append(f"{col}: {val}")
            if parts:
                rows.append(". ".join(parts) + ".")
        return rows

    def _parse_with_llamaparse(self, filepath: Path) -> str:
        """Parsa con LlamaParse v2 (cost_effective) per PDF/DOCX/PPTX."""
        from llama_cloud_services import LlamaParse

        logger.info(f"📄 Parsing LlamaParse v2 (tier={self.config.llama_tier}): {filepath.name}")

        page_separator = "\n\n## PAGE_HEADER {pageNumber} ##\n\n"

        parser = LlamaParse(
            api_key=self.config.llama_api_key,
            result_type=self.config.llama_result_type,
            language=self.config.llama_language,
            num_workers=self.config.llama_num_workers,
            verbose=True,
            base_url="https://api.cloud.eu.llamaindex.ai",
            tier=self.config.llama_tier,
            version=self.config.llama_version,
            page_separator=page_separator,
            extract_printed_page_number=True,
        )

        start_time = time.time()

        try:
            documents = parser.load_data(str(filepath))
            content = "".join([doc.text for doc in documents])

            elapsed = time.time() - start_time
            logger.info(f"✅ Parsing completato: {filepath.name} ({elapsed:.2f}s, {len(content)} chars)")
            return content

        except Exception as e:
            logger.error(f"❌ Errore parsing {filepath.name}: {e!s}")
            raise


# ============================================================================
# METADATA ENRICHMENT
# ============================================================================


class MetadataExtractor:
    @staticmethod
    def extract_year(filename: str) -> int | None:
        import re

        match = re.search(r"\b(19|20)\d{2}\b", filename)
        if match:
            return int(match.group(0))
        return None

    @staticmethod
    def infer_category(filename: str) -> str:
        fn = filename.lower()
        if "statuto" in fn or "atto_costitutivo" in fn:
            return "Statuto"
        if "codice_etico" in fn or "etico" in fn:
            return "Etica"
        if "dlgs" in fn or "decreto" in fn or "legge" in fn or "norme" in fn:
            return "Normativa"
        if "regolamento" in fn:
            return "Regolamento"
        if "linee_guida" in fn or "manuale" in fn or "strategia" in fn:
            return "Linee Guida"
        if "donator" in fn or "sangue" in fn:
            return "Sanitario/Donazioni"
        if "formazione" in fn:
            return "Formazione"
        if "volontar" in fn:
            return "Volontariato"
        if "delegati" in fn or "competenze" in fn:
            return "Organizzazione"
        return "Altro"


# ============================================================================
# PAGE-AWARE CHUNKER
# ============================================================================


class PageAwareChunker:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " "],
            keep_separator=True,
            length_function=len,
        )
        import re

        self.page_pattern = re.compile(r"## PAGE_HEADER (\d+) ##")

    def chunk_document(self, content: str, filename: str, source_path: str, file_hash: str) -> list[Document]:
        if not content.strip():
            logger.warning(f"📭 Contenuto vuoto per {filename}")
            return []

        parts = self.page_pattern.split(content)

        pages_content = []
        if parts[0].strip():
            pages_content.append({"page": 1, "text": parts[0]})

        i = 1
        while i < len(parts):
            try:
                p_num = int(parts[i])
                p_text = parts[i + 1]
                pages_content.append({"page": p_num, "text": p_text})
            except (ValueError, IndexError):
                pass
            i += 2

        logger.info(f"   🔗 Trovate {len(pages_content)} pagine fisiche.")

        final_chunks = []
        doc_type = self._get_doc_type(filename)
        year = MetadataExtractor.extract_year(filename)
        category = MetadataExtractor.infer_category(filename)
        processed_date = datetime.now().isoformat()

        for page in pages_content:
            page_num = page["page"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            page_doc = Document(page_content=page_text)
            raw_chunks = self.splitter.split_documents([page_doc])

            for sub_chunk in raw_chunks:
                text = sub_chunk.page_content.strip()
                if not text:
                    continue

                sub_chunk.metadata = {
                    "filename": filename,
                    "source": source_path,
                    "document_type": doc_type,
                    "file_hash": file_hash,
                    "page_number": page_num,
                    "year": year,
                    "category": category,
                    "processed_date": processed_date,
                    "ingest_version": "volontariato-1.0-cost_effective",
                }
                final_chunks.append(sub_chunk)

        optimized_chunks = self._optimize_chunks(final_chunks)

        total_chunks = len(optimized_chunks)
        for i, chunk in enumerate(optimized_chunks):
            text = chunk.page_content
            chunk.metadata["chunk_id"] = i + 1
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["chunk_title"] = text.split("\n")[0].lstrip("#").strip()[:60] + "..."
            chunk.metadata["char_count"] = len(text)
            chunk.metadata["word_count"] = len(text.split())
            chunk.metadata["position"] = (
                "start" if i < total_chunks * 0.2 else "end" if i > total_chunks * 0.8 else "middle"
            )

        logger.info(f"✂️  Chunking: {filename} → {len(optimized_chunks)} chunks")
        return optimized_chunks

    def _optimize_chunks(self, chunks: list[Document], min_chars: int = 400) -> list[Document]:
        if not chunks:
            return []
        merged = []
        buffer_doc = None

        for doc in chunks:
            text = doc.page_content.strip()
            if buffer_doc:
                new_text = buffer_doc.page_content + "\n\n" + text
                buffer_doc.page_content = new_text
                if len(new_text) >= min_chars:
                    merged.append(buffer_doc)
                    buffer_doc = None
                continue
            if len(text) < min_chars:
                buffer_doc = doc
            else:
                merged.append(doc)

        if buffer_doc:
            if merged:
                prev = merged[-1]
                prev.page_content += "\n\n" + buffer_doc.page_content
            else:
                merged.append(buffer_doc)
        return merged

    def _get_doc_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mapping = {
            ".pdf": "PDF",
            ".xlsx": "Excel",
            ".xls": "Excel",
            ".doc": "Word",
            ".docx": "Word",
            ".pptx": "PowerPoint",
        }
        return mapping.get(ext, "Unknown")


# ============================================================================
# HYBRID EMBEDDING MANAGER
# ============================================================================


class HybridEmbeddingManager:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.dense_model = OpenAIEmbeddings(model=config.dense_model, openai_api_key=config.openai_api_key)
        self.sparse_model = None
        if config.enable_sparse:
            self.sparse_model = SparseTextEmbedding(model_name=config.sparse_model)

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
        logger.info("  📊 Generazione dense embeddings...")
        dense_embeddings = self.dense_model.embed_documents(texts)

        sparse_embeddings = []
        if self.sparse_model and self.config.enable_sparse:
            logger.info("  🔑 Generazione sparse embeddings (BM25)...")
            sparse_results = list(self.sparse_model.embed(texts))
            for sparse_emb in sparse_results:
                sparse_embeddings.append(
                    SparseVector(indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist())
                )
        return dense_embeddings, sparse_embeddings


# ============================================================================
# QDRANT HYBRID MANAGER
# ============================================================================


class QdrantHybridManager:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=180.0)
        self.embedding_manager = HybridEmbeddingManager(config)

    def ensure_collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.qdrant_collection not in collection_names:
                logger.info(f"🆕 Creazione collection: {self.config.qdrant_collection}")

                vectors_config = {
                    self.config.dense_vector_name: VectorParams(
                        size=self.config.dense_dimensions, distance=Distance.COSINE
                    )
                }
                sparse_vectors_config = None
                if self.config.enable_sparse:
                    sparse_vectors_config = {self.config.sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF)}

                self.client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
                self._create_payload_indexes()

            else:
                logger.info(f"✅ Collection esistente: {self.config.qdrant_collection}")
                self._create_payload_indexes()

            return True
        except Exception as e:
            logger.error(f"❌ Errore collection: {e!s}")
            return False

    def _create_payload_indexes(self):
        logger.info("📑 Creazione indici payload per Qdrant 1.16+...")

        simple_indexes = [
            ("metadata.filename", PayloadSchemaType.KEYWORD),
            ("metadata.category", PayloadSchemaType.KEYWORD),
            ("metadata.document_type", PayloadSchemaType.KEYWORD),
            ("metadata.file_hash", PayloadSchemaType.KEYWORD),
            ("metadata.year", PayloadSchemaType.INTEGER),
            ("metadata.page_number", PayloadSchemaType.INTEGER),
            ("filename", PayloadSchemaType.KEYWORD),
            ("category", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in simple_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.config.qdrant_collection,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=True,
                )
                logger.info(f"   ✅ Indice: {field_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"   ℹ️  Esistente: {field_name}")
                else:
                    logger.warning(f"   ⚠️  Errore indice {field_name}: {e!s}")

        # Text index con stemmer italiano
        logger.info("📚 Creazione TEXT INDEX con stemmer italiano...")
        try:
            self.client.create_payload_index(
                collection_name=self.config.qdrant_collection,
                field_name="page_content",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                    ascii_folding=True,
                    stemmer=SnowballParams(type="snowball", language=SnowballLanguage.ITALIAN),
                    stopwords=StopwordsSet(languages=[Language.ITALIAN]),
                    phrase_matching=True,
                ),
                wait=True,
            )
            logger.info("   ✅ TEXT INDEX italiano configurato")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("   ℹ️  TEXT INDEX già esistente")
            else:
                logger.warning(f"   ⚠️  Errore TEXT INDEX: {e!s}")

    def _get_filename_variants(self, filename: str) -> list[str]:
        basename = Path(filename).stem
        extensions = [".md", ".pdf", ".txt", ".xlsx", ".xls", ".doc", ".docx", ".pptx"]
        variants = [basename + ext for ext in extensions]
        variants.append(filename)
        return list(set(variants))

    def count_chunks_by_filename(self, filename: str) -> int:
        try:
            variants = self._get_filename_variants(filename)
            conditions = []
            for variant in variants:
                conditions.append(
                    models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=variant))
                )
                conditions.append(models.FieldCondition(key="filename", match=models.MatchValue(value=variant)))
            result = self.client.count(
                collection_name=self.config.qdrant_collection, count_filter=models.Filter(should=conditions)
            )
            return result.count
        except Exception:
            return 0

    def delete_by_filename(self, filename: str) -> int:
        count = self.count_chunks_by_filename(filename)
        if count == 0:
            return 0
        logger.info(f"🗑️ Eliminazione {count} chunks per: {Path(filename).stem}.*")
        try:
            variants = self._get_filename_variants(filename)
            conditions = []
            for variant in variants:
                conditions.append(
                    models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=variant))
                )
                conditions.append(models.FieldCondition(key="filename", match=models.MatchValue(value=variant)))
            self.client.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=models.FilterSelector(filter=models.Filter(should=conditions)),
            )
            return count
        except Exception as e:
            logger.error(f"❌ Errore eliminazione: {e!s}")
            return 0

    def _upsert_with_retry(self, points: list, max_retries: int = 3):
        """Upsert con retry automatico per timeout transienti."""
        for attempt in range(max_retries):
            try:
                self.client.upsert(collection_name=self.config.qdrant_collection, points=points)
                return
            except Exception:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"   ⚠️ Tentativo {attempt + 1} fallito, retry tra {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def upsert_chunks_hybrid(self, chunks: list[Document]) -> bool:
        if not chunks:
            return False

        logger.info(f"📤 Upsert {len(chunks)} chunks (hybrid)...")
        start_time = time.time()

        try:
            batch_size = self.config.batch_size
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                batch = chunks[start_idx:end_idx]

                logger.info(f"  📦 Batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks)")

                texts = [chunk.page_content for chunk in batch]
                metadatas = [chunk.metadata for chunk in batch]

                dense_embeddings, sparse_embeddings = self.embedding_manager.embed_documents(texts)

                points = []
                for i, (text, metadata, dense_emb) in enumerate(zip(texts, metadatas, dense_embeddings)):
                    point_id = self._generate_point_id(metadata["filename"], metadata["chunk_id"])

                    vectors = {self.config.dense_vector_name: dense_emb}
                    if self.config.enable_sparse and sparse_embeddings:
                        vectors[self.config.sparse_vector_name] = sparse_embeddings[i]

                    points.append(
                        PointStruct(id=point_id, vector=vectors, payload={"page_content": text, "metadata": metadata})
                    )

                self._upsert_with_retry(points)

            elapsed = time.time() - start_time
            logger.info(f"✅ Inseriti {len(chunks)} chunks in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Errore upsert: {e!s}")
            import traceback

            traceback.print_exc()
            return False

    def _generate_point_id(self, filename: str, chunk_id: int) -> int:
        content = f"{filename}:{chunk_id}"
        hash_bytes = hashlib.md5(content.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    def get_stats(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.config.qdrant_collection)
            return {
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except:
            return {}


# ============================================================================
# PIPELINE
# ============================================================================


class HybridIngestPipeline:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.parser = DocumentParser(config)
        self.chunker = PageAwareChunker(config)
        self.qdrant = QdrantHybridManager(config)

    def get_file_hash(self, filepath: str) -> str:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_file(self, filepath: str, mode: str = "replace", dry_run: bool = False) -> dict[str, Any]:
        filepath = Path(filepath)
        filename = filepath.name
        result = {"filename": filename, "status": "pending", "chunks_deleted": 0, "chunks_inserted": 0, "error": None}

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📄 Processing: {filename}")
        logger.info(f"{'=' * 60}")

        if not filepath.exists():
            result["status"] = "error"
            result["error"] = "File non trovato"
            return result

        ext = filepath.suffix.lower()
        if ext not in self.config.supported_extensions:
            result["status"] = "error"
            result["error"] = f"Estensione non supportata: {ext}"
            return result

        existing_count = self.qdrant.count_chunks_by_filename(filename)
        if existing_count > 0:
            if mode == "add-only":
                result["status"] = "skipped"
                result["error"] = f"Documento già esistente ({existing_count} chunks)"
                return result
            if dry_run:
                logger.info(f"[DRY RUN] Eliminerebbe {existing_count} chunks")
            else:
                result["chunks_deleted"] = self.qdrant.delete_by_filename(filename)

        if dry_run:
            result["status"] = "dry_run"
            return result

        try:
            parsed = self.parser.parse_file(str(filepath))
            file_hash = self.get_file_hash(str(filepath))

            if isinstance(parsed, list):
                # Excel: ogni elemento è già un chunk pronto (1 riga = 1 chunk)
                if not parsed:
                    result["status"] = "error"
                    result["error"] = "Contenuto vuoto"
                    return result

                doc_type = "Excel"
                year = MetadataExtractor.extract_year(filename)
                category = MetadataExtractor.infer_category(filename)
                processed_date = datetime.now().isoformat()

                chunks = []
                for i, text in enumerate(parsed):
                    doc = Document(
                        page_content=text,
                        metadata={
                            "filename": filename,
                            "source": str(filepath.absolute()),
                            "document_type": doc_type,
                            "file_hash": file_hash,
                            "page_number": 1,
                            "year": year,
                            "category": category,
                            "processed_date": processed_date,
                            "ingest_version": "volontariato-1.0-pandas",
                            "chunk_id": i + 1,
                            "total_chunks": len(parsed),
                            "chunk_title": text[:60] + "...",
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "position": (
                                "start" if i < len(parsed) * 0.2
                                else "end" if i > len(parsed) * 0.8
                                else "middle"
                            ),
                        },
                    )
                    chunks.append(doc)

                logger.info(f"📊 Excel → {len(chunks)} chunks (1 riga = 1 chunk)")
            else:
                # PDF/DOCX: flusso standard LlamaParse + chunker
                content = parsed
                if not content.strip():
                    result["status"] = "error"
                    result["error"] = "Contenuto vuoto"
                    return result

                chunks = self.chunker.chunk_document(
                    content, filename, str(filepath.absolute()), file_hash
                )

            if not chunks:
                result["status"] = "error"
                result["error"] = "Nessun chunk generato"
                return result

            if self.qdrant.upsert_chunks_hybrid(chunks):
                result["status"] = "success"
                result["chunks_inserted"] = len(chunks)
            else:
                result["status"] = "error"
                result["error"] = "Errore upsert"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ Errore: {e!s}")

        return result

    def run(self, input_path: str, mode: str = "replace", dry_run: bool = False) -> list[dict[str, Any]]:
        if not self.qdrant.ensure_collection_exists():
            return []

        path = Path(input_path)
        if path.is_file():
            return [self.process_file(str(path), mode, dry_run)]
        elif path.is_dir():
            files = []
            for ext in self.config.supported_extensions:
                files.extend(path.glob(f"*{ext}"))
                files.extend(path.glob(f"*{ext.upper()}"))
            files = sorted(set(files))
            if not files:
                logger.warning(f"⚠️ Nessun file trovato in: {path}")
                return []
            logger.info(f"\n📁 Trovati {len(files)} file:")
            for f in files:
                logger.info(f"   - {f.name}")
            return [self.process_file(str(f), mode, dry_run) for f in files]
        return []


# ============================================================================
# CLI
# ============================================================================


def print_summary(results: list[dict[str, Any]]):
    if not results:
        return
    print("\n" + "=" * 60)
    print(f"📊 RIEPILOGO - Collection: {COLLECTION_NAME}")
    print("=" * 60)

    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    total_deleted = sum(r["chunks_deleted"] for r in results)
    total_inserted = sum(r["chunks_inserted"] for r in results)

    print(f"\n📁 File processati: {len(results)}")
    print(f"   ✅ Successo: {len(success)}")
    print(f"   ❌ Errori: {len(errors)}")
    print(f"   ⏭️ Saltati: {len(skipped)}")
    print(f"\n📦 Chunks: 🗑️ {total_deleted} eliminati | 📤 {total_inserted} inseriti")

    if errors:
        print("\n❌ ERRORI:")
        for r in errors:
            print(f"   - {r['filename']}: {r['error']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=f"Ingest Pipeline → Collection: {COLLECTION_NAME} (LlamaParse cost_effective)"
    )
    parser.add_argument("input_path", help="File o cartella da processare")
    parser.add_argument("--mode", choices=["replace", "add-only"], default="replace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sparse", action="store_true", help="Disabilita sparse vectors")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = HybridConfig()
    config.chunk_size = args.chunk_size
    config.chunk_overlap = args.chunk_overlap
    config.enable_sparse = not args.no_sparse

    missing = []
    if not config.llama_api_key:
        missing.append("LLAMA_CLOUD_API_KEY")
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not config.qdrant_url:
        missing.append("QDRANT_URL")
    if not config.qdrant_api_key:
        missing.append("QDRANT_API_KEY")

    if missing:
        logger.error(f"❌ Variabili mancanti: {', '.join(missing)}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"🚀 INGEST → {COLLECTION_NAME}")
    print("=" * 60)
    print(f"📂 Input: {args.input_path}")
    print(f"⚙️ Mode: {args.mode} | Dry run: {args.dry_run}")
    print(f"📏 Chunk: {config.chunk_size} chars, overlap {config.chunk_overlap}")
    print(f"🤖 LlamaParse: tier={config.llama_tier}, version={config.llama_version}")
    print(f"🔮 Dense: {config.dense_model} ({config.dense_dimensions}d)")
    print(f"🔑 Sparse: {config.sparse_model if config.enable_sparse else 'DISABLED'}")
    print(f"🗄️ Collection: {config.qdrant_collection}")
    print("=" * 60 + "\n")

    pipeline = HybridIngestPipeline(config)
    start_time = time.time()
    results = pipeline.run(args.input_path, args.mode, args.dry_run)
    elapsed = time.time() - start_time

    print_summary(results)
    print(f"⏱️ Tempo totale: {elapsed:.2f}s\n")

    if not args.dry_run and results:
        stats = pipeline.qdrant.get_stats()
        if stats:
            print(f"📊 Collection '{config.qdrant_collection}': {stats.get('points_count', 'N/A')} punti")

    errors = [r for r in results if r["status"] == "error"]
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
