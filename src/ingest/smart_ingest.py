"""
Smart Ingest Pipeline v3.1 - HYBRID SEARCH + SAFE EXCEL
Pipeline per parsing, chunking e indicizzazione documenti in Qdrant.

CHANGELOG v3.1:
- Sostituito LlamaParse con PANDAS per i file Excel (.xlsx, .xls)
  per evitare "hallucinations" e garantire precisione al 100% sui dati tabellari.
- I PDF/DOC usano ancora LlamaParse (Page-Aware).

Usage:
    python smart_ingest_hybrid.py new_docs/              # Processa cartella
    python smart_ingest_hybrid.py doc.pdf                # Singolo file
    python smart_ingest_hybrid.py new_docs/ --dry-run    # Preview
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

# Gestione Excel Deterministica
import pandas as pd
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding  # Per BM25
from langchain_core.documents import Document

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document processing
from llama_cloud_services import LlamaParse

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVector, SparseVectorParams, VectorParams

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAZIONE
# ============================================================================


@dataclass
class HybridConfig:
    """Configurazione centralizzata per pipeline ibrida."""

    # === LlamaParse ===
    llama_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY", ""))
    llama_result_type: str = "markdown"
    llama_language: str = "it"
    llama_num_workers: int = 4
    use_premium_mode: bool = False

    # === Chunking ===
    chunk_size: int = 2000  # ~500 tokens
    chunk_overlap: int = 300  # 15% overlap

    # === Dense Embeddings (Semantic) ===
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    dense_model: str = "text-embedding-3-large"
    dense_dimensions: int = 3072
    dense_vector_name: str = "dense"

    # === Sparse Embeddings (BM25 Keyword) ===
    enable_sparse: bool = True
    sparse_model: str = "Qdrant/bm25"
    sparse_vector_name: str = "sparse"

    # === Qdrant ===
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", ""))

    # === Processing ===
    batch_size: int = 32

    # === Estensioni supportate ===
    supported_extensions: tuple = (".pdf", ".xlsx", ".xls", ".doc", ".docx", ".pptx")


# ============================================================================
# DOCUMENT PARSER (V3.1 - PANDAS HYBRID)
# ============================================================================


class DocumentParser:
    """
    Parser Ibrido:
    - LlamaParse: per PDF, DOCX, PPTX (struttura complessa, immagini, layout)
    - Pandas: per EXCEL (dati tabellari puri, zero allucinazioni)
    """

    def __init__(self, config: HybridConfig):
        self.config = config

    def parse_file(self, filepath: str) -> str:
        """Sceglie la strategia di parsing in base all'estensione."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in self.config.supported_extensions:
            raise ValueError(f"Estensione non supportata: {ext}")

        # === STRATEGIA A: EXCEL DETERMINISTICO (PANDAS) ===
        if ext in [".xlsx", ".xls"]:
            logger.info(f"📊 Parsing Excel (Mode: SAFE PANDAS): {filepath.name}")
            return self._parse_excel_safe(str(filepath))

        # === STRATEGIA B: LLAMAPARSE INTELLIGENTE (PDF, DOC, PPT) ===
        return self._parse_llamaparse(filepath)

    def _parse_excel_safe(self, filepath: str) -> str:
        """
        Legge Excel con Pandas riga per riga.
        Formatta ogni riga come testo semantico: 'Colonna: Valore | Colonna: Valore'.
        """
        start_time = time.time()
        text_output = []

        try:
            # Legge tutte le sheet
            xls = pd.read_excel(filepath, sheet_name=None)

            page_counter = 1
            for sheet_name, df in xls.items():
                # Pulizia base
                df = df.dropna(how="all")  # Via righe vuote
                df = df.fillna("")  # Via NaN

                # Inietta HEADER COMPATIBILE con PageAwareChunker
                # Trattiamo ogni Sheet come una "Pagina"
                text_output.append(f"\n\n## PAGE_HEADER {page_counter} ##\n\n")
                text_output.append(f"# Foglio Excel: {sheet_name}\n")

                # Conversione Righe
                records = df.to_dict(orient="records")
                for i, row in enumerate(records):
                    row_text = []
                    for col, val in row.items():
                        s_val = str(val).strip()
                        if not s_val:
                            continue  # Salta celle vuote

                        # Formato: "Intestazione: Valore"
                        clean_col = str(col).strip()
                        row_text.append(f"{clean_col}: {s_val}")

                    if row_text:
                        # Unisce la riga
                        line = " | ".join(row_text)
                        text_output.append(f"- Riga {i + 1}: {line}")

                page_counter += 1

            content = "\n".join(text_output)
            elapsed = time.time() - start_time
            logger.info(f"✅ Parsing Excel completato: {len(content)} chars ({elapsed:.2f}s)")
            return content

        except Exception as e:
            logger.error(f"❌ Errore Pandas Excel su {filepath}: {e!s}")
            raise

    def _parse_llamaparse(self, filepath: Path) -> str:
        """Logica standard LlamaParse per documenti non strutturati."""
        use_premium = self.config.use_premium_mode
        mode_str = "PREMIUM 💎" if use_premium else "FAST ⚡"

        logger.info(f"📄 Parsing LlamaParse ({mode_str}): {filepath.name}")

        page_separator = "\n\n## PAGE_HEADER {pageNumber} ##\n\n"

        parser = LlamaParse(
            api_key=self.config.llama_api_key,
            result_type=self.config.llama_result_type,
            language=self.config.llama_language,
            num_workers=self.config.llama_num_workers,
            verbose=True,
            base_url="https://api.cloud.eu.llamaindex.ai",
            premium_mode=use_premium,
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
            logger.error(f"❌ Errore LlamaParse {filepath.name}: {e!s}")
            raise


# ============================================================================
# METADATA ENRICHMENT
# ============================================================================


class MetadataExtractor:
    """Estrae metadati avanzati dal filename."""

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
        if "delegati" in fn or "competenze" in fn:
            return "Organizzazione"
        return "Altro"


# ============================================================================
# PAGE-AWARE CHUNKER
# ============================================================================


class PageAwareChunker:
    """Chunker che rispetta i numeri di pagina (o Sheet Excel) e ottimizza i micro-chunk."""

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
            logger.warning(f"🔔 Contenuto vuoto per {filename}")
            return []

        # 1. SPLIT BY PAGE HEADER
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

        logger.info(f"   🔗 Identificate {len(pages_content)} sezioni logiche (Pagine/Sheets).")

        # 2. CHUNK EACH PAGE
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
                    "ingest_version": "3.1-hybrid-pandas",
                }
                final_chunks.append(sub_chunk)

        # 3. OPTIMIZE
        optimized_chunks = self._optimize_chunks(final_chunks)

        # 4. FINALIZE METADATA
        total_chunks = len(optimized_chunks)
        for i, chunk in enumerate(optimized_chunks):
            chunk.metadata["chunk_id"] = i + 1
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["chunk_title"] = chunk.page_content.split("\n")[0][:60] + "..."
            chunk.metadata["char_count"] = len(chunk.page_content)

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
        mapping = {".pdf": "PDF", ".xlsx": "Excel", ".xls": "Excel", ".doc": "Word", ".docx": "Word"}
        return mapping.get(ext, "Unknown")


# ============================================================================
# HYBRID EMBEDDING MANAGER
# ============================================================================


class HybridEmbeddingManager:
    """Gestisce embeddings dense (OpenAI) + sparse (BM25)."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.dense_model = OpenAIEmbeddings(model=config.dense_model, openai_api_key=config.openai_api_key)
        self.sparse_model = None
        if config.enable_sparse:
            self.sparse_model = SparseTextEmbedding(model_name=config.sparse_model)

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
        logger.info("   📊 Generazione dense embeddings...")
        dense_embeddings = self.dense_model.embed_documents(texts)

        sparse_embeddings = []
        if self.sparse_model and self.config.enable_sparse:
            logger.info("   🔑 Generazione sparse embeddings (BM25)...")
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
    """Gestisce collection e upsert Hybrid."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=180.0)
        self.embedding_manager = HybridEmbeddingManager(config)

    def ensure_collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.qdrant_collection not in collection_names:
                logger.info(f"🆕 Creazione collection HYBRID: {self.config.qdrant_collection}")

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
            return True
        except Exception as e:
            logger.error(f"❌ Errore collection: {e!s}")
            return False

    def _get_filename_variants(self, filename: str) -> list[str]:
        basename = Path(filename).stem
        extensions = [".md", ".pdf", ".txt", ".xlsx", ".xls", ".doc", ".docx"]
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
        except Exception:
            return 0

    def upsert_chunks_hybrid(self, chunks: list[Document]) -> bool:
        if not chunks:
            return False

        logger.info(f"📤 Upsert {len(chunks)} chunks (hybrid)...")
        try:
            batch_size = self.config.batch_size
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                batch = chunks[start_idx:end_idx]

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

                self.client.upsert(collection_name=self.config.qdrant_collection, points=points)
            return True
        except Exception as e:
            logger.error(f"❌ Errore upsert: {e!s}")
            return False

    def _generate_point_id(self, filename: str, chunk_id: int) -> int:
        content = f"{filename}:{chunk_id}"
        hash_bytes = hashlib.md5(content.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    def get_stats(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.config.qdrant_collection)
            return {"points_count": info.points_count, "status": str(info.status)}
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

        logger.info(f"\n{'=' * 60}\n📄 Processing: {filename}\n{'=' * 60}")

        existing_count = self.qdrant.count_chunks_by_filename(filename)
        if existing_count > 0:
            if mode == "add-only":
                result["status"] = "skipped"
                return result
            if dry_run:
                logger.info(f"[DRY RUN] Eliminerebbe {existing_count} chunks")
            else:
                result["chunks_deleted"] = self.qdrant.delete_by_filename(filename)

        if dry_run:
            result["status"] = "dry_run"
            return result

        try:
            # 1. Parse (Hybrid LlamaParse / Pandas)
            content = self.parser.parse_file(str(filepath))
            if not content.strip():
                raise ValueError("Contenuto vuoto")

            # 2. Chunk
            file_hash = self.get_file_hash(str(filepath))
            chunks = self.chunker.chunk_document(content, filename, str(filepath.absolute()), file_hash)

            # 3. Upsert
            if self.qdrant.upsert_chunks_hybrid(chunks):
                result["status"] = "success"
                result["chunks_inserted"] = len(chunks)
            else:
                raise ValueError("Upsert fallito")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ Errore: {e!s}")

        return result

    def run(self, input_path: str, mode: str = "replace", dry_run: bool = False):
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
            return [self.process_file(str(f), mode, dry_run) for f in set(files)]
        return []


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Smart Ingest v3.1 (Hybrid + Excel Safe)")
    parser.add_argument("input_path", help="File o cartella")
    parser.add_argument("--mode", choices=["replace", "add-only"], default="replace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sparse", action="store_true")
    parser.add_argument("--premium", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2000)

    args = parser.parse_args()

    config = HybridConfig()
    config.chunk_size = args.chunk_size
    config.enable_sparse = not args.no_sparse
    config.use_premium_mode = args.premium

    if not all([config.llama_api_key, config.openai_api_key, config.qdrant_url, config.qdrant_api_key]):
        logger.error("❌ Mancano variabili d'ambiente (.env)")
        sys.exit(1)

    pipeline = HybridIngestPipeline(config)
    results = pipeline.run(args.input_path, args.mode, args.dry_run)

    print(f"\n✅ Completato. Files: {len(results)}")
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"❌ {len(errors)} Errori riscontrati.")
        sys.exit(1)


if __name__ == "__main__":
    main()
