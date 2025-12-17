"""
Smart Ingest Pipeline v3.0 - HYBRID SEARCH EDITION
Pipeline per parsing, chunking e indicizzazione documenti in Qdrant
con supporto COMPLETO per ricerca ibrida (semantica + keyword).

Features:
- Dense vectors (OpenAI text-embedding-3-large) per ricerca semantica
- Sparse vectors (BM25 via FastEmbed) per ricerca keyword
- Reciprocal Rank Fusion (RRF) per combinare risultati
- Chunking semantico adattivo
- Supporto reranking con ColBERT (opzionale)
- Metadati arricchiti per filtering avanzato

Usage:
    python smart_ingest_hybrid.py new_docs/              # Processa cartella
    python smart_ingest_hybrid.py doc.pdf                # Singolo file
    python smart_ingest_hybrid.py new_docs/ --dry-run    # Preview
    python smart_ingest_hybrid.py new_docs/ --no-sparse  # Solo dense
"""

import os
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv

# Document processing
from llama_cloud_services import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings
from langchain_openai import OpenAIEmbeddings
from fastembed import SparseTextEmbedding  # Per BM25

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance, Modifier,
    PointStruct, SparseVector, NamedVector, NamedSparseVector
)

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
    use_premium_mode: bool = False  # New: Premium mode flag
    
    # === Chunking ===
    # Best practice NVIDIA 2024: 512-1024 tokens ottimale
    # Per italiano ~4 chars/token, quindi 500 tokens ≈ 2000 chars
    # Query factoid: 256-512 tokens | Query analitiche: 1024+ tokens
    chunk_size: int = 2000   # ~500 tokens - nel range ottimale NVIDIA
    chunk_overlap: int = 300  # 15% overlap - ottimale secondo benchmark NVIDIA
    
    # === Dense Embeddings (Semantic) ===
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    dense_model: str = "text-embedding-3-large"
    dense_dimensions: int = 3072
    dense_vector_name: str = "dense"  # Nome vettore in Qdrant
    
    # === Sparse Embeddings (BM25 Keyword) ===
    enable_sparse: bool = True
    sparse_model: str = "Qdrant/bm25"  # Modello BM25 via FastEmbed
    sparse_vector_name: str = "sparse"  # Nome vettore sparse in Qdrant
    
    # === Qdrant ===
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", ""))
    
    # === Processing ===
    batch_size: int = 32  # Ridotto per gestire sia dense che sparse
    
    # === Estensioni supportate ===
    supported_extensions: tuple = (".pdf", ".xlsx", ".xls", ".doc", ".docx", ".pptx")


# ============================================================================
# DOCUMENT PARSER (invariato dal tuo script)
# ============================================================================

class DocumentParser:
    """Parser documenti tramite LlamaParse (Page-Aware)."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
    def parse_file(self, filepath: str) -> str:
        """Parsa un file e restituisce contenuto Markdown con header di pagina espliciti."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        ext = filepath.suffix.lower()
        if ext not in self.config.supported_extensions:
            raise ValueError(f"Estensione non supportata: {ext}")
        
        # Determina modalitÃ 
        is_excel = ext in [".xlsx", ".xls"]
        use_premium = self.config.use_premium_mode or is_excel
        
        mode_str = "PREMIUM ðŸ’Ž" if use_premium else "FAST âš¡"
        if is_excel and not self.config.use_premium_mode:
             mode_str += " (Auto-Excel)"
             
        logger.info(f"ðŸ“„ Parsing ({mode_str}): {filepath.name}")
        
        # PAGE-AWARE TRICK:
        # Usiamo un separatore personalizzato che include il numero di pagina.
        # {pageNumber} viene sostituito da LlamaParse con l'indice.
        page_separator = "\n\n## PAGE_HEADER {pageNumber} ##\n\n"
        
        parser = LlamaParse(
            api_key=self.config.llama_api_key,
            result_type=self.config.llama_result_type,
            language=self.config.llama_language,
            num_workers=self.config.llama_num_workers,
            verbose=True,
            base_url="https://api.cloud.eu.llamaindex.ai",
            premium_mode=use_premium,
            page_separator=page_separator,  # <--- MAGIC HERE: Inietta il numero di pagina
            extract_printed_page_number=True, # Prova anche a estrarre numero stampato
        )
        
        start_time = time.time()
        
        try:
            documents = parser.load_data(str(filepath))
            
            # Una volta ottenuto il testo, per sicurezza uniamo tutto.
            # LlamaParse inserisce il page_separator nel testo dei documenti.
            content = ""
            for doc in documents:
                content += doc.text
            
            elapsed = time.time() - start_time
            logger.info(f"âœ… Parsing completato: {filepath.name} ({elapsed:.2f}s, {len(content)} chars)")
            
            return content
            
        except Exception as e:
            logger.error(f"â Œ Errore parsing {filepath.name}: {str(e)}")
            raise


# ============================================================================
# METADATA ENRICHMENT - NEW 2025 READY
# ============================================================================

class MetadataExtractor:
    """Estrae metadati avanzati dal filename."""
    
    @staticmethod
    def extract_year(filename: str) -> Optional[int]:
        """Estrae anno (4 cifre) dal filename. Es: 'Regolamento_2017.pdf' -> 2017"""
        import re
        match = re.search(r'\b(19|20)\d{2}\b', filename)
        if match:
            return int(match.group(0))
        return None

    @staticmethod
    def infer_category(filename: str) -> str:
        """Deduce categoria da keyword nel filename."""
        fn = filename.lower()
        
        # Mappa prioritaria
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
        if "donator" in fn or "sangue" in fn: # donatore, donatori
            return "Sanitario/Donazioni"
        if "formazione" in fn:
            return "Formazione"
        if "delegati" in fn or "competenze" in fn:
            return "Organizzazione"
            
        return "Altro"


# ============================================================================
# PAGE-AWARE CHUNKER (Optimized)
# ============================================================================

class PageAwareChunker:
    """
    Chunker che rispetta i numeri di pagina e ottimizza i micro-chunk.
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
        # Splitter semantico per il testo all'interno delle pagine
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " "],
            keep_separator=True,
            length_function=len,
        )
        
        import re
        self.page_pattern = re.compile(r"## PAGE_HEADER (\d+) ##")

    def chunk_document(
        self,
        content: str,
        filename: str,
        source_path: str,
        file_hash: str
    ) -> List[Document]:
        """Divide documento preservando numeri di pagina e ottimizzando la dimensione."""
        
        if not content.strip():
            logger.warning(f"ðŸ”” Contenuto vuoto per {filename}")
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
                p_text = parts[i+1] # il testo segue il numero
                pages_content.append({"page": p_num, "text": p_text})
            except (ValueError, IndexError):
                pass
            i += 2
            
        logger.info(f"   ðŸ”— Trovate {len(pages_content)} pagine fisiche.")
        
        # 2. CHUNK EACH PAGE (Raw)
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
                
                # METADATA DI BASE (senza ID definitivi)
                sub_chunk.metadata = {
                    "filename": filename,
                    "source": source_path,
                    "document_type": doc_type,
                    "file_hash": file_hash,
                    "page_number": page_num, # <--- KEY FEATURE
                    "year": year,
                    "category": category,
                    "processed_date": processed_date,
                    "ingest_version": "4.1-optimized",
                }
                final_chunks.append(sub_chunk)

        # 3. OPTIMIZE: MERGE MICRO-CHUNKS
        optimized_chunks = self._optimize_chunks(final_chunks)
        
        # 4. FINALIZE METADATA (IDs, Titles, Stats)
        total_chunks = len(optimized_chunks)
        for i, chunk in enumerate(optimized_chunks):
            text = chunk.page_content
            chunk.metadata["chunk_id"] = i + 1
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["chunk_title"] = (text.split('\n')[0].lstrip('#').strip()[:60] + '...')
            chunk.metadata["char_count"] = len(text)
            chunk.metadata["word_count"] = len(text.split())
            chunk.metadata["position"] = "start" if i < total_chunks * 0.2 else \
                                         "end" if i > total_chunks * 0.8 else "middle"
                
        logger.info(f"âœ‚ï¸  Chunking: {filename} â†’ {len(optimized_chunks)} chunks (Page-Aware + Optimized)")
        return optimized_chunks

    def _optimize_chunks(self, chunks: List[Document], min_chars: int = 400) -> List[Document]:
        """
        Fonde i micro-chunks (sotto min_chars) con il chunk successivo.
        """
        if not chunks:
            return []
            
        merged = []
        buffer_doc = None
        
        for doc in chunks:
            text = doc.page_content.strip()
            
            if buffer_doc:
                # UNISCI al buffer
                new_text = buffer_doc.page_content + "\n\n" + text
                buffer_doc.page_content = new_text
                
                # Se ora Ã¨ abbastanza grande, flush
                if len(new_text) >= min_chars:
                    merged.append(buffer_doc)
                    buffer_doc = None
                continue
            
            # Se siamo qui, buffer vuoto. Controlliamo il doc corrente.
            if len(text) < min_chars:
                # Inizia buffer
                buffer_doc = doc
            else:
                # Doc valido
                merged.append(doc)
        
        # Flush eventuale buffer residuo (unendolo al precedente se esiste)
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
    """
    Gestisce generazione embeddings ibridi (dense + sparse).
    
    - Dense: OpenAI text-embedding-3-large (semantic understanding)
    - Sparse: BM25 via FastEmbed (keyword matching)
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
        # Dense embeddings (OpenAI)
        self.dense_model = OpenAIEmbeddings(
            model=config.dense_model,
            openai_api_key=config.openai_api_key
        )
        logger.info(f"🔮 Dense model: {config.dense_model} ({config.dense_dimensions} dims)")
        
        # Sparse embeddings (BM25)
        self.sparse_model = None
        if config.enable_sparse:
            self.sparse_model = SparseTextEmbedding(model_name=config.sparse_model)
            logger.info(f"🔑 Sparse model: {config.sparse_model}")
    
    def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[SparseVector]]:
        """
        Genera embeddings dense e sparse per una lista di testi.
        
        Returns:
            Tuple[dense_embeddings, sparse_embeddings]
        """
        # Dense embeddings
        logger.info(f"  📊 Generazione dense embeddings...")
        dense_embeddings = self.dense_model.embed_documents(texts)
        
        # Sparse embeddings (BM25)
        sparse_embeddings = []
        if self.sparse_model and self.config.enable_sparse:
            logger.info(f"  🔑 Generazione sparse embeddings (BM25)...")
            sparse_results = list(self.sparse_model.embed(texts))
            
            for sparse_emb in sparse_results:
                # Converti in formato Qdrant SparseVector
                sparse_embeddings.append(
                    SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    )
                )
        
        return dense_embeddings, sparse_embeddings


# ============================================================================
# QDRANT HYBRID MANAGER
# ============================================================================

class QdrantHybridManager:
    """
    Gestisce operazioni Qdrant con supporto hybrid search.
    
    Collection schema:
    - Named vector "dense": 3072 dims, COSINE
    - Named sparse vector "sparse": BM25 con IDF modifier
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=180.0
        )
        self.embedding_manager = HybridEmbeddingManager(config)
    
    def ensure_collection_exists(self) -> bool:
        """
        Crea o verifica collection con schema hybrid.
        
        IMPORTANTE: La collection deve avere sia vectors_config
        che sparse_vectors_config per supportare hybrid search.
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.qdrant_collection not in collection_names:
                logger.info(f"🆕 Creazione collection HYBRID: {self.config.qdrant_collection}")
                
                # Configurazione vettori dense
                vectors_config = {
                    self.config.dense_vector_name: VectorParams(
                        size=self.config.dense_dimensions,
                        distance=Distance.COSINE
                    )
                }
                
                # Configurazione vettori sparse (BM25 con IDF)
                sparse_vectors_config = None
                if self.config.enable_sparse:
                    sparse_vectors_config = {
                        self.config.sparse_vector_name: SparseVectorParams(
                            modifier=Modifier.IDF  # CRITICO: abilita IDF per BM25
                        )
                    }
                
                self.client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                
                logger.info(f"✅ Collection creata con schema:")
                logger.info(f"   - Dense: {self.config.dense_vector_name} ({self.config.dense_dimensions}d, COSINE)")
                if self.config.enable_sparse:
                    logger.info(f"   - Sparse: {self.config.sparse_vector_name} (BM25 + IDF)")
            else:
                logger.info(f"✅ Collection esistente: {self.config.qdrant_collection}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore collection: {str(e)}")
            return False
    
    def _get_filename_variants(self, filename: str) -> List[str]:
        """Genera varianti filename per matching."""
        basename = Path(filename).stem
        extensions = [".md", ".pdf", ".txt", ".xlsx", ".xls", ".doc", ".docx", ".pptx"]
        variants = [basename + ext for ext in extensions]
        variants.append(filename)
        return list(set(variants))
    
    def count_chunks_by_filename(self, filename: str) -> int:
        """Conta chunks esistenti per un filename."""
        try:
            variants = self._get_filename_variants(filename)
            conditions = []
            
            for variant in variants:
                # Formato nested (compatibilità)
                conditions.append(
                    models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchValue(value=variant)
                    )
                )
                # Formato flat
                conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=variant)
                    )
                )
            
            result = self.client.count(
                collection_name=self.config.qdrant_collection,
                count_filter=models.Filter(should=conditions)
            )
            return result.count
            
        except Exception as e:
            logger.error(f"❌ Errore conteggio: {str(e)}")
            return 0
    
    def delete_by_filename(self, filename: str) -> int:
        """Elimina tutti i chunks per un filename."""
        count = self.count_chunks_by_filename(filename)
        
        if count == 0:
            return 0
        
        basename = Path(filename).stem
        logger.info(f"🗑️ Eliminazione {count} chunks per: {basename}.*")
        
        try:
            variants = self._get_filename_variants(filename)
            conditions = []
            
            for variant in variants:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchValue(value=variant)
                    )
                )
                conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=variant)
                    )
                )
            
            self.client.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(should=conditions)
                )
            )
            return count
            
        except Exception as e:
            logger.error(f"❌ Errore eliminazione: {str(e)}")
            return 0
    
    def upsert_chunks_hybrid(self, chunks: List[Document]) -> bool:
        """
        Inserisce chunks con embeddings IBRIDI (dense + sparse).
        
        Ogni punto in Qdrant avrà:
        - vector["dense"]: embedding semantico (3072 dims)
        - vector["sparse"]: embedding BM25 keyword
        - payload: testo + metadati
        """
        if not chunks:
            logger.warning("⚠️ Nessun chunk da inserire")
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
                
                # Estrai testi
                texts = [chunk.page_content for chunk in batch]
                metadatas = [chunk.metadata for chunk in batch]
                
                # Genera embeddings ibridi
                dense_embeddings, sparse_embeddings = \
                    self.embedding_manager.embed_documents(texts)
                
                # Crea punti Qdrant
                points = []
                for i, (text, metadata, dense_emb) in enumerate(zip(texts, metadatas, dense_embeddings)):
                    
                    # Genera ID deterministico
                    point_id = self._generate_point_id(
                        metadata["filename"],
                        metadata["chunk_id"]
                    )
                    
                    # Prepara vettori
                    vectors = {
                        self.config.dense_vector_name: dense_emb
                    }
                    
                    # Aggiungi sparse se abilitato
                    if self.config.enable_sparse and sparse_embeddings:
                        vectors[self.config.sparse_vector_name] = sparse_embeddings[i]
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=vectors,
                        payload={
                            "page_content": text,
                            "metadata": metadata  # Nested per compatibilità
                        }
                    ))
                
                # Upsert batch
                self.client.upsert(
                    collection_name=self.config.qdrant_collection,
                    points=points
                )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Inseriti {len(chunks)} chunks in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore upsert: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_point_id(self, filename: str, chunk_id: int) -> int:
        """Genera ID deterministico per un chunk."""
        content = f"{filename}:{chunk_id}"
        hash_bytes = hashlib.md5(content.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder='big')
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche collection."""
        try:
            info = self.client.get_collection(self.config.qdrant_collection)
            return {
                "points_count": info.points_count,
                # "vectors_count": info.vectors_count, # Removed for compatibility
                "status": str(info.status),
                "config": {
                    "dense_vector": self.config.dense_vector_name,
                    "sparse_vector": self.config.sparse_vector_name if self.config.enable_sparse else None,
                }
            }
        except Exception as e:
            logger.error(f"❌ Errore stats: {str(e)}")
            return {}


# ============================================================================
# HYBRID INGEST PIPELINE
# ============================================================================

class HybridIngestPipeline:
    """Pipeline principale con supporto hybrid search."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.parser = DocumentParser(config)
        self.chunker = PageAwareChunker(config)
        self.qdrant = QdrantHybridManager(config)
    
    def get_file_hash(self, filepath: str) -> str:
        """Calcola hash MD5 del file."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def process_file(
        self,
        filepath: str,
        mode: str = "replace",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Processa un singolo file."""
        filepath = Path(filepath)
        filename = filepath.name
        
        result = {
            "filename": filename,
            "status": "pending",
            "chunks_deleted": 0,
            "chunks_inserted": 0,
            "error": None
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 Processing: {filename}")
        logger.info(f"{'='*60}")
        
        # Verifica file
        if not filepath.exists():
            result["status"] = "error"
            result["error"] = "File non trovato"
            return result
        
        ext = filepath.suffix.lower()
        if ext not in self.config.supported_extensions:
            result["status"] = "error"
            result["error"] = f"Estensione non supportata: {ext}"
            return result
        
        # Verifica esistenza
        existing_count = self.qdrant.count_chunks_by_filename(filename)
        
        if existing_count > 0:
            if mode == "add-only":
                result["status"] = "skipped"
                result["error"] = f"Documento già esistente ({existing_count} chunks)"
                logger.warning(f"⏭️ Skipped: {filename}")
                return result
            
            if dry_run:
                logger.info(f"[DRY RUN] Eliminerebbe {existing_count} chunks")
            else:
                result["chunks_deleted"] = self.qdrant.delete_by_filename(filename)
        
        if dry_run:
            logger.info(f"[DRY RUN] Indicizzerebbe: {filename}")
            result["status"] = "dry_run"
            return result
        
        try:
            # 1. Parse
            content = self.parser.parse_file(str(filepath))
            
            if not content.strip():
                result["status"] = "error"
                result["error"] = "Contenuto vuoto"
                return result
            
            # 2. Chunk
            file_hash = self.get_file_hash(str(filepath))
            chunks = self.chunker.chunk_document(
                content=content,
                filename=filename,
                source_path=str(filepath.absolute()),
                file_hash=file_hash
            )
            
            if not chunks:
                result["status"] = "error"
                result["error"] = "Nessun chunk generato"
                return result
            
            # 3. Upsert HYBRID
            success = self.qdrant.upsert_chunks_hybrid(chunks)
            
            if success:
                result["status"] = "success"
                result["chunks_inserted"] = len(chunks)
                logger.info(f"✅ SUCCESS: {filename} → {len(chunks)} chunks (hybrid)")
            else:
                result["status"] = "error"
                result["error"] = "Errore upsert"
            
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ Errore: {str(e)}")
            return result
    
    def process_folder(
        self,
        folder_path: str,
        mode: str = "replace",
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Processa tutti i file in una cartella."""
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            logger.error(f"❌ Cartella non valida: {folder}")
            return []
        
        # Trova file supportati
        files = []
        for ext in self.config.supported_extensions:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))
        
        files = sorted(set(files))
        
        if not files:
            logger.warning(f"⚠️ Nessun file trovato in: {folder}")
            return []
        
        logger.info(f"\n📁 Trovati {len(files)} file:")
        for f in files:
            logger.info(f"   - {f.name}")
        
        results = []
        for filepath in files:
            result = self.process_file(str(filepath), mode=mode, dry_run=dry_run)
            results.append(result)
        
        return results
    
    def run(
        self,
        input_path: str,
        mode: str = "replace",
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Esegue la pipeline."""
        # Verifica/Crea collection SEMPRE (anche in dry_run) per evitare errori 404 sui conteggi
        # e garantire che la collection esista come richiesto.
        if not self.qdrant.ensure_collection_exists():
            logger.error("❌ Impossibile creare/accedere collection")
            return []
        
        path = Path(input_path)
        
        if path.is_file():
            return [self.process_file(str(path), mode=mode, dry_run=dry_run)]
        elif path.is_dir():
            return self.process_folder(str(path), mode=mode, dry_run=dry_run)
        else:
            logger.error(f"❌ Percorso non valido: {input_path}")
            return []


# ============================================================================
# CLI
# ============================================================================

def print_summary(results: List[Dict[str, Any]]):
    """Stampa riepilogo."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("📊 RIEPILOGO")
    print("="*60)
    
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]
    dry_run = [r for r in results if r["status"] == "dry_run"]
    
    total_deleted = sum(r["chunks_deleted"] for r in results)
    total_inserted = sum(r["chunks_inserted"] for r in results)
    
    print(f"\n📁 File processati: {len(results)}")
    print(f"   ✅ Successo: {len(success)}")
    print(f"   ❌ Errori: {len(errors)}")
    print(f"   ⏭️ Saltati: {len(skipped)}")
    if dry_run:
        print(f"   🔍 Dry run: {len(dry_run)}")
    
    print(f"\n📦 Chunks:")
    print(f"   🗑️ Eliminati: {total_deleted}")
    print(f"   📤 Inseriti: {total_inserted}")
    
    if errors:
        print("\n❌ ERRORI:")
        for r in errors:
            print(f"   - {r['filename']}: {r['error']}")
    
    print()


def main():
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Smart Ingest Pipeline v3.0 - HYBRID SEARCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python smart_ingest_hybrid.py docs/                    # Processa cartella
  python smart_ingest_hybrid.py documento.pdf            # Singolo file
  python smart_ingest_hybrid.py docs/ --dry-run          # Preview
  python smart_ingest_hybrid.py docs/ --no-sparse        # Solo semantic
  python smart_ingest_hybrid.py docs/ --chunk-size 512   # Chunks più piccoli
        """
    )
    
    parser.add_argument("input_path", help="File o cartella da processare")
    parser.add_argument("--mode", choices=["replace", "add-only"], default="replace",
                        help="Modalità: replace (default) o add-only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra cosa farebbe senza eseguire")
    parser.add_argument("--no-sparse", action="store_true",
                        help="Disabilita sparse vectors (solo dense)")
    parser.add_argument("--premium", action="store_true",
                        help="Forza Premium Mode per TUTTI i file (Excel è già attivo di default).")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Dimensione chunks (default: 2000 chars ~500 tokens)")
    parser.add_argument("--chunk-overlap", type=int, default=300,
                        help="Overlap chunks (default: 300 = 15%%)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Output verboso")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crea configurazione
    config = HybridConfig()
    config.chunk_size = args.chunk_size
    config.chunk_overlap = args.chunk_overlap
    config.enable_sparse = not args.no_sparse
    config.use_premium_mode = args.premium
    
    # Se l'utente non ha specificato --premium ma sta processando file .xlsx, potremmo avvisarlo?
    # Per ora lasciamo il controllo esplicito, ma nel log iniziale lo indichiamo.
    
    # Verifica env vars
    missing = []
    if not config.llama_api_key:
        missing.append("LLAMA_CLOUD_API_KEY")
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not config.qdrant_url:
        missing.append("QDRANT_URL")
    if not config.qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if not config.qdrant_collection:
        missing.append("QDRANT_COLLECTION")
    
    if missing:
        logger.error(f"❌ Variabili mancanti: {', '.join(missing)}")
        sys.exit(1)
    
    # Header
    print("\n" + "="*60)
    print("🚀 SMART INGEST PIPELINE v3.0 - HYBRID SEARCH")
    print("="*60)
    print(f"📂 Input: {args.input_path}")
    print(f"⚙️ Mode: {args.mode}")
    print(f"🔍 Dry run: {args.dry_run}")
    print(f"📏 Chunk size: {config.chunk_size}")
    print(f"🔗 Overlap: {config.chunk_overlap}")
    print(f"💎 Premium Mode: {'ON (Forced)' if config.use_premium_mode else 'AUTO (ON for Excel only)'}")
    print(f"🔮 Dense: {config.dense_model} ({config.dense_dimensions}d)")
    print(f"🔑 Sparse: {config.sparse_model if config.enable_sparse else 'DISABLED'}")
    print(f"🗄️ Collection: {config.qdrant_collection}")
    print("="*60 + "\n")
    
    # Esegui
    pipeline = HybridIngestPipeline(config)
    
    start_time = time.time()
    results = pipeline.run(
        input_path=args.input_path,
        mode=args.mode,
        dry_run=args.dry_run
    )
    elapsed = time.time() - start_time
    
    print_summary(results)
    print(f"⏱️ Tempo totale: {elapsed:.2f}s\n")
    
    # Stats finali
    if not args.dry_run and results:
        stats = pipeline.qdrant.get_stats()
        if stats:
            print(f"📊 Collection '{config.qdrant_collection}':")
            print(f"   - Punti: {stats.get('points_count', 'N/A')}")
            print(f"   - Status: {stats.get('status', 'N/A')}")
            print(f"   - Dense: {stats.get('config', {}).get('dense_vector', 'N/A')}")
            print(f"   - Sparse: {stats.get('config', {}).get('sparse_vector', 'N/A')}")
            print()
    
    errors = [r for r in results if r["status"] == "error"]
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()