"""
Script di ingestion per caricare documenti nel Qdrant Cloud vector database.
Questo script processa i file dalla cartella 'parsed' e li indicizza nel vector database Qdrant Cloud
con elaborazione asincrona per aumentare significativamente la velocità di processamento.
Ad ogni esecuzione, lo script elimina la collezione esistente (se presente) e la ricrea da zero
con tutti i documenti presenti nella cartella specificata.
"""

import os
import glob
import argparse
import json
import hashlib
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import logging
from datetime import datetime
import concurrent.futures
from qdrant_client import QdrantClient

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Configurazione percorsi
BASE_DIR = os.path.dirname(__file__)
PARSED_DIR = os.path.join(BASE_DIR, "parsed")  # Cartella dei documenti originali
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)

# Configurazione chunking - valori modificabili tramite argparse
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MAX_CONCURRENCY = 10  # Numero massimo di operazioni simultanee
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def get_document_hash(filepath: str) -> str:
    """Genera un hash per un documento per tracciare cambiamenti."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_metadata() -> Dict[str, Any]:
    """Carica i metadati esistenti o crea un nuovo file di metadati."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"files": {}, "last_updated": None}


def save_metadata(metadata: Dict[str, Any]):
    """Salva i metadati su disco."""
    metadata["last_updated"] = datetime.now().isoformat()
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def enrich_document_metadata(doc: Document, filepath: str, metadata: Dict[str, Any]) -> Document:
    """Arricchisce i metadati del documento con informazioni utili per il RAG."""
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filename)[1].lower()
    directory = os.path.dirname(filepath)
    rel_path = os.path.relpath(filepath, PARSED_DIR)
    
    # Determina il tipo di documento basato sull'estensione
    doc_type = {
        '.pdf': 'PDF',
        '.txt': 'Text',
        '.md': 'Markdown',
        '.html': 'HTML',
        '.docx': 'Word'
    }.get(file_ext, 'Unknown')
    
    # Arricchisci i metadati
    doc.metadata.update({
        "source": filepath,
        "filename": filename,
        "extension": file_ext,
        "document_type": doc_type,
        "relative_path": rel_path,
        "directory": directory,
        "processed_date": datetime.now().isoformat(),
        "file_size_bytes": os.path.getsize(filepath),
        "file_hash": metadata["files"].get(filepath, {}).get("hash", get_document_hash(filepath))
    })
    
    # Aggiungi metadati specifici per diversi tipi di documento
    if doc_type == 'PDF' and 'page' in doc.metadata:
        doc.metadata['page_number'] = doc.metadata['page'] + 1  # Converti a numerazione 1-based
    
    return doc


async def process_file_async(filepath: str, loader_cls, metadata: Dict[str, Any], force_reload: bool = False, **loader_kwargs) -> List[Document]:
    """Processa un singolo file in modo asincrono."""
    # Controlla se il file è stato modificato
    current_hash = get_document_hash(filepath)
    if filepath in metadata["files"] and not force_reload:
        if metadata["files"][filepath]["hash"] == current_hash:
            logger.info(f"Saltato {filepath} (non modificato)")
            return []
    
    logger.info(f"Caricamento: {filepath}")
    try:
        # L'operazione di caricamento del file è intensiva di I/O, quindi la eseguiamo in un thread separato
        loop = asyncio.get_event_loop()
        loader = loader_cls(filepath, **loader_kwargs)
        # Esegue il metodo load() in un thread pool per non bloccare il loop degli eventi
        file_docs = await loop.run_in_executor(None, loader.load)
        
        # Arricchisci i metadati per ogni chunk del documento
        for doc in file_docs:
            doc = enrich_document_metadata(doc, filepath, metadata)
        
        # Aggiorna i metadati del file
        metadata["files"][filepath] = {
            "hash": current_hash,
            "last_processed": datetime.now().isoformat(),
            "size_bytes": os.path.getsize(filepath)
        }
        
        return file_docs
    except Exception as e:
        logger.error(f"Errore nel caricamento di {filepath}: {str(e)}")
        return []


async def load_documents_async(input_dir: str, metadata: Dict[str, Any], max_concurrency: int = DEFAULT_MAX_CONCURRENCY, force_reload: bool = False) -> List[Document]:
    """Carica tutti i documenti dalla directory specificata con processamento asincrono."""
    all_documents = []
    files_to_process = []
    
    # Verifica che la directory esista
    if not os.path.exists(input_dir):
        logger.error(f"La directory {input_dir} non esiste.")
        return []
    
    # Raccogli tutti i file da processare
    for pattern, loader_cls, kwargs in [
        (os.path.join(input_dir, "**/*.pdf"), PyPDFLoader, {}),
        (os.path.join(input_dir, "**/*.txt"), TextLoader, {"encoding": "utf-8"}),
        (os.path.join(input_dir, "**/*.md"), UnstructuredMarkdownLoader, {})
    ]:
        for filepath in glob.glob(pattern, recursive=True):
            files_to_process.append((filepath, loader_cls, kwargs))
    
    logger.info(f"Trovati {len(files_to_process)} file da controllare")
    
    # Utilizza un semaforo per limitare la concorrenza
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(filepath, loader_cls, kwargs):
        async with semaphore:
            return await process_file_async(filepath, loader_cls, metadata, force_reload, **kwargs)
    
    # Crea task per ogni file
    tasks = [process_with_semaphore(filepath, loader_cls, kwargs) for filepath, loader_cls, kwargs in files_to_process]
    
    # Esegui tutte le task contemporaneamente, ma con concorrenza limitata dal semaforo
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time
    
    # Appiattisci i risultati
    for docs in results:
        all_documents.extend(docs)
    
    files_processed = len([docs for docs in results if docs])
    
    logger.info(f"Elaborati {files_processed} file in {elapsed_time:.2f} secondi")
    logger.info(f"Caricati {len(all_documents)} documenti in totale")
    
    return all_documents


def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Divide i documenti in chunks più piccoli con strategia migliorata."""
    if not documents:
        return []
    
    # Utilizzo di separatori ottimizzati per garantire che i chunks abbiano senso semantico
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ", "\n### ",  # Titoli Markdown  
            "\n\n", "\n",       # Paragrafi e righe
            ". ", "! ", "? ",   # Fine frasi
            ";", ":",           # Separatori frase
            ",",                # Virgole
            " ", ""             # Ultimo caso: singole parole o caratteri
        ],
        keep_separator=True
    )
    
    start_time = time.time()
    chunks = text_splitter.split_documents(documents)
    elapsed_time = time.time() - start_time
    
    # Aggiungi metadati ai chunks relativi alla loro posizione nella sequenza
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        
        # Genera un titolo per il chunk basato sul contenuto
        text = chunk.page_content.strip()
        first_line = text.split('\n')[0] if '\n' in text else text
        title = (first_line[:50] + '...') if len(first_line) > 50 else first_line
        chunk.metadata["chunk_title"] = title
    
    logger.info(f"Documenti suddivisi in {len(chunks)} chunks in {elapsed_time:.2f} secondi (dimensione: {chunk_size}, sovrapposizione: {chunk_overlap})")
    return chunks


async def create_vector_store_async(chunks: List[Document], embedding_model: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 100):
    """Crea o aggiorna il vector store utilizzando Qdrant Cloud con batch asincroni."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    if not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        raise ValueError("Configurazione Qdrant Cloud incompleta. Verifica .env file.")
    
    if not chunks:
        logger.warning("Nessun chunk da indicizzare. Vector store non aggiornato.")
        return None
    
    # Prepara il modello di embedding
    embeddings_model = OpenAIEmbeddings(model=embedding_model)
    
    # Inizializza il client Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Elimina la collezione esistente se presente
    try:
        collections = await asyncio.to_thread(client.get_collections)
        collection_names = [collection.name for collection in collections.collections]
        
        if QDRANT_COLLECTION in collection_names:
            logger.info(f"Eliminazione della collezione esistente: {QDRANT_COLLECTION}")
            await asyncio.to_thread(client.delete_collection, collection_name=QDRANT_COLLECTION)
            logger.info(f"Collezione {QDRANT_COLLECTION} eliminata con successo")
    except Exception as e:
        logger.error(f"Errore durante l'eliminazione della collezione esistente: {str(e)}")
    
    # Processo di upload in batch con gestione asincrona
    start_time = time.time()
    
    # Dividi i chunks in batch per l'elaborazione
    total_chunks = len(chunks)
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    logger.info(f"Caricamento di {total_chunks} chunks in Qdrant Cloud ({len(batches)} batch)")
    
    # Crea il vector store con una nuova collezione
    try:
        # Utilizziamo la versione asincrona dell'operazione
        vector_store = await asyncio.to_thread(
            Qdrant.from_documents,
            documents=chunks,
            embedding=embeddings_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION,
            force_recreate=True  # Ora garantiamo che la collezione sia ricreata
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Vector store creato in {elapsed_time:.2f} secondi ({total_chunks} chunks)")
        return vector_store
    
    except Exception as e:
        logger.error(f"Errore durante la creazione del vector store: {str(e)}")
        return None


async def main_async():
    """Funzione principale asincrona."""
    # Configurazione parser per argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Indicizza documenti in Qdrant Cloud eliminando e ricreando la collezione.')
    parser.add_argument('--input-dir', type=str, default=PARSED_DIR,
                        help=f'Directory contenente i documenti da indicizzare (default: {PARSED_DIR})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Dimensione dei chunks di testo (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f'Sovrapposizione tra chunks (default: {DEFAULT_CHUNK_OVERLAP})')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_MAX_CONCURRENCY,
                        help=f'Numero massimo di operazioni simultanee (default: {DEFAULT_MAX_CONCURRENCY})')
    parser.add_argument('--embedding-model', type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f'Modello di embedding OpenAI da utilizzare (default: {DEFAULT_EMBEDDING_MODEL})')
    parser.add_argument('--force', action='store_true',
                        help='Forza la riprocessazione di tutti i file, anche se non modificati')
    
    args = parser.parse_args()
    
    logger.info("Avvio processo di indicizzazione...")
    logger.info(f"Cartella input: {args.input_dir}")
    logger.info(f"Configurazione chunking: dimensione={args.chunk_size}, sovrapposizione={args.chunk_overlap}")
    logger.info(f"Concorrenza massima: {args.concurrency}")
    logger.info(f"Modello embedding: {args.embedding_model}")
    logger.info(f"Forzare riprocessazione: {args.force}")
    logger.info(f"La collezione Qdrant esistente '{QDRANT_COLLECTION}' sarà eliminata e ricreata")
    
    # Verifica configurazione Qdrant
    if not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        logger.error("Configurazione Qdrant Cloud incompleta. Verifica .env file.")
        return
    
    # Carica i metadati esistenti
    metadata = load_metadata()
    
    # Carica e processa i documenti
    try:
        start_time = time.time()
        
        # 1. Carica tutti i documenti
        documents = await load_documents_async(
            args.input_dir, 
            metadata, 
            max_concurrency=args.concurrency,
            force_reload=args.force
        )
        
        if not documents:
            logger.info("Nessun documento da processare. Uscita.")
            return
        
        # 2. Dividi i documenti in chunks
        chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
        
        # 3. Crea o aggiorna il vector store
        vector_store = await create_vector_store_async(chunks, args.embedding_model)
        
        # 4. Salva i metadati aggiornati
        save_metadata(metadata)
        
        total_time = time.time() - start_time
        logger.info(f"Processo completato in {total_time:.2f} secondi.")
        
        # Statistiche finali
        logger.info(f"Documenti processati: {len(documents)}")
        logger.info(f"Chunks creati: {len(chunks)}")
        logger.info(f"Vector store: Qdrant Cloud (collection: {QDRANT_COLLECTION})")
        
    except Exception as e:
        logger.error(f"Errore durante il processo di indicizzazione: {str(e)}")


def main():
    """Entry point principale che avvia il loop asincrono."""
    try:
        # Ottieni o crea un loop di eventi
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Se non esiste un loop di eventi nella thread corrente, creane uno nuovo
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Esegui la funzione principale nel loop asincrono
        loop.run_until_complete(main_async())
    finally:
        # Chiudi il loop alla fine dell'esecuzione
        loop.close()


if __name__ == "__main__":
    main() 