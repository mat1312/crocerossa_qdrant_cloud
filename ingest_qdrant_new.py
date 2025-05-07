"""
Script di ingestion per Qdrant Cloud.
Carica tutti i documenti dalla cartella 'parsed' nel vector database Qdrant Cloud.
Ad ogni esecuzione, elimina la collezione esistente e ricrea completamente l'indice.
"""

import os
import glob
import asyncio
import logging
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
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
PARSED_DIR = os.path.join(BASE_DIR, "parsed")

# Parametri di configurazione
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-large"

async def load_all_documents() -> List[Document]:
    """Carica tutti i documenti dalla cartella parsed."""
    documents = []
    md_files = glob.glob(os.path.join(PARSED_DIR, "**/*.md"), recursive=True)
    
    logger.info(f"Trovati {len(md_files)} file markdown")
    
    for file_path in md_files:
        try:
            logger.info(f"Caricamento: {file_path}")
            loader = UnstructuredMarkdownLoader(file_path)
            file_docs = loader.load()
            
            # Aggiungi metadati utili
            for doc in file_docs:
                filename = os.path.basename(file_path)
                doc.metadata.update({
                    "source": file_path,
                    "filename": filename,
                    "document_type": "Markdown"
                })
            
            documents.extend(file_docs)
            logger.info(f"Caricato documento: {filename}")
        except Exception as e:
            logger.error(f"Errore nel caricamento di {file_path}: {str(e)}")
    
    logger.info(f"Totale documenti caricati: {len(documents)}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Divide i documenti in chunks più piccoli."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ", "\n### ",  # Titoli Markdown
            "\n\n", "\n",       # Paragrafi e righe
            ". ", "! ", "? ",   # Fine frasi
            ";", ":",           # Separatori frase
            " ", ""             # Fallback
        ],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Aggiungi metadati ai chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        
        # Crea un titolo per il chunk
        text = chunk.page_content.strip()
        first_line = text.split('\n')[0] if '\n' in text else text
        title = (first_line[:50] + '...') if len(first_line) > 50 else first_line
        chunk.metadata["chunk_title"] = title
    
    logger.info(f"Documenti suddivisi in {len(chunks)} chunks (dimensione: {CHUNK_SIZE}, sovrapposizione: {CHUNK_OVERLAP})")
    return chunks

async def delete_collection_if_exists():
    """Elimina la collezione se esiste."""
    if not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        raise ValueError("Configurazione Qdrant Cloud incompleta. Verifica le variabili d'ambiente.")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if QDRANT_COLLECTION in collection_names:
            logger.info(f"Eliminazione della collezione esistente: {QDRANT_COLLECTION}")
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            logger.info(f"Collezione {QDRANT_COLLECTION} eliminata con successo")
    except Exception as e:
        logger.error(f"Errore durante l'eliminazione della collezione: {str(e)}")

async def create_vector_store(chunks: List[Document]):
    """Crea il vector store su Qdrant Cloud."""
    if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        raise ValueError("Configurazione incompleta. Verifica le variabili d'ambiente.")
    
    if not chunks:
        logger.warning("Nessun chunk da indicizzare. Uscita.")
        return None
    
    # Prepara il modello di embedding
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Inizializza il client Qdrant con timeout più lunghi
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        timeout=120.0  # Timeout più lungo (2 minuti)
    )
    
    # Crea la collezione (se non esiste già)
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if QDRANT_COLLECTION in collection_names:
            logger.info(f"Eliminazione della collezione esistente: {QDRANT_COLLECTION}")
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            logger.info(f"Collezione {QDRANT_COLLECTION} eliminata con successo")
        
        # Crea una nuova collezione
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "size": 3072,  # dimensione per text-embedding-3-large
                "distance": "Cosine"
            }
        )
        logger.info(f"Creata nuova collezione: {QDRANT_COLLECTION}")
    except Exception as e:
        logger.error(f"Errore durante la gestione della collezione: {str(e)}")
        return None
    
    # Processo di upload in batch con gestione asincrona
    start_time = time.time()
    
    try:
        # Suddividi in batch più piccoli per evitare timeout
        batch_size = 50  # Batch più piccoli
        total_chunks = len(chunks)
        batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        
        logger.info(f"Caricamento di {total_chunks} chunks in {len(batches)} batch (dimensione batch: {batch_size})")
        
        # Processa i batch
        for i, batch in enumerate(batches):
            logger.info(f"Elaborazione batch {i+1}/{len(batches)} ({len(batch)} chunks)")
            
            # Ottieni gli embedding per il batch corrente
            batch_texts = [doc.page_content for doc in batch]
            batch_metadatas = [doc.metadata for doc in batch]
            
            # Ottieni gli embedding in maniera sincrona
            embeddings_batch = embeddings.embed_documents(batch_texts)
            
            # Crea i punti da caricare
            points = []
            for j, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadatas, embeddings_batch)):
                # Crea un ID univoco basato sul batch e sull'indice
                point_id = i * batch_size + j
                
                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "page_content": text,
                        "metadata": metadata
                    }
                })
            
            # Carica i punti in Qdrant
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            
            logger.info(f"Batch {i+1}/{len(batches)} completato")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Vector store creato in {elapsed_time:.2f} secondi")
        
        # Verifica che l'indicizzazione sia completa
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"Statistiche della collezione: {collection_info}")
        
        # Crea l'oggetto vector_store per compatibilità
        vector_store = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embeddings=embeddings
        )
        
        return vector_store
    
    except Exception as e:
        logger.error(f"Errore durante la creazione del vector store: {str(e)}")
        return None

async def main():
    """Funzione principale."""
    logger.info("Inizio processo di ingestion")
    
    try:
        # 1. Elimina la collezione esistente
        await delete_collection_if_exists()
        
        # 2. Carica tutti i documenti
        documents = await load_all_documents()
        
        if not documents:
            logger.warning("Nessun documento caricato. Uscita.")
            return
        
        # 3. Dividi i documenti in chunks
        chunks = split_documents(documents)
        
        # 4. Crea il vector store
        await create_vector_store(chunks)
        
        logger.info("Processo di ingestion completato con successo")
    
    except Exception as e:
        logger.error(f"Errore durante il processo di ingestion: {str(e)}")

if __name__ == "__main__":
    # Gestione del loop di eventi asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {str(e)}") 