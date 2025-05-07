"""
Script di ingestion per aggiungere un singolo file markdown alla collezione Qdrant Cloud esistente.
A differenza degli altri script di ingestion, questo script NON elimina la collezione esistente
ma aggiunge semplicemente il nuovo file specificato alla collezione esistente.
"""

import os
import logging
import time
import argparse
from typing import List
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

# Parametri di configurazione (gli stessi usati negli altri script)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-large"

def load_single_document(file_path: str) -> List[Document]:
    """Carica un singolo documento markdown."""
    if not os.path.exists(file_path):
        logger.error(f"Il file {file_path} non esiste.")
        return []
    
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
        
        logger.info(f"Caricato documento: {os.path.basename(file_path)}")
        return file_docs
    except Exception as e:
        logger.error(f"Errore nel caricamento di {file_path}: {str(e)}")
        return []

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
    
    logger.info(f"Documento suddiviso in {len(chunks)} chunks (dimensione: {CHUNK_SIZE}, sovrapposizione: {CHUNK_OVERLAP})")
    return chunks

def add_to_vector_store(chunks: List[Document]):
    """Aggiunge i chunks alla collezione Qdrant esistente."""
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
    
    # Verifica che la collezione esista
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if QDRANT_COLLECTION not in collection_names:
            logger.error(f"La collezione {QDRANT_COLLECTION} non esiste. Esegui prima lo script di ingestion completo.")
            return None
        
        # Ottieni informazioni sulla collezione esistente
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION)
        current_points = collection_info.points_count
        logger.info(f"Collezione {QDRANT_COLLECTION} trovata con {current_points} punti")
        
    except Exception as e:
        logger.error(f"Errore durante la verifica della collezione: {str(e)}")
        return None
    
    # Processo di upload in batch
    start_time = time.time()
    
    try:
        # Suddividi in batch più piccoli per evitare timeout
        batch_size = 50
        total_chunks = len(chunks)
        batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        
        logger.info(f"Caricamento di {total_chunks} nuovi chunks in {len(batches)} batch")
        
        # Processa i batch
        new_points = []
        for i, batch in enumerate(batches):
            logger.info(f"Elaborazione batch {i+1}/{len(batches)} ({len(batch)} chunks)")
            
            # Ottieni gli embedding per il batch corrente
            batch_texts = [doc.page_content for doc in batch]
            batch_metadatas = [doc.metadata for doc in batch]
            
            # Ottieni gli embedding
            embeddings_batch = embeddings.embed_documents(batch_texts)
            
            # Crea i punti da caricare
            for j, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadatas, embeddings_batch)):
                # Genera un ID basato sull'indice ma assicurati che sia unico
                # Usa l'offset dal numero corrente di punti
                point_id = current_points + i * batch_size + j
                
                new_points.append({
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
                points=new_points
            )
            
            logger.info(f"Batch {i+1}/{len(batches)} completato")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Aggiunta di {len(new_points)} nuovi punti in {elapsed_time:.2f} secondi")
        
        # Verifica che l'aggiunta sia completa
        updated_collection = client.get_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"Collezione aggiornata: ora contiene {updated_collection.points_count} punti")
        
        return True
    
    except Exception as e:
        logger.error(f"Errore durante l'aggiunta al vector store: {str(e)}")
        return None

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description="Aggiungi un singolo file markdown al vector store esistente.")
    parser.add_argument("file_path", type=str, help="Percorso al file markdown da aggiungere")
    args = parser.parse_args()
    
    file_path = args.file_path
    # Se il percorso è relativo, costruisci il percorso completo
    if not os.path.isabs(file_path):
        file_path = os.path.join(PARSED_DIR, file_path)
    
    logger.info(f"Inizio processo di ingestion per il file {file_path}")
    
    try:
        # 1. Carica il documento
        documents = load_single_document(file_path)
        
        if not documents:
            logger.warning("Nessun documento caricato. Uscita.")
            return
        
        # 2. Dividi il documento in chunks
        chunks = split_documents(documents)
        
        # 3. Aggiungi i chunks al vector store esistente
        add_to_vector_store(chunks)
        
        logger.info("Processo di ingestion completato")
    
    except Exception as e:
        logger.error(f"Errore durante il processo: {str(e)}")

if __name__ == "__main__":
    main() 