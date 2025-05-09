"""
Backend principale per l'Assistente della Croce Rossa Italiana.
Include RAG con Qdrant Cloud, gestione conversazioni e API per il frontend.
Adattato per lavorare con il nuovo sistema di ingest asincrono.
"""

import os
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from pathlib import Path

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

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
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.json"

# Configurazione LLM
MODEL_NAME = "gpt-4.1"

MAX_TOKENS = 30000
SIMILARITY_TOP_K = 70
MAX_HISTORY_LENGTH = 4
TEMPERATURE = 0.2

# Modelli Pydantic per le richieste e risposte API
class Source(BaseModel):
    file_name: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None
    chunk_id: Optional[int] = None
    document_type: Optional[str] = None
    relative_path: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = []

class ResetRequest(BaseModel):
    session_id: str

class ResetResponse(BaseModel):
    status: str
    message: str

class TranscriptResponse(BaseModel):
    transcript_html: str

class ContactInfoResponse(BaseModel):
    contact_info: str

# Modello Pydantic per la richiesta e risposta del webhook ElevenLabs
class ElevenLabsWebhookRequest(BaseModel):
    text: str

class ElevenLabsWebhookResponse(BaseModel):
    response: str

# Memoria delle conversazioni per ogni sessione
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# Inizializza FastAPI
app = FastAPI(title="Assistente Croce Rossa Italiana API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Collega i file statici
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Sistema di prompt
condense_question_prompt = PromptTemplate.from_template("""
Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up
in una domanda autonoma che riprende il contesto della conversazione se necessario.

Storico conversazione:
{chat_history}

Domanda di follow-up: {question}

Domanda autonoma riformulata:
""")

# Importo datetime per ottenere data e ora attuali
import datetime

# Definisco il prompt QA con un parametro aggiuntivo per la data e ora
qa_prompt = PromptTemplate.from_template("""
# Prompt per l'Assistente Virtuale della Croce Rossa Italiana

## Compito Principale
Sei un assistente AI della Croce Rossa Italiana (CRI). Il tuo unico scopo è rispondere alla domanda dell'utente (`{question}`) basandoti **esclusivamente** sulle informazioni fornite nel contesto (`{context}`).

## Data e Ora Attuali
Data e ora correnti: {current_datetime}

## Istruzioni Fondamentali
1.  **Usa SOLO il `{context}`**: La tua risposta deve derivare unicamente dal testo fornito.
2.  **Info Mancante**: Se la risposta non è nel `{context}`, **dichiaralo esplicitamente** ("L'informazione non è presente nel contesto fornito."). Non inventare o aggiungere nulla.
3.  **Precisione Struttura CRI**: Mantieni sempre la corretta distinzione tra livello Nazionale, Regionale e Territoriale, se menzionati nel contesto.
4.  **Formattazione Chiara**: Presenta la risposta in modo **ben strutturato, chiaro e facile da leggere**. Usa paragrafi, elenchi puntati o grassetto se migliorano la leggibilità e l'organizzazione dell'informazione.
5.  **Sicurezza ed Emergenze**:
    * Non dare MAI consigli medici.
    * Per emergenze sanitarie, **rimanda SEMPRE e SOLO al 112/118**.
    * Per urgenze operative, indica i contatti delle Sale Operative competenti **solo se presenti nel `{context}`**.
6.  **Neutralità**: Non esprimere opinioni personali, politiche o giudizi. Attieniti ai fatti del `{context}`.

---

**Contesto:**
{context}

**Domanda:**
{question}

**Risposta:**
""")

def load_metadata() -> Dict:
    """Carica i metadati dell'ingest se disponibili."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "last_updated": None}

def get_vectorstore():
    """Carica il vector store da Qdrant Cloud."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    if not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        raise ValueError("Configurazione Qdrant Cloud incompleta. Verifica .env file.")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Connessione a Qdrant Cloud
    from qdrant_client import QdrantClient
    
    # Inizializza il client Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # Usa il client esistente per connettersi alla collezione
    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings
    )
    
    return vector_store

def get_conversation_chain(session_id: str):
    """Crea la catena conversazionale con RAG."""
    # Inizializza la memoria se non esiste
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Prepara la memoria per la conversazione
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    # Carica la conversazione dalla memoria
    for message in conversation_history[session_id]:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])
    
    # Carica il vectorstore
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SIMILARITY_TOP_K}
    )
    
    # Configura il modello LLM
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    
    # Ottieni data e ora correnti in formato italiano
    current_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Crea un nuovo prompt template che include la data e ora correnti
    dynamic_qa_prompt = PromptTemplate(
        template=qa_prompt.template,
        input_variables=["context", "question"],
        partial_variables={"current_datetime": current_datetime}
    )
    
    # Crea la catena conversazionale
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": dynamic_qa_prompt}
    )
    
    return chain

def format_sources(source_docs) -> List[Source]:
    """Formatta i documenti di origine in un formato più leggibile."""
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        
        # Estrai il nome del file dal percorso completo
        file_name = None
        if "source" in metadata:
            # Gestisci sia percorsi con / che con \
            path = metadata["source"].replace('\\', '/')
            file_name_with_ext = path.split('/')[-1]
            
            # Rimuovi l'estensione
            file_name = os.path.splitext(file_name_with_ext)[0]
        
        source = Source(
            file_name=file_name,
            page=metadata.get("page_number", metadata.get("page", None)),
            text=doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
            chunk_id=metadata.get("chunk_id", None),
            document_type=metadata.get("document_type", None),
            relative_path=metadata.get("relative_path", None)
        )
        sources.append(source)
    return sources

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Endpoint principale che serve la pagina HTML."""
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/langchain-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Endpoint per processare le domande dell'utente."""
    try:
        # Ottiene o crea la catena conversazionale
        chain = get_conversation_chain(request.session_id)
        
        # Salva la domanda utente nella storia
        conversation_history.setdefault(request.session_id, [])
        conversation_history[request.session_id].append({
            "role": "user",
            "content": request.query
        })
        
        # Mantiene la storia limitata per evitare di superare i limiti del contesto
        if len(conversation_history[request.session_id]) > MAX_HISTORY_LENGTH * 2:
            conversation_history[request.session_id] = conversation_history[request.session_id][-MAX_HISTORY_LENGTH*2:]
        
        # Esegue la query
        result = chain({"question": request.query})
        
        # Salva la risposta nella storia
        conversation_history[request.session_id].append({
            "role": "assistant",
            "content": result["answer"]
        })
        
        # Formatta le fonti
        sources = format_sources(result.get("source_documents", []))
        
        # Ritorna il risultato
        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Errore nel processare la query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")

@app.post("/reset-conversation", response_model=ResetResponse)
async def reset_conversation(request: ResetRequest):
    """Resetta la conversazione per una specifica sessione."""
    session_id = request.session_id
    
    if session_id in conversation_history:
        conversation_history[session_id] = []
        return ResetResponse(status="success", message="Conversazione resettata con successo")
    
    return ResetResponse(status="success", message="Nessuna conversazione trovata per questa sessione")

@app.get("/api/transcript", response_model=TranscriptResponse)
async def get_transcript():
    """Genera un transcript HTML della conversazione."""
    # Per semplicità, utilizza la prima sessione disponibile
    # In una implementazione reale, passeresti la session_id come parametro
    if not conversation_history:
        return TranscriptResponse(transcript_html="<p>Nessuna conversazione disponibile</p>")
    
    # Prendi la prima sessione disponibile
    session_id = next(iter(conversation_history))
    messages = conversation_history[session_id]
    
    if not messages:
        return TranscriptResponse(transcript_html="<p>Nessuna conversazione disponibile</p>")
    
    # Formatta il transcript in HTML
    html = ""
    for idx, message in enumerate(messages):
        role_class = "text-red-600 font-medium" if message["role"] == "assistant" else "text-blue-600 font-medium"
        role_name = "Assistente CRI" if message["role"] == "assistant" else "Utente"
        
        html += f"""
        <div class="mb-4 pb-3 border-b border-gray-100">
            <div class="mb-1"><span class="{role_class}">{role_name}:</span></div>
            <p class="pl-2">{message["content"]}</p>
        </div>
        """
    
    return TranscriptResponse(transcript_html=html)

@app.get("/api/extract_contacts", response_model=ContactInfoResponse)
async def extract_contact_info():
    """Estrae informazioni di contatto dalla conversazione."""
    if not conversation_history:
        return ContactInfoResponse(contact_info="<p>Nessuna informazione di contatto rilevata</p>")
    
    # Per semplicità, utilizza la prima sessione disponibile
    session_id = next(iter(conversation_history))
    messages = conversation_history[session_id]
    
    if not messages:
        return ContactInfoResponse(contact_info="<p>Nessuna informazione di contatto rilevata</p>")
    
    # Chiedi a GPT di estrarre le informazioni di contatto dalla conversazione
    messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=1000
    )
    
    extraction_prompt = f"""
    Analizza la seguente conversazione ed estrai tutte le informazioni utili come:
    - Dati personali (nome, cognome, età)
    - Informazioni di contatto (email, telefono)
    - Richieste specifiche di servizi
    - Località o sedi menzionate
    - Date e orari di interesse
    - Qualsiasi altra informazione rilevante
    
    Formatta il risultato in modo strutturato e chiaro.
    Se non trovi informazioni in una categoria, omettila.
    
    Conversazione:
    {messages_text}
    """
    
    try:
        response = llm.invoke(extraction_prompt)
        
        # Formatta HTML
        contact_info_html = f"""
        <div class="mb-4">
            <h3 class="text-lg font-medium mb-2 text-red-600">Informazioni Estratte</h3>
            <div class="whitespace-pre-line">{response.content}</div>
        </div>
        """
        
        return ContactInfoResponse(contact_info=contact_info_html)
    
    except Exception as e:
        logger.error(f"Errore nell'estrazione dei contatti: {str(e)}", exc_info=True)
        return ContactInfoResponse(
            contact_info="<p>Si è verificato un errore durante l'estrazione delle informazioni.</p>"
        )

@app.post("/elevenlabs-webhook", response_model=ElevenLabsWebhookResponse)
async def elevenlabs_webhook(request: ElevenLabsWebhookRequest):
    """Endpoint per webhook ElevenLabs che restituisce i top chunk dalla RAG."""
    try:
        # Ottiene il vectorstore
        vector_store = get_vectorstore()
        
        # Crea retriever con k per ottenere i top chunk
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Prende i top chunk
        )
        
        # Recupera i documenti
        docs = retriever.get_relevant_documents(request.text)
        
        # Combina tutti i chunk in un'unica risposta
        combined_response = "\n\n".join([doc.page_content for doc in docs])
        
        return ElevenLabsWebhookResponse(response=combined_response)
    
    except Exception as e:
        logger.error(f"Errore nel processare la richiesta webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")

@app.get("/api/metadata")
async def get_metadata():
    """Ottiene informazioni sui documenti indicizzati."""
    try:
        metadata = load_metadata()
        
        # Calcola alcune statistiche
        total_files = len(metadata.get("files", {}))
        last_updated = metadata.get("last_updated", "Sconosciuto")
        
        # Raggruppa per tipo di documento
        doc_types = {}
        for filepath, file_info in metadata.get("files", {}).items():
            ext = os.path.splitext(filepath)[1].lower()
            doc_types[ext] = doc_types.get(ext, 0) + 1
        
        # Ritorna le informazioni
        return {
            "total_files": total_files,
            "last_updated": last_updated,
            "document_types": doc_types,
            "vector_store_path": str(QDRANT_URL)
        }
        
    except Exception as e:
        logger.error(f"Errore nel caricamento dei metadati: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "message": "Errore nel caricamento dei metadati. Eseguire prima ingest.py."
        }

# Avvio dell'applicazione
if __name__ == "__main__":
    import uvicorn
    try:
        # Verifica connessione a Qdrant Cloud
        get_vectorstore()
        logger.info("Connessione a Qdrant Cloud stabilita. Avvio del server...")
    except Exception as e:
        logger.error(f"Errore durante la connessione a Qdrant Cloud: {str(e)}", exc_info=True)
        exit(1)
        
    # Avvia il server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)