"""
Backend principale per l'Assistente della Croce Rossa Italiana.
Include Retrieval‑Augmented Generation (RAG) con Qdrant Cloud, gestione
conversazioni e API per il frontend.
Questa versione integra un passaggio di re‑rank dei chunk tramite Cohere
Rerank v3.5.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate

# 🔹 Nuovo: retriever 2‑step (similarity + Cohere Rerank)
from retrieval import build_rerank_retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env & paths
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# LLM & chain params
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4.1"
MAX_TOKENS = 30_000
MAX_HISTORY_LENGTH = 4  # nr messaggi user+assistant tenuti in memoria
TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
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

class ElevenLabsWebhookRequest(BaseModel):
    text: str

class ElevenLabsWebhookResponse(BaseModel):
    response: str

# ---------------------------------------------------------------------------
# Conversazione in‑memory
# ---------------------------------------------------------------------------
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Assistente Croce Rossa Italiana API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Prompt templates (identici alla versione originale tranne il taglio per spazio)
# ---------------------------------------------------------------------------
condense_question_prompt = PromptTemplate.from_template(
    """
    Data la seguente conversazione e una domanda di follow‑up, riformula la domanda di follow‑up
    in una domanda autonoma includendo contesto dove necessario.

    Storico conversazione:
    {chat_history}

    Domanda di follow‑up: {question}

    Domanda autonoma riformulata:
    """
)

qa_prompt = PromptTemplate.from_template("""
# Prompt per l'Assistente Virtuale della Croce Rossa Italiana

## Compito Principale
Sei un assistente AI della Croce Rossa Italiana (CRI). Il tuo unico scopo è rispondere alla domanda dell'utente (`{question}`) basandoti **esclusivamente** sulle informazioni fornite nel contesto (`{context}`).

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

# ---------------------------------------------------------------------------
# Helper: metadata loader
# ---------------------------------------------------------------------------

def load_metadata() -> Dict:
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "last_updated": None}

# ---------------------------------------------------------------------------
# Helper: vector store
# ---------------------------------------------------------------------------

def get_vectorstore():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    if not QDRANT_URL or not QDRANT_API_KEY or not QDRANT_COLLECTION:
        raise ValueError("Configurazione Qdrant Cloud incompleta. Controlla .env.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    return Qdrant(client=client,
                  collection_name=QDRANT_COLLECTION,
                  embeddings=embeddings)

# ---------------------------------------------------------------------------
# Conversational chain (con rerank)
# ---------------------------------------------------------------------------

def get_conversation_chain(session_id: str):
    # memoria per questa sessione
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True,
                                      output_key="answer")

    # ripristina i messaggi precedenti nella memoria
    for m in conversation_history[session_id]:
        if m["role"] == "user":
            memory.chat_memory.add_user_message(m["content"])
        else:
            memory.chat_memory.add_ai_message(m["content"])

    vector_store = get_vectorstore()
    retriever = build_rerank_retriever(vector_store)

    llm = ChatOpenAI(model_name=MODEL_NAME,
                     max_tokens=MAX_TOKENS,
                     temperature=TEMPERATURE)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

# ---------------------------------------------------------------------------
# Helper: format sources
# ---------------------------------------------------------------------------

def format_sources(source_docs) -> List[Source]:
    formatted: List[Source] = []
    for doc in source_docs:
        md = doc.metadata
        file_name = None
        if "source" in md:
            path = md["source"].replace("\\", "/")
            file_name = os.path.splitext(path.split("/")[-1])[0]

        snippet = doc.page_content
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."

        formatted.append(Source(
            file_name=file_name,
            page=md.get("page_number", md.get("page")),
            text=snippet,
            chunk_id=md.get("chunk_id"),
            document_type=md.get("document_type"),
            relative_path=md.get("relative_path"),
        ))
    return formatted

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/langchain-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        chain = get_conversation_chain(request.session_id)

        # salva domanda utente in cronologia
        conversation_history.setdefault(request.session_id, []).append(
            {"role": "user", "content": request.query}
        )

        # trim storia
        if len(conversation_history[request.session_id]) > MAX_HISTORY_LENGTH * 2:
            conversation_history[request.session_id] = (
                conversation_history[request.session_id][-MAX_HISTORY_LENGTH * 2 :]
            )

        result = chain({"question": request.query})

        conversation_history[request.session_id].append(
            {"role": "assistant", "content": result["answer"]}
        )

        return QueryResponse(
            answer=result["answer"],
            sources=format_sources(result.get("source_documents", [])),
        )
    except Exception as e:
        logger.error("Errore nel processare la query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")


@app.post("/reset-conversation", response_model=ResetResponse)
async def reset_conversation(request: ResetRequest):
    conversation_history[request.session_id] = []
    return ResetResponse(status="success", message="Conversazione resettata con successo")


@app.get("/api/transcript", response_model=TranscriptResponse)
async def get_transcript():
    if not conversation_history:
        return TranscriptResponse(transcript_html="<p>Nessuna conversazione disponibile</p>")

    session_id = next(iter(conversation_history))
    messages = conversation


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_rerank:app", host="0.0.0.0", port=8001, reload=True)
