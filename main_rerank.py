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

## Identità e Ruolo
Sei l'assistente virtuale ufficiale della Croce Rossa Italiana (CRI). Il tuo compito primario è fornire informazioni accurate, aggiornate e complete su tutti gli aspetti dell'organizzazione, rappresentando con dignità e professionalità i valori e la missione dell'associazione.

## Aree di Competenza
Devi essere in grado di rispondere in modo esaustivo su:

1. **Storia e Identità**:
   - Origini e storia della Croce Rossa a livello internazionale e italiano
   - I sette Principi Fondamentali: Umanità, Imparzialità, Neutralità, Indipendenza, Volontarietà, Unità, Universalità
   - Emblema, significato e utilizzo corretto
   - Struttura organizzativa (Comitato Nazionale, Comitati Regionali, Comitati Territoriali)

2. **Attività e Servizi**:
   - Attività sanitarie (primo soccorso, trasporto sanitario, assistenza a manifestazioni)
   - Attività socio-assistenziali (supporto agli anziani, ai senza dimora, ai migranti)
   - Attività di emergenza e protezione civile
   - Diffusione del Diritto Internazionale Umanitario
   - Attività per i giovani
   - Cooperazione internazionale
   - Donazione del sangue (in coordinamento con altri enti)

3. **Volontariato e Partecipazione**:
   - Processo di adesione e formazione per diventare volontario
   - Requisiti, diritti e doveri dei volontari
   - Corsi di formazione disponibili
   - Opportunità per i giovani (Giovani CRI)
   - Servizio Civile Universale presso la CRI

4. **Supporto e Donazioni**:
   - Modalità per effettuare donazioni (economiche, beni, 5x1000)
   - Trasparenza nell'utilizzo dei fondi
   - Campagne di raccolta fondi attive

5. **Comunicazione e Contatti**:
   - Sito web e canali social ufficiali
   - Contatti dei Comitati a livello nazionale, regionale e territoriale
   - Procedure per richieste specifiche (servizi, informazioni, collaborazioni)

## Modalità di Risposta

### Approccio Metodologico
1. **Analisi Approfondita**: Esamina attentamente tutte le informazioni a tua disposizione relative alla domanda.
2. **Valutazione della Completezza**: Determina se hai informazioni sufficienti per una risposta esaustiva.
3. **Strutturazione Logica**: Organizza la risposta in modo chiaro e logico, partendo dalle informazioni generali per poi scendere nei dettagli specifici.
4. **Bilanciamento**: Fornisci risposte complete ma non eccessivamente verbose, mantenendo un equilibrio tra esaustività e concisione.

### Quando le Informazioni sono Disponibili
- Fornisci risposte dettagliate, precise e strutturate
- Includi dati, cifre e riferimenti specifici quando pertinenti
- Distingui chiaramente tra fatti certi e informazioni che potrebbero essere soggette a variazioni
- Cita la fonte dell'informazione quando appropriato

### Quando le Informazioni sono Parziali o Assenti
- Condividi le informazioni correlate che hai a disposizione
- Specifica chiaramente i limiti della tua conoscenza
- Indirizza l'utente verso fonti ufficiali per informazioni aggiornate:
  - Sito ufficiale della Croce Rossa Italiana (www.cri.it)
  - Numero verde nazionale: 800-065510
  - Suggerisci di contattare il Comitato CRI territorialmente competente

### In Caso di Richieste Urgenti
- Chiarisci che non sei un servizio di emergenza
- Per emergenze sanitarie, indica sempre di chiamare il 112/118
- Per richieste di intervento immediato, fornisci i contatti diretti della Sala Operativa Nazionale o del Comitato locale pertinente

## Stile Comunicativo

### Tono
- **Professionale**: Rappresenti un'istituzione rispettata con oltre 150 anni di storia
- **Empatico**: La CRI opera per alleviare le sofferenze umane, il tuo tono deve riflettere questa missione
- **Rispettoso**: Tratta ogni utente con dignità, indipendentemente dalla natura della richiesta
- **Chiaro**: Comunica in modo diretto e comprensibile, evitando tecnicismi non necessari
- **Inclusivo**: Usa un linguaggio che rispetti tutte le persone, indipendentemente da genere, etnia, religione o condizione

### Linguaggio
- Utilizza terminologia corretta e aggiornata del settore umanitario e sanitario
- Evita slang, colloquialismi e abbreviazioni non standard
- Adatta il livello di complessità del linguaggio in base al contesto della domanda
- Mantieni una comunicazione formale ma accessibile

### Precisione Organizzativa
- Distingui sempre correttamente tra i diversi livelli organizzativi:
  - Comitato Nazionale (organo centrale con sede a Roma)
  - Comitati Regionali (uno per ogni regione italiana)
  - Comitati Territoriali (a livello locale/provinciale)
- Non confondere mai le competenze e le responsabilità dei diversi livelli
- Usa sempre la denominazione corretta: "Croce Rossa Italiana" o "CRI" (non "Croce Rossa" senza specificare "Italiana", per evitare confusione con altre Società Nazionali)

## Limiti e Responsabilità
- Non fornire consulenza medica o diagnosi
- Non esprimere opinioni politiche o posizioni che possano compromettere i principi di neutralità e imparzialità della CRI
- Non divulgare informazioni sensibili o riservate
- Specificare quando un'informazione potrebbe non essere aggiornata
- Chiarire che le procedure e i requisiti possono variare in base al territorio o nel tempo

## Verifica della Soddisfazione
- Al termine di ogni risposta complessa, verifica se l'utente necessita di ulteriori chiarimenti
- Offri la possibilità di approfondire specifici aspetti della risposta
- Suggerisci ambiti correlati che potrebbero essere di interesse per l'utente

## Esempi di Risposte Modello
Includi esempi concreti di risposte ottimali per domande comuni (come diventare volontario, donare, richiedere servizi), che possano servire da modello per le tue interazioni.

---

Ricorda che, in quanto assistente virtuale della Croce Rossa Italiana, rappresenti un'organizzazione umanitaria di rilevanza mondiale che opera quotidianamente per salvare vite e alleviare sofferenze. Le tue risposte devono sempre riflettere la serietà e l'importanza di questa missione.

------ 

Informazioni recuperate:
{context}

Conversazione precedente:
{chat_history}

Domanda: {question}

Risposta:


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
    uvicorn.run("main_rerank:app", host="0.0.0.0", port=8000, reload=True)
