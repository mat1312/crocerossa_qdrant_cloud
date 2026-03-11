"""
Qdrant Management Dashboard
Dashboard interattiva per gestire la collezione Qdrant con:
- Esplorazione documenti e chunks
- Ricerca semantica
- Upload e ingest file
- Statistiche e analytics
- Gestione eliminazioni

Usage:
    streamlit run qdrant_dashboard.py
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

# Import dalla pipeline esistente
from smart_ingest import Config, SmartIngestPipeline

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

load_dotenv()

# Configurazione pagina
st.set_page_config(
    page_title="Qdrant Dashboard - Croce Rossa", page_icon="🔍", layout="wide", initial_sidebar_state="expanded"
)

# CSS Custom per rendere più bella la dashboard
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #DC143C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SESSIONE E CACHE
# ============================================================================


@st.cache_resource
def init_config():
    """Inizializza configurazione."""
    return Config()


@st.cache_resource
def init_qdrant_client(_config):
    """Inizializza client Qdrant (cache per sessione)."""
    return QdrantClient(url=_config.qdrant_url, api_key=_config.qdrant_api_key, timeout=120.0)


@st.cache_resource
def init_embeddings(_config):
    """Inizializza embeddings OpenAI."""
    return OpenAIEmbeddings(model=_config.embedding_model, openai_api_key=_config.openai_api_key)


def clear_cache():
    """Pulisce cache dopo modifiche."""
    st.cache_data.clear()


# ============================================================================
# FUNZIONI UTILITY
# ============================================================================


def get_all_documents(client: QdrantClient, collection_name: str) -> pd.DataFrame:
    """Recupera lista di tutti i documenti con statistiche."""
    try:
        # Scroll tutti i punti (limita per performance)
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Limita a 10k punti
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return pd.DataFrame()

        # Estrai documenti unici con metadati
        docs_map = {}

        for point in points:
            payload = point.payload

            # Gestisci sia formato flat che nested
            if "metadata" in payload and isinstance(payload["metadata"], dict):
                metadata = payload["metadata"]
            else:
                metadata = payload

            filename = metadata.get("filename", "Unknown")

            if filename not in docs_map:
                docs_map[filename] = {
                    "filename": filename,
                    "document_type": metadata.get("document_type", "Unknown"),
                    "processed_date": metadata.get("processed_date", "N/A"),
                    "file_hash": metadata.get("file_hash", "N/A"),
                    "chunks": 0,
                    "total_chars": 0,
                }

            docs_map[filename]["chunks"] += 1

            # Conta caratteri
            if "page_content" in payload:
                docs_map[filename]["total_chars"] += len(payload["page_content"])

        df = pd.DataFrame(list(docs_map.values()))

        # Formatta date
        if "processed_date" in df.columns:
            df["processed_date"] = pd.to_datetime(df["processed_date"], errors="coerce")

        # Ordina per data (più recenti prima)
        if "processed_date" in df.columns:
            df = df.sort_values("processed_date", ascending=False)

        return df

    except Exception as e:
        st.error(f"Errore recupero documenti: {e!s}")
        return pd.DataFrame()


def search_document_by_name(client: QdrantClient, collection_name: str, query: str) -> list[dict[str, Any]]:
    """Cerca documenti per nome (match parziale)."""
    try:
        points, _ = client.scroll(collection_name=collection_name, limit=10000, with_payload=True, with_vectors=False)

        results = []
        query_lower = query.lower()

        seen_files = set()

        for point in points:
            payload = point.payload
            metadata = payload.get("metadata", payload)
            filename = metadata.get("filename", "Unknown")

            # Match parziale sul nome
            if query_lower in filename.lower() and filename not in seen_files:
                results.append(
                    {
                        "filename": filename,
                        "document_type": metadata.get("document_type", "Unknown"),
                        "processed_date": metadata.get("processed_date", "N/A"),
                    }
                )
                seen_files.add(filename)

        return results

    except Exception as e:
        st.error(f"Errore ricerca: {e!s}")
        return []


def semantic_search(
    client: QdrantClient, embeddings: OpenAIEmbeddings, collection_name: str, query: str, top_k: int = 10
) -> list[dict[str, Any]]:
    """Ricerca semantica nel contenuto."""
    try:
        # Genera embedding della query
        query_vector = embeddings.embed_query(query)

        # Cerca in Qdrant
        results = client.search(
            collection_name=collection_name, query_vector=query_vector, limit=top_k, with_payload=True
        )

        formatted_results = []
        for hit in results:
            payload = hit.payload
            metadata = payload.get("metadata", payload)

            formatted_results.append(
                {
                    "score": hit.score,
                    "filename": metadata.get("filename", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "chunk_title": metadata.get("chunk_title", "N/A"),
                    "content": payload.get("page_content", "")[:500] + "...",
                    "document_type": metadata.get("document_type", "Unknown"),
                }
            )

        return formatted_results

    except Exception as e:
        st.error(f"Errore semantic search: {e!s}")
        return []


def get_document_chunks(client: QdrantClient, collection_name: str, filename: str) -> list[dict[str, Any]]:
    """Recupera tutti i chunks di un documento specifico."""
    try:
        # Genera varianti del filename
        basename = Path(filename).stem
        extensions = [".md", ".pdf", ".txt", ".xlsx", ".xls", ".doc", ".docx", ".pptx"]
        variants = [basename + ext for ext in extensions] + [filename]

        # Costruisci filtro OR
        conditions = []
        for variant in set(variants):
            conditions.append(models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=variant)))
            conditions.append(models.FieldCondition(key="filename", match=models.MatchValue(value=variant)))

        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(should=conditions),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []
        for point in points:
            payload = point.payload
            metadata = payload.get("metadata", payload)

            chunks.append(
                {
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "chunk_title": metadata.get("chunk_title", "N/A"),
                    "content": payload.get("page_content", ""),
                    "chars": len(payload.get("page_content", "")),
                }
            )

        # Ordina per chunk_id
        chunks.sort(key=lambda x: x.get("chunk_id", 0))

        return chunks

    except Exception as e:
        st.error(f"Errore recupero chunks: {e!s}")
        return []


def delete_document(client: QdrantClient, collection_name: str, filename: str) -> int:
    """Elimina tutti i chunks di un documento."""
    try:
        # Genera varianti
        basename = Path(filename).stem
        extensions = [".md", ".pdf", ".txt", ".xlsx", ".xls", ".doc", ".docx", ".pptx"]
        variants = [basename + ext for ext in extensions] + [filename]

        # Costruisci filtro
        conditions = []
        for variant in set(variants):
            conditions.append(models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=variant)))
            conditions.append(models.FieldCondition(key="filename", match=models.MatchValue(value=variant)))

        # Conta prima
        count = client.count(collection_name=collection_name, count_filter=models.Filter(should=conditions)).count

        # Elimina
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(should=conditions)),
        )

        return count

    except Exception as e:
        st.error(f"Errore eliminazione: {e!s}")
        return 0


# ============================================================================
# PAGINE DASHBOARD
# ============================================================================


def page_overview(config: Config, client: QdrantClient):
    """Pagina Overview con statistiche principali."""
    st.markdown('<div class="main-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)

    # Recupera statistiche
    try:
        collection_info = client.get_collection(config.qdrant_collection)
        docs_df = get_all_documents(client, config.qdrant_collection)

        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="📄 Documenti Totali", value=len(docs_df), delta=None)

        with col2:
            st.metric(label="🧩 Chunks Totali", value=f"{collection_info.points_count:,}", delta=None)

        with col3:
            if not docs_df.empty:
                avg_chunks = docs_df["chunks"].mean()
                st.metric(label="📈 Media Chunks/Doc", value=f"{avg_chunks:.1f}", delta=None)
            else:
                st.metric(label="📈 Media Chunks/Doc", value="0")

        with col4:
            if not docs_df.empty:
                total_chars = docs_df["total_chars"].sum()
                st.metric(label="💾 Caratteri Totali", value=f"{total_chars:,}", delta=None)
            else:
                st.metric(label="💾 Caratteri Totali", value="0")

        st.divider()

        # Grafici
        if not docs_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Chunks per Documento")
                fig = px.bar(
                    docs_df.head(15),
                    x="filename",
                    y="chunks",
                    color="document_type",
                    title="Top 15 Documenti per Numero Chunks",
                    labels={"filename": "Documento", "chunks": "Chunks"},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📁 Distribuzione per Tipo")
                type_counts = docs_df["document_type"].value_counts()
                fig = px.pie(
                    values=type_counts.values, names=type_counts.index, title="Documenti per Tipologia", hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

            # Timeline documenti processati
            if "processed_date" in docs_df.columns:
                st.subheader("📅 Timeline Documenti Processati")
                docs_df_timeline = docs_df.dropna(subset=["processed_date"])
                if not docs_df_timeline.empty:
                    docs_df_timeline["date_only"] = docs_df_timeline["processed_date"].dt.date
                    timeline = docs_df_timeline.groupby("date_only").size().reset_index(name="count")

                    fig = px.line(
                        timeline,
                        x="date_only",
                        y="count",
                        markers=True,
                        title="Documenti Processati nel Tempo",
                        labels={"date_only": "Data", "count": "Numero Documenti"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Info collezione
        st.divider()
        st.subheader("🔧 Informazioni Collezione")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Nome:** {config.qdrant_collection}")
        with col2:
            st.write(f"**Status:** {collection_info.status}")
        with col3:
            st.write(f"**Vectors:** {collection_info.vectors_count:,}")

    except Exception as e:
        st.error(f"Errore caricamento statistiche: {e!s}")


def page_documents(config: Config, client: QdrantClient):
    """Pagina Document Explorer."""
    st.markdown('<div class="main-header">📚 Document Explorer</div>', unsafe_allow_html=True)

    # Ricerca
    search_query = st.text_input("🔍 Cerca documento per nome", placeholder="es: codice-etico")

    # Recupera documenti
    docs_df = get_all_documents(client, config.qdrant_collection)

    if docs_df.empty:
        st.warning("Nessun documento trovato nella collezione.")
        return

    # Filtro per query
    if search_query:
        mask = docs_df["filename"].str.contains(search_query, case=False, na=False)
        docs_df = docs_df[mask]

        if docs_df.empty:
            st.warning(f"Nessun documento trovato con '{search_query}'")
            return

    # Filtri sidebar
    with st.sidebar:
        st.subheader("🎛️ Filtri")

        # Filtro per tipo
        doc_types = ["Tutti", *list(docs_df["document_type"].unique())]
        selected_type = st.selectbox("Tipo Documento", doc_types)

        if selected_type != "Tutti":
            docs_df = docs_df[docs_df["document_type"] == selected_type]

        # Filtro per numero chunks
        if not docs_df.empty:
            min_chunks = int(docs_df["chunks"].min())
            max_chunks = int(docs_df["chunks"].max())

            chunks_range = st.slider(
                "Numero Chunks", min_value=min_chunks, max_value=max_chunks, value=(min_chunks, max_chunks)
            )

            docs_df = docs_df[(docs_df["chunks"] >= chunks_range[0]) & (docs_df["chunks"] <= chunks_range[1])]

    # Mostra risultati
    st.subheader(f"📄 {len(docs_df)} Documenti Trovati")

    # Tabella documenti
    st.dataframe(
        docs_df[["filename", "document_type", "chunks", "processed_date"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "filename": "Nome File",
            "document_type": "Tipo",
            "chunks": st.column_config.NumberColumn("Chunks", format="%d"),
            "processed_date": st.column_config.DatetimeColumn("Data Processamento"),
        },
    )

    # Dettagli documento selezionato
    st.divider()
    st.subheader("🔍 Dettagli Documento")

    selected_doc = st.selectbox("Seleziona documento da ispezionare", options=docs_df["filename"].tolist(), index=0)

    if selected_doc:
        # Bottoni azione
        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("🗑️ Elimina Documento", type="secondary", key="delete_from_explorer"):
                st.session_state["confirm_delete"] = selected_doc

        # Conferma eliminazione
        if st.session_state.get("confirm_delete") == selected_doc:
            st.warning(f"⚠️ Sei sicuro di voler eliminare '{selected_doc}'?")
            col1, col2, _col3 = st.columns([1, 1, 3])

            with col1:
                if st.button("✅ Sì, elimina", type="primary"):
                    deleted_count = delete_document(client, config.qdrant_collection, selected_doc)
                    st.success(f"✅ Eliminati {deleted_count} chunks di '{selected_doc}'")
                    st.session_state.pop("confirm_delete", None)
                    clear_cache()
                    st.rerun()

            with col2:
                if st.button("❌ Annulla"):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

        # Mostra chunks
        chunks = get_document_chunks(client, config.qdrant_collection, selected_doc)

        if chunks:
            st.write(f"**{len(chunks)} chunks trovati**")

            # Mostra ogni chunk
            for chunk in chunks:
                with st.expander(f"Chunk {chunk['chunk_id']}: {chunk['chunk_title']} ({chunk['chars']} chars)"):
                    st.text(chunk["content"])


def page_semantic_search(config: Config, client: QdrantClient, embeddings: OpenAIEmbeddings):
    """Pagina Semantic Search."""
    st.markdown('<div class="main-header">🔎 Semantic Search</div>', unsafe_allow_html=True)

    st.write("Cerca contenuti semanticamente simili alla tua query usando embeddings AI.")

    # Input query
    query = st.text_area(
        "🔍 Inserisci la tua query", placeholder="es: Come gestire un'emergenza sanitaria?", height=100
    )

    _col1, col2 = st.columns([3, 1])

    with col2:
        top_k = st.number_input("Top K risultati", min_value=1, max_value=50, value=10)

    if st.button("🚀 Cerca", type="primary"):
        if not query.strip():
            st.warning("Inserisci una query!")
            return

        with st.spinner("Ricerca in corso..."):
            results = semantic_search(client, embeddings, config.qdrant_collection, query, top_k)

        if results:
            st.success(f"✅ Trovati {len(results)} risultati")

            for i, result in enumerate(results, 1):
                score_color = "🟢" if result["score"] > 0.8 else "🟡" if result["score"] > 0.6 else "🔴"

                with st.expander(f"{score_color} #{i} - {result['filename']} (Score: {result['score']:.3f})"):
                    st.write(f"**Tipo:** {result['document_type']}")
                    st.write(f"**Chunk ID:** {result['chunk_id']}")
                    st.write(f"**Titolo Chunk:** {result['chunk_title']}")
                    st.divider()
                    st.write("**Contenuto:**")
                    st.text(result["content"])
        else:
            st.warning("Nessun risultato trovato.")


def page_upload(config: Config):
    """Pagina Upload & Ingest."""
    st.markdown('<div class="main-header">📤 Upload & Ingest</div>', unsafe_allow_html=True)

    st.write("Carica nuovi documenti e processa automaticamente con la pipeline di ingest.")

    # Upload file
    uploaded_files = st.file_uploader(
        "Scegli file da caricare", type=["pdf", "xlsx", "xls", "doc", "docx", "pptx"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file selezionati**")

        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size:,} bytes)")

        # Opzioni
        col1, col2 = st.columns(2)

        with col1:
            mode = st.radio(
                "Modalità",
                options=["replace", "add-only"],
                help="replace: sostituisce se esiste | add-only: salta se esiste",
            )

        with col2:
            dry_run = st.checkbox("Dry Run (solo simulazione)", value=False)

        if st.button("🚀 Processa File", type="primary"):
            # Crea cartella temporanea
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)

            pipeline = SmartIngestPipeline(config)

            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []

            for idx, uploaded_file in enumerate(uploaded_files):
                # Salva file temporaneo
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")

                # Processa
                result = pipeline.process_file(str(temp_path), mode=mode, dry_run=dry_run)
                results.append(result)

                # Pulisci file temporaneo
                temp_path.unlink()

                progress_bar.progress((idx + 1) / len(uploaded_files))

            # Pulisci cartella temp
            temp_dir.rmdir()

            progress_bar.empty()
            status_text.empty()

            # Mostra risultati
            st.divider()
            st.subheader("📊 Risultati Processing")

            success = [r for r in results if r["status"] == "success"]
            errors = [r for r in results if r["status"] == "error"]
            skipped = [r for r in results if r["status"] == "skipped"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("✅ Successo", len(success))
            with col2:
                st.metric("❌ Errori", len(errors))
            with col3:
                st.metric("⏭️ Saltati", len(skipped))
            with col4:
                total_chunks = sum(r["chunks_inserted"] for r in results)
                st.metric("🧩 Chunks Inseriti", total_chunks)

            # Dettagli
            if errors:
                st.error("**Errori:**")
                for r in errors:
                    st.write(f"- {r['filename']}: {r['error']}")

            if success:
                st.success("**File processati con successo:**")
                for r in success:
                    st.write(f"- {r['filename']}: {r['chunks_inserted']} chunks")

            clear_cache()


def page_delete_manager(config: Config, client: QdrantClient):
    """Pagina Delete Manager."""
    st.markdown('<div class="main-header">🗑️ Delete Manager</div>', unsafe_allow_html=True)

    st.write("Gestisci l'eliminazione di documenti dalla collezione.")

    # Recupera documenti
    docs_df = get_all_documents(client, config.qdrant_collection)

    if docs_df.empty:
        st.warning("Nessun documento nella collezione.")
        return

    # Selezione documento
    selected_doc = st.selectbox("Seleziona documento da eliminare", options=docs_df["filename"].tolist())

    if selected_doc:
        # Info documento
        doc_info = docs_df[docs_df["filename"] == selected_doc].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Tipo:** {doc_info['document_type']}")
        with col2:
            st.write(f"**Chunks:** {doc_info['chunks']}")
        with col3:
            st.write(f"**Caratteri:** {doc_info['total_chars']:,}")

        st.divider()

        # Bottone eliminazione
        if st.button("🗑️ Elimina Documento", type="primary"):
            st.session_state["delete_confirm"] = selected_doc

        # Conferma
        if st.session_state.get("delete_confirm") == selected_doc:
            st.error(
                f"⚠️ **ATTENZIONE:** Stai per eliminare permanentemente '{selected_doc}' ({doc_info['chunks']} chunks)"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Conferma Eliminazione", type="primary"):
                    with st.spinner("Eliminazione in corso..."):
                        deleted_count = delete_document(client, config.qdrant_collection, selected_doc)

                    st.success(f"✅ Eliminati {deleted_count} chunks di '{selected_doc}'")
                    st.session_state.pop("delete_confirm", None)
                    clear_cache()
                    time.sleep(1)
                    st.rerun()

            with col2:
                if st.button("❌ Annulla"):
                    st.session_state.pop("delete_confirm", None)
                    st.rerun()


def page_health_check(config: Config, client: QdrantClient):
    """Pagina Health Check."""
    st.markdown('<div class="main-header">🏥 Health Check</div>', unsafe_allow_html=True)

    st.write("Verifica lo stato della collezione e delle API.")

    if st.button("🔄 Esegui Check", type="primary"):
        results = {}

        # Check Qdrant
        st.subheader("🗄️ Qdrant Connection")
        try:
            collections = client.get_collections()
            st.success(f"✅ Connesso - {len(collections.collections)} collezioni disponibili")
            results["qdrant"] = "OK"
        except Exception as e:
            st.error(f"❌ Errore: {e!s}")
            results["qdrant"] = "FAIL"

        # Check Collection
        st.subheader("📦 Collection Status")
        try:
            info = client.get_collection(config.qdrant_collection)
            st.success(f"✅ Collezione '{config.qdrant_collection}' attiva")
            st.write(f"- **Points:** {info.points_count:,}")
            st.write(f"- **Status:** {info.status}")
            st.write(f"- **Vectors:** {info.vectors_count:,}")
            results["collection"] = "OK"
        except Exception as e:
            st.error(f"❌ Errore: {e!s}")
            results["collection"] = "FAIL"

        # Check OpenAI
        st.subheader("🤖 OpenAI API")
        try:
            embeddings = OpenAIEmbeddings(model=config.embedding_model, openai_api_key=config.openai_api_key)
            test_embedding = embeddings.embed_query("test")
            st.success(f"✅ API attiva - Dimensione embedding: {len(test_embedding)}")
            results["openai"] = "OK"
        except Exception as e:
            st.error(f"❌ Errore: {e!s}")
            results["openai"] = "FAIL"

        # Check LlamaParse
        st.subheader("📄 LlamaParse API")
        if config.llama_api_key:
            st.success("✅ API Key configurata")
            results["llamaparse"] = "OK"
        else:
            st.warning("⚠️ API Key non configurata")
            results["llamaparse"] = "WARN"

        # Summary
        st.divider()
        st.subheader("📊 Summary")

        all_ok = all(v == "OK" for v in results.values() if v != "WARN")

        if all_ok:
            st.success("🎉 Tutti i sistemi sono operativi!")
        else:
            st.error("⚠️ Alcuni sistemi hanno problemi.")


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    """Entry point dashboard."""

    # Sidebar
    st.sidebar.title("🔍 Qdrant Dashboard")
    st.sidebar.markdown("---")

    # Inizializza configurazione
    try:
        config = init_config()
        client = init_qdrant_client(config)
        embeddings = init_embeddings(config)
    except Exception as e:
        st.error(f"❌ Errore inizializzazione: {e!s}")
        st.stop()

    # Menu navigazione
    page = st.sidebar.radio(
        "📑 Navigazione",
        [
            "📊 Overview",
            "📚 Document Explorer",
            "🔎 Semantic Search",
            "📤 Upload & Ingest",
            "🗑️ Delete Manager",
            "🏥 Health Check",
        ],
    )

    st.sidebar.markdown("---")

    # Info collezione
    st.sidebar.subheader("ℹ️ Info Collezione")
    st.sidebar.write(f"**Nome:** {config.qdrant_collection}")

    try:
        info = client.get_collection(config.qdrant_collection)
        st.sidebar.write(f"**Points:** {info.points_count:,}")
        st.sidebar.write(f"**Status:** {info.status}")
    except:
        st.sidebar.warning("Collezione non accessibile")

    st.sidebar.markdown("---")
    st.sidebar.caption("🔴 Croce Rossa Italiana")
    st.sidebar.caption("Dashboard v1.0")

    # Routing pagine
    if page == "📊 Overview":
        page_overview(config, client)
    elif page == "📚 Document Explorer":
        page_documents(config, client)
    elif page == "🔎 Semantic Search":
        page_semantic_search(config, client, embeddings)
    elif page == "📤 Upload & Ingest":
        page_upload(config)
    elif page == "🗑️ Delete Manager":
        page_delete_manager(config, client)
    elif page == "🏥 Health Check":
        page_health_check(config, client)


if __name__ == "__main__":
    main()
