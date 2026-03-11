"""
Generate Golden Dataset - Script per la generazione di un dataset sintetico per valutazione RAG
================================================================================

Questo script utilizza Ragas per generare automaticamente un Golden Dataset
(domande/risposte/contesti) a partire da documenti PDF, senza bisogno di
annotazioni manuali.

Requisiti:
    pip install ragas pypdf langchain-community langchain-openai openai

Configurazione:
    Creare un file .env con:
    OPENAI_API_KEY=sk-...

Usage:
    python generate_dataset.py                          # Usa PDF di default
    python generate_dataset.py --pdf path/to/file.pdf   # Specifica PDF
    python generate_dataset.py --size 50                # Genera 50 esempi
    python generate_dataset.py --output my_dataset.csv  # Nome output custom

Autore: AI Engineer Pipeline
Versione: 2.1 (Ragas 0.4.1 - Custom transforms senza HeadlineSplitter per PDF italiani)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

# ============================================================================
# CONFIGURAZIONE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()


# ============================================================================
# CONFIGURAZIONE DEFAULT
# ============================================================================

# Path PDF di default (modifica questo percorso con il tuo documento)
DEFAULT_PDF_PATH = "cri_docs/Codice-etico.pdf"

# Numero di esempi da generare di default
DEFAULT_TESTSET_SIZE = 20

# Output file
DEFAULT_OUTPUT_FILE = "golden_dataset.csv"


# ============================================================================
# FUNZIONE PRINCIPALE DI GENERAZIONE
# ============================================================================


def generate_golden_dataset(
    pdf_path: str, testset_size: int = DEFAULT_TESTSET_SIZE, output_file: str = DEFAULT_OUTPUT_FILE
) -> pd.DataFrame:
    """
    Genera un Golden Dataset sintetico a partire da un documento PDF.

    Utilizza transforms personalizzate che NON includono HeadlineSplitter,
    permettendo di processare PDF italiani senza struttura titoli riconoscibile.

    Args:
        pdf_path: Percorso al file PDF da processare
        testset_size: Numero di esempi da generare
        output_file: Nome del file CSV di output

    Returns:
        DataFrame con le colonne generate da Ragas
    """

    # -------------------------------------------------------------------------
    # STEP 1: Verifica prerequisiti
    # -------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("AVVIO GENERAZIONE GOLDEN DATASET (Ragas 0.4.1)")
    logger.info("=" * 60)

    # Verifica API key OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise OSError("OPENAI_API_KEY non trovata! Aggiungi la chiave nel file .env")
    logger.info("API Key OpenAI trovata")

    # Verifica esistenza PDF
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF non trovato: {pdf_path}")
    logger.info(f"PDF selezionato: {pdf_path.name}")

    # -------------------------------------------------------------------------
    # STEP 2: Caricamento documento con LangChain
    # -------------------------------------------------------------------------

    logger.info("\nCaricamento documento PDF...")

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        logger.info(f"Caricate {len(documents)} pagine dal PDF")

        # Log delle prime righe per debug
        if documents:
            preview = documents[0].page_content[:200].replace("\n", " ")
            logger.info(f"   Preview: {preview}...")

    except ImportError as err:
        raise ImportError("PyPDFLoader non disponibile! Installa con: pip install pypdf langchain-community") from err
    except Exception as e:
        logger.error(f"Errore caricamento PDF: {e!s}")
        raise

    # -------------------------------------------------------------------------
    # STEP 3: Configurazione LLM e Embeddings (API Moderna 0.4.1)
    # -------------------------------------------------------------------------

    logger.info("\nConfigurazione modelli AI (Ragas 0.4.1 - llm_factory)...")

    try:
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory

        # Crea client OpenAI
        client = openai.OpenAI(api_key=openai_api_key)

        # Usa llm_factory (API moderna, non deprecata)
        generator_llm = llm_factory(model="gpt-4o-mini", client=client)

        # Usa embedding_factory (API moderna, non deprecata)
        generator_embeddings = embedding_factory(model="text-embedding-3-small", client=client)

        logger.info("Modelli configurati (API moderna):")
        logger.info("   - LLM: gpt-4o (llm_factory)")
        logger.info("   - Embeddings: text-embedding-3-small (embedding_factory)")

    except ImportError as e:
        logger.error(f"Dipendenze mancanti: {e!s}")
        logger.error("   Installa con: pip install ragas openai")
        raise
    except Exception as e:
        logger.error(f"Errore configurazione LLM: {e!s}")
        raise

    # -------------------------------------------------------------------------
    # STEP 4: Creazione KnowledgeGraph dai documenti LangChain
    # -------------------------------------------------------------------------

    logger.info("\nCreazione KnowledgeGraph dai documenti...")

    try:
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType

        # Crea KnowledgeGraph vuoto
        kg = KnowledgeGraph()

        # Aggiungi ogni documento come nodo DOCUMENT
        for _i, doc in enumerate(documents):
            node = Node(
                type=NodeType.DOCUMENT, properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
            kg.nodes.append(node)

        logger.info(f"KnowledgeGraph creato con {len(kg.nodes)} nodi documento")

    except Exception as e:
        logger.error(f"Errore creazione KnowledgeGraph: {e!s}")
        raise

    # -------------------------------------------------------------------------
    # STEP 5: Applicazione Transforms PERSONALIZZATE (senza HeadlineSplitter!)
    # -------------------------------------------------------------------------

    logger.info("\nApplicazione transforms personalizzate (NO HeadlineSplitter)...")

    try:
        from ragas.testset.transforms import (
            CosineSimilarityBuilder,
            EmbeddingExtractor,
            KeyphrasesExtractor,
            SummaryExtractor,
            apply_transforms,
        )

        # Transforms personalizzate SENZA HeadlinesExtractor e HeadlineSplitter
        # Questo evita l'errore "headlines property not found"
        custom_transforms = [
            # Step 1: Estrai embeddings del contenuto
            EmbeddingExtractor(embedding_model=generator_embeddings),
            # Step 2: Estrai summary dei contenuti
            SummaryExtractor(llm=generator_llm),
            # Step 3: Estrai keyphrases
            KeyphrasesExtractor(llm=generator_llm),
            # Step 4: Estrai embeddings del SUMMARY (richiesto per persona generation!)
            EmbeddingExtractor(
                embedding_model=generator_embeddings,
                property_name="summary_embedding",  # Output: summary_embedding
                embed_property_name="summary",  # Input: leggi da summary
            ),
            # Step 5: Costruisci relazioni basate su similarita' coseno
            CosineSimilarityBuilder(threshold=0.6),
        ]

        logger.info("Transforms personalizzate configurate:")
        logger.info("   - EmbeddingExtractor (page_content -> embedding)")
        logger.info("   - SummaryExtractor")
        logger.info("   - KeyphrasesExtractor")
        logger.info("   - EmbeddingExtractor (summary -> summary_embedding)")
        logger.info("   - CosineSimilarityBuilder")
        logger.info("   (HeadlineSplitter ESCLUSO)")

        # Applica le transforms al KnowledgeGraph
        logger.info("\nApplicazione transforms in corso...")
        apply_transforms(kg, custom_transforms)

        logger.info(f"Transforms applicate: {len(kg.nodes)} nodi, {len(kg.relationships)} relazioni")

    except ImportError as e:
        logger.warning(f"Alcune transforms non disponibili: {e!s}")
        logger.info("Continuo con transforms base...")

        # Fallback minimale
        try:
            from ragas.testset.transforms import (
                EmbeddingExtractor,
                apply_transforms,
            )

            minimal_transforms = [
                EmbeddingExtractor(embedding_model=generator_embeddings),
            ]

            apply_transforms(kg, minimal_transforms)
            logger.info("Transforms minimali applicate")

        except Exception as e2:
            logger.warning(f"Anche transforms minimali fallite: {e2!s}")

    except Exception as e:
        logger.warning(f"Errore applicazione transforms: {e!s}")
        logger.info("Continuo con KnowledgeGraph base...")

    # -------------------------------------------------------------------------
    # STEP 6: Configurazione TestsetGenerator con KnowledgeGraph
    # -------------------------------------------------------------------------

    logger.info("\nConfigurazione TestsetGenerator...")

    try:
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution

        # Crea il generatore CON il KnowledgeGraph gia' popolato
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,  # Usa il KG con transforms gia' applicate
        )

        # Ottieni la distribuzione di query di default
        query_distribution = default_query_distribution(generator_llm)

        logger.info("TestsetGenerator configurato con KnowledgeGraph personalizzato")

    except ImportError as e:
        logger.error(f"Ragas non installato: {e!s}")
        raise
    except Exception as e:
        logger.error(f"Errore configurazione Ragas: {e!s}")
        raise

    # -------------------------------------------------------------------------
    # STEP 7: Generazione del Testset
    # -------------------------------------------------------------------------

    logger.info(f"\nGenerazione {testset_size} esempi in corso...")
    logger.info("   (Questo puo' richiedere alcuni minuti...)")

    start_time = datetime.now()

    try:
        # Genera il testset usando generate() con il KG gia' pronto
        dataset = generator.generate(testset_size=testset_size, query_distribution=query_distribution)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generazione completata in {elapsed:.1f} secondi")

    except Exception as e:
        logger.error(f"Errore durante la generazione: {e!s}")
        logger.error("   Possibili cause:")
        logger.error("   - Il documento potrebbe essere troppo corto")
        logger.error("   - Problemi di rate limit con OpenAI")
        logger.error("   - Contenuto non adatto (es. solo tabelle/immagini)")
        import traceback

        traceback.print_exc()
        raise

    # -------------------------------------------------------------------------
    # STEP 8: Conversione in DataFrame e salvataggio
    # -------------------------------------------------------------------------

    logger.info("\nSalvataggio dataset...")

    try:
        # Converti in pandas DataFrame (metodo ufficiale Ragas)
        df = dataset.to_pandas()

        logger.info(f"   Colonne generate: {list(df.columns)}")

        # Aggiungi metadata utili
        df["generated_at"] = datetime.now().isoformat()
        df["source_document"] = pdf_path.name

        # Salva in CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(f"Dataset salvato: {output_path.absolute()}")
        logger.info(f"   Righe generate: {len(df)}")

        # Preview delle prime domande
        logger.info("\nPreview delle prime 3 domande generate:")
        # Trova la colonna corretta per la domanda
        question_col = None
        for col in ["user_input", "question", "query"]:
            if col in df.columns:
                question_col = col
                break

        if question_col:
            for i, row in df.head(3).iterrows():
                question = str(row[question_col])[:100]
                logger.info(f"   {i + 1}. {question}...")

        return df

    except Exception as e:
        logger.error(f"Errore salvataggio: {e!s}")
        raise


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Entry point dello script con supporto CLI."""

    parser = argparse.ArgumentParser(
        description="Genera un Golden Dataset sintetico per valutazione RAG usando Ragas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python generate_dataset.py
  python generate_dataset.py --pdf docs/manuale.pdf --size 50
  python generate_dataset.py --output test_dataset.csv
        """,
    )

    parser.add_argument(
        "--pdf", type=str, default=DEFAULT_PDF_PATH, help=f"Percorso al file PDF (default: {DEFAULT_PDF_PATH})"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_TESTSET_SIZE,
        help=f"Numero di esempi da generare (default: {DEFAULT_TESTSET_SIZE})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Nome file CSV di output (default: {DEFAULT_OUTPUT_FILE})",
    )

    args = parser.parse_args()

    try:
        # Esegui generazione
        generate_golden_dataset(pdf_path=args.pdf, testset_size=args.size, output_file=args.output)

        logger.info("\n" + "=" * 60)
        logger.info("GENERAZIONE COMPLETATA CON SUCCESSO!")
        logger.info("=" * 60)
        logger.info("\nProssimo step: esegui la valutazione con:")
        logger.info(f"  python evaluate_rag.py --dataset {args.output}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n{e!s}")
        logger.error("\nVerifica che il percorso al PDF sia corretto.")
        return 1

    except OSError as e:
        logger.error(f"\n{e!s}")
        return 1

    except Exception as e:
        logger.error(f"\nErrore imprevisto: {e!s}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
