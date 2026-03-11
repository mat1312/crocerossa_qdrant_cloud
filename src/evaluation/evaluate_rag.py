"""
Evaluate RAG - Script per la valutazione delle performance del sistema RAG
================================================================================

Questo script utilizza Ragas per calcolare metriche di qualita' su un sistema
RAG esistente, utilizzando un Golden Dataset generato in precedenza.

Metriche calcolate (API Collections 0.4.1):
    - ContextRecall: Quanto il retriever recupera contesti rilevanti
    - ContextPrecision: Quanto i contesti recuperati sono pertinenti
    - Faithfulness: Quanto la risposta e' fedele al contesto (no allucinazioni)
    - ResponseRelevancy: Quanto la risposta e' pertinente alla domanda

Requisiti:
    pip install ragas langchain-openai pandas openpyxl

Configurazione:
    Creare un file .env con:
    OPENAI_API_KEY=sk-...

Usage:
    python evaluate_rag.py                                    # Valutazione standard
    python evaluate_rag.py --dataset my_golden_dataset.csv    # Dataset custom
    python evaluate_rag.py --output results.xlsx              # Output Excel

Autore: AI Engineer Pipeline
Versione: 2.0 (compatibile con Ragas 0.4.1 - API collections moderne)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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

# File input (generato da generate_dataset.py)
DEFAULT_DATASET_FILE = "golden_dataset.csv"

# File output risultati
DEFAULT_OUTPUT_FILE = "evaluation_results.csv"


# ============================================================================
# PLACEHOLDER: SISTEMA RAG DA VALUTARE
# ============================================================================


def my_rag_system(query: str) -> tuple[str, list[str]]:
    """
    PLACEHOLDER - Sostituisci questa funzione con la chiamata al tuo sistema RAG!

    Questa funzione deve:
    1. Ricevere una domanda (query) in input
    2. Eseguire il retrieval sui tuoi documenti
    3. Generare una risposta usando il tuo LLM
    4. Restituire sia la risposta che i contesti recuperati

    Args:
        query: La domanda da porre al sistema RAG

    Returns:
        Tuple contenente:
        - answer (str): La risposta generata dal sistema
        - contexts (list[str]): Lista dei chunk di testo recuperati

    Esempio di implementazione reale con Qdrant:

        from qdrant_client import QdrantClient
        from langchain_openai import ChatOpenAI

        # 1. Retrieval
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results = client.search(collection_name="crocerossa_docs", query_vector=embed(query), limit=5)
        contexts = [r.payload["page_content"] for r in results]

        # 2. Generation
        llm = ChatOpenAI(model="gpt-4o")
        prompt = f"Rispondi alla domanda basandoti sul contesto:\\n{contexts}\\n\\nDomanda: {query}"
        answer = llm.invoke(prompt).content

        return answer, contexts
    """

    # =========================================================================
    # IMPLEMENTAZIONE DEMO (SOSTITUISCI CON IL TUO CODICE!)
    # =========================================================================

    # Questa e' solo una simulazione per testare la pipeline.
    # In produzione, qui va la chiamata reale al tuo sistema RAG.

    logger.warning("Usando implementazione DEMO di my_rag_system!")
    logger.warning("   Sostituisci questa funzione con il tuo sistema RAG reale.")

    # Risposta simulata (demo)
    demo_answer = (
        f"Questa e' una risposta di esempio alla domanda: '{query[:50]}...'. "
        "In produzione, questa sarebbe generata dal tuo LLM."
    )

    # Contesti simulati (demo)
    demo_contexts = [
        "Contesto di esempio 1: La Croce Rossa Italiana e' un'associazione di volontariato...",
        "Contesto di esempio 2: Il codice etico definisce i principi fondamentali...",
    ]

    return demo_answer, demo_contexts


# ============================================================================
# FUNZIONE DI VALUTAZIONE
# ============================================================================


def evaluate_rag_system(
    dataset_file: str = DEFAULT_DATASET_FILE, output_file: str = DEFAULT_OUTPUT_FILE, verbose: bool = True
) -> pd.DataFrame:
    """
    Valuta un sistema RAG usando il Golden Dataset e le metriche Ragas.

    Utilizza l'API moderna di Ragas 0.4.1 con metriche dal modulo collections.

    Args:
        dataset_file: Percorso al CSV con il Golden Dataset
        output_file: Percorso per il file di output con i risultati
        verbose: Se True, mostra dettagli durante la valutazione

    Returns:
        DataFrame con i risultati della valutazione
    """

    # -------------------------------------------------------------------------
    # STEP 1: Verifica prerequisiti
    # -------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("AVVIO VALUTAZIONE SISTEMA RAG (Ragas 0.4.1)")
    logger.info("=" * 60)

    # Verifica API key OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise OSError("OPENAI_API_KEY non trovata! Aggiungi la chiave nel file .env")
    logger.info("API Key OpenAI trovata")

    # Verifica esistenza dataset
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden Dataset non trovato: {dataset_path}\n   Esegui prima: python generate_dataset.py"
        )
    logger.info(f"Dataset: {dataset_path.name}")

    # -------------------------------------------------------------------------
    # STEP 2: Caricamento Golden Dataset
    # -------------------------------------------------------------------------

    logger.info("\nCaricamento Golden Dataset...")

    try:
        df_golden = pd.read_csv(dataset_path)
        logger.info(f"Caricate {len(df_golden)} domande dal dataset")

        # Identifica le colonne corrette (Ragas puo' usare nomi diversi)
        question_col = None
        ground_truth_col = None

        # Cerca colonna domande
        for col in ["question", "user_input", "query"]:
            if col in df_golden.columns:
                question_col = col
                break

        # Cerca colonna ground truth
        for col in ["ground_truth", "reference", "expected_answer", "reference_answer"]:
            if col in df_golden.columns:
                ground_truth_col = col
                break

        if not question_col:
            raise ValueError(
                f"Colonna 'question' non trovata nel dataset!\n   Colonne disponibili: {list(df_golden.columns)}"
            )

        logger.info(f"   Colonna domande: {question_col}")
        logger.info(f"   Colonna ground truth: {ground_truth_col or 'Non trovata (verra usata risposta RAG)'}")

    except Exception as e:
        logger.error(f"Errore caricamento dataset: {e!s}")
        raise

    # -------------------------------------------------------------------------
    # STEP 3: Esecuzione query sul sistema RAG
    # -------------------------------------------------------------------------

    logger.info("\nInterrogazione sistema RAG...")

    evaluation_samples = []
    total_questions = len(df_golden)

    for idx, row in df_golden.iterrows():
        question = row[question_col]
        ground_truth = row.get(ground_truth_col, "") if ground_truth_col else ""

        if verbose:
            logger.info(f"\n   [{idx + 1}/{total_questions}] Elaborazione domanda...")
            logger.info(f"   Q: {question[:80]}...")

        try:
            # Chiama il sistema RAG
            answer, contexts = my_rag_system(question)

            # Raccogli dati per la valutazione usando SingleTurnSample
            evaluation_samples.append(
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": ground_truth if ground_truth else answer,
                }
            )

            if verbose:
                logger.info(f"   A: {answer[:80]}...")
                logger.info(f"   Contesti recuperati: {len(contexts)}")

        except Exception as e:
            logger.error(f"   Errore elaborando domanda {idx + 1}: {e!s}")
            continue

    if not evaluation_samples:
        raise RuntimeError("Nessuna domanda elaborata con successo!")

    logger.info(f"\nElaborate {len(evaluation_samples)}/{total_questions} domande")

    # -------------------------------------------------------------------------
    # STEP 4: Configurazione metriche Ragas (API Collections 0.4.1)
    # -------------------------------------------------------------------------

    logger.info("\nConfigurazione metriche Ragas (Collections API 0.4.1)...")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics._answer_relevance import ResponseRelevancy
        from ragas.metrics._context_precision import ContextPrecision

        # Import metriche dal modulo collections (API moderna)
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness

        # LLM per la valutazione
        eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key))

        # Embeddings per ResponseRelevancy
        eval_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        )

        # Istanzia le metriche con LLM/embeddings
        metrics = [
            ContextRecall(llm=eval_llm),
            ContextPrecision(llm=eval_llm),
            Faithfulness(llm=eval_llm),
            ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ]

        logger.info("Metriche configurate:")
        for m in metrics:
            logger.info(f"   - {m.name}")

    except ImportError as e:
        logger.warning(f"Import collections fallito, provo API legacy: {e!s}")
        # Fallback ad API legacy se collections non disponibile
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import EvaluationDataset, evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key))

            eval_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
            )

            metrics = [context_recall, context_precision, faithfulness, answer_relevancy]
            logger.info("Metriche configurate (API legacy):")
            for m in metrics:
                logger.info(f"   - {m.name}")

        except ImportError as e2:
            logger.error(f"Ragas non installato correttamente: {e2!s}")
            logger.error("   Installa con: pip install ragas")
            raise

    # -------------------------------------------------------------------------
    # STEP 5: Creazione EvaluationDataset ed Esecuzione valutazione
    # -------------------------------------------------------------------------

    logger.info("\nEsecuzione valutazione (puo' richiedere alcuni minuti)...")

    start_time = datetime.now()

    try:
        # Crea EvaluationDataset da lista di dict
        eval_dataset = EvaluationDataset.from_list(evaluation_samples)

        # Esegui valutazione
        results = evaluate(dataset=eval_dataset, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Valutazione completata in {elapsed:.1f} secondi")

    except Exception as e:
        logger.error(f"Errore durante la valutazione: {e!s}")
        import traceback

        traceback.print_exc()
        raise

    # -------------------------------------------------------------------------
    # STEP 6: Elaborazione e salvataggio risultati
    # -------------------------------------------------------------------------

    logger.info("\nElaborazione risultati...")

    try:
        # Converti in DataFrame
        df_results = results.to_pandas()

        # Calcola metriche aggregate
        avg_scores = {}
        metric_columns = [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "response_relevancy",
        ]

        logger.info("\n" + "=" * 60)
        logger.info("RISULTATI VALUTAZIONE")
        logger.info("=" * 60)

        for col in metric_columns:
            if col in df_results.columns:
                avg_score = df_results[col].mean()
                avg_scores[col] = avg_score

                # Indicatore in base al punteggio
                indicator = "[OK]" if avg_score >= 0.7 else "[--]" if avg_score >= 0.5 else "[!!]"
                logger.info(f"   {indicator} {col}: {avg_score:.3f}")

        # Punteggio medio globale
        global_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        logger.info("-" * 40)
        logger.info(f"   PUNTEGGIO MEDIO: {global_avg:.3f}")

        # Aggiungi timestamp e metadata
        df_results["evaluated_at"] = datetime.now().isoformat()

        # Salva in formato appropriato
        output_path = Path(output_file)

        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            # Salva come Excel con formattazione
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_results.to_excel(writer, sheet_name="Dettaglio", index=False)

                # Aggiungi foglio riepilogo
                df_summary = pd.DataFrame([{"Metrica": k, "Punteggio": v} for k, v in avg_scores.items()])
                df_summary.to_excel(writer, sheet_name="Riepilogo", index=False)

            logger.info(f"\nRisultati salvati: {output_path.absolute()}")
            logger.info("   Fogli: 'Dettaglio' (ogni domanda), 'Riepilogo' (medie)")
        else:
            # Salva come CSV
            df_results.to_csv(output_path, index=False, encoding="utf-8")

            # Salva anche riepilogo separato
            summary_path = output_path.with_name(output_path.stem + "_summary.csv")
            df_summary = pd.DataFrame(
                [
                    {
                        "Metrica": k,
                        "Punteggio": v,
                        "Valutazione": "Buono" if v >= 0.7 else "Sufficiente" if v >= 0.5 else "Da migliorare",
                    }
                    for k, v in avg_scores.items()
                ]
            )
            df_summary.to_csv(summary_path, index=False, encoding="utf-8")

            logger.info("\nRisultati salvati:")
            logger.info(f"   - Dettaglio: {output_path.absolute()}")
            logger.info(f"   - Riepilogo: {summary_path.absolute()}")

        return df_results

    except Exception as e:
        logger.error(f"Errore salvataggio: {e!s}")
        raise


# ============================================================================
# INTERPRETAZIONE RISULTATI
# ============================================================================


def print_interpretation():
    """Stampa una guida per interpretare i risultati."""

    print("""
================================================================================
GUIDA ALL'INTERPRETAZIONE DEI RISULTATI
================================================================================

* CONTEXT RECALL (Recupero contesto)
   Misura: Quanto il retriever recupera i contesti rilevanti
   0.8-1.0: Eccellente - Il retriever trova quasi tutti i contesti utili
   0.5-0.8: Buono - Recupera la maggior parte dei contesti
   <0.5: Da migliorare - Molti contesti rilevanti non vengono recuperati

   -> Se basso: Migliora embeddings, aumenta k, riduci chunk size

* CONTEXT PRECISION (Precisione contesto)
   Misura: Quanto i contesti recuperati sono effettivamente utili
   0.8-1.0: Eccellente - Quasi tutti i contesti recuperati sono utili
   0.5-0.8: Buono - La maggior parte dei contesti e' pertinente
   <0.5: Da migliorare - Troppi contesti irrilevanti

   -> Se basso: Migliora reranking, affina chunking, filtra per metadata

* FAITHFULNESS (Fedelta')
   Misura: Quanto la risposta si basa sui contesti (no allucinazioni)
   0.8-1.0: Eccellente - Risposta completamente basata sui fatti
   0.5-0.8: Buono - Qualche inferenza ma sostanzialmente corretta
   <0.5: Critico - Alte probabilita' di allucinazioni

   -> Se basso: Modifica prompt, usa modelli piu' robusti, aggiungi retrieval check

* RESPONSE RELEVANCY (Pertinenza risposta)
   Misura: Quanto la risposta e' pertinente alla domanda
   0.8-1.0: Eccellente - Risposta perfettamente centrata
   0.5-0.8: Buono - Risposta utile ma non perfetta
   <0.5: Da migliorare - Risposta fuori tema

   -> Se basso: Migliora prompt, usa Chain-of-Thought, affina retrieval

================================================================================
""")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Entry point dello script con supporto CLI."""

    parser = argparse.ArgumentParser(
        description="Valuta un sistema RAG usando il Golden Dataset e metriche Ragas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python evaluate_rag.py
  python evaluate_rag.py --dataset my_dataset.csv
  python evaluate_rag.py --output results.xlsx --interpret
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_FILE,
        help=f"Percorso al Golden Dataset CSV (default: {DEFAULT_DATASET_FILE})",
    )

    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_FILE, help=f"Nome file di output (default: {DEFAULT_OUTPUT_FILE})"
    )

    parser.add_argument("--interpret", action="store_true", help="Mostra guida all'interpretazione dei risultati")

    parser.add_argument("--quiet", action="store_true", help="Modalita' silenziosa (meno output)")

    args = parser.parse_args()

    try:
        # Esegui valutazione
        evaluate_rag_system(dataset_file=args.dataset, output_file=args.output, verbose=not args.quiet)

        logger.info("\n" + "=" * 60)
        logger.info("VALUTAZIONE COMPLETATA CON SUCCESSO!")
        logger.info("=" * 60)

        # Mostra interpretazione se richiesta
        if args.interpret:
            print_interpretation()
        else:
            logger.info("\nTip: Usa --interpret per la guida ai risultati")

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n{e!s}")
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
