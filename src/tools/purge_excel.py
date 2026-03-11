"""
Purge Excel Script
Scansiona la collection Qdrant ed elimina tutti i chunk appartenenti a file Excel (.xlsx, .xls).
Utile per ripulire dati "sporchi" prima di re-indicizzare con la nuova logica Pandas.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Carica variabili d'ambiente
load_dotenv()

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PurgeExcel")

# Configurazione Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

if not all([QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION]):
    logger.error("❌ Errore: Variabili d'ambiente mancanti (.env)")
    sys.exit(1)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_excel_filenames() -> set[str]:
    """Scansiona la collection per trovare nomi di file Excel univoci."""
    logger.info("🔍 Scansione collection in corso... (potrebbe richiedere tempo)")

    excel_files = set()
    next_offset = None

    while True:
        # Preleva chunk a blocchi (solo payload per leggere il filename)
        records, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION, limit=1000, offset=next_offset, with_payload=True, with_vectors=False
        )

        for record in records:
            # Cerca il filename nei metadati (supporta sia flat che nested per sicurezza)
            payload = record.payload or {}

            # Tenta di leggere 'filename' o 'metadata.filename'
            fname = payload.get("filename")
            if not fname and "metadata" in payload:
                fname = payload["metadata"].get("filename")

            if fname:
                fn_lower = fname.lower()
                if fn_lower.endswith(".xlsx") or fn_lower.endswith(".xls"):
                    excel_files.add(fname)

        if next_offset is None:
            break

    return excel_files


def delete_files(filenames: set[str]):
    """Elimina i file trovati."""
    total = len(filenames)
    if total == 0:
        logger.info("✅ Nessun file Excel trovato. La collection è pulita.")
        return

    print(f"\n⚠️  ATTENZIONE: Stai per eliminare {total} file Excel dalla collection '{QDRANT_COLLECTION}'.")
    print("File trovati:")
    for f in list(filenames)[:10]:  # Mostra solo i primi 10
        print(f" - {f}")
    if total > 10:
        print(f" ... e altri {total - 10}")

    confirm = input("\nSei sicuro di voler procedere? (scrivi 'si' per confermare): ")
    if confirm.lower() != "si":
        print("❌ Operazione annullata.")
        return

    print("\n🗑️  Inizio eliminazione...")

    deleted_count = 0
    for fname in filenames:
        try:
            # Costruisce il filtro per eliminare tutte le varianti
            # (Lo stesso filtro usato nello script di ingest per sicurezza)
            basename = os.path.splitext(fname)[0]
            variants = [fname, basename + ".xlsx", basename + ".xls"]  # Varianti comuni

            conditions = []
            for v in set(variants):
                # Filtra sia nested (nuovo standard) che flat (vecchio standard)
                conditions.append(models.FieldCondition(key="metadata.filename", match=models.MatchValue(value=v)))
                conditions.append(models.FieldCondition(key="filename", match=models.MatchValue(value=v)))

            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(filter=models.Filter(should=conditions)),
            )
            print(f"   ✅ Eliminato: {fname}")
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ Errore su {fname}: {e!s}")

    print(f"\n✨ Pulizia completata! {deleted_count}/{total} file rimossi.")


if __name__ == "__main__":
    print(f"🔌 Connesso a Qdrant: {QDRANT_URL}")
    print(f"📂 Collection: {QDRANT_COLLECTION}")

    try:
        # 1. Trova
        files_to_delete = get_excel_filenames()

        # 2. Elimina
        delete_files(files_to_delete)

    except Exception as e:
        logger.error(f"❌ Errore critico: {e}")
