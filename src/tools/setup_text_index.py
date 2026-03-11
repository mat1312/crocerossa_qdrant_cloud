"""
Setup Text Index con Stemmer Italiano - Script standalone.
Esegui UNA SOLA VOLTA per aggiungere l'indice testuale.

Uso:
    python setup_text_index.py
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Language,
    SnowballLanguage,
    SnowballParams,
    StopwordsSet,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
)

load_dotenv()


def setup_italian_text_index():
    """Configura il text index con stemmer italiano."""

    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    collection = os.getenv("QDRANT_COLLECTION")

    print("=" * 60)
    print("  SETUP TEXT INDEX CON STEMMER ITALIANO")
    print("=" * 60)

    # Verifica collection
    info = client.get_collection(collection)
    print(f"\nCollection: {collection}")
    print(f"Documenti: {info.points_count}")

    # Verifica se esiste già
    if info.payload_schema and "page_content" in info.payload_schema:
        schema = info.payload_schema["page_content"]
        print(f"\n⚠️  Index esistente su 'page_content': {schema}")

        # Se params=None, è un index base senza stemmer
        if schema.params is None:
            print("\n⚠️  L'index attuale NON ha lo stemmer italiano!")
            print("   Per abilitare la ricerca fuzzy (corso→corsi), devo:")
            print("   1. Eliminare l'index vecchio")
            print("   2. Ricreare con stemmer italiano")

            response = input("\nVuoi sostituire l'index? (s/N): ")
            if response.lower() != "s":
                print("Operazione annullata.")
                return

            print("\n🗑️  Eliminando index vecchio...")
            client.delete_payload_index(collection_name=collection, field_name="page_content")
            print("   ✅ Index eliminato!")
        else:
            print("\n✅ L'index ha già i parametri configurati!")
            print("Nessuna modifica necessaria.")
            return

    print("\n📚 Creando TEXT INDEX con stemmer italiano...")

    client.create_payload_index(
        collection_name=collection,
        field_name="page_content",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=40,
            lowercase=True,
            ascii_folding=True,
            stemmer=SnowballParams(type="snowball", language=SnowballLanguage.ITALIAN),
            stopwords=StopwordsSet(languages=[Language.ITALIAN]),
            phrase_matching=True,
        ),
        wait=True,
    )

    print("\n✅ TEXT INDEX configurato con successo!")
    print("   - Stemmer: Snowball Italian")
    print("   - Stopwords: Italian")
    print("   - ASCII folding: abilitato")
    print("   - Phrase matching: abilitato")


if __name__ == "__main__":
    setup_italian_text_index()
