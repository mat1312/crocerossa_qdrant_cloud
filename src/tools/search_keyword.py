"""
Script per cercare chunk contenenti una parola specifica in Qdrant.
Supporta sia ricerca BM25 (sparse) che scroll con filtro locale.

Usage:
    python search_keyword.py paliano
    python search_keyword.py paliano --limit 20
    python search_keyword.py paliano --method scroll
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

load_dotenv()


def search_with_sparse(client: QdrantClient, collection: str, keyword: str, limit: int = 10):
    """
    Cerca usando sparse vectors (BM25) - ottimo per keyword search.
    """
    print(f"\n[BM25] Ricerca per: '{keyword}'")
    print("=" * 60)

    # Genera sparse embedding per la query
    from fastembed import SparseTextEmbedding

    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_result = next(iter(sparse_model.embed([keyword])))

    query_sparse = SparseVector(indices=sparse_result.indices.tolist(), values=sparse_result.values.tolist())

    # Esegui query
    results = client.query_points(
        collection_name=collection, query=query_sparse, using="sparse", limit=limit, with_payload=True
    )

    return results.points


def search_with_scroll(client: QdrantClient, collection: str, keyword: str, limit: int = 100):
    """
    Cerca scrollando tutti i punti e filtrando localmente.
    Piu' lento ma trova match esatti nel testo.
    """
    print(f"\n[SCROLL] Ricerca per: '{keyword}'")
    print("=" * 60)

    keyword_lower = keyword.lower()
    matches = []
    offset = None
    batch_size = 100
    scanned = 0

    while True:
        results, next_offset = client.scroll(
            collection_name=collection, limit=batch_size, offset=offset, with_payload=True, with_vectors=False
        )

        scanned += len(results)

        for point in results:
            payload = point.payload
            content = payload.get("page_content", "")

            if keyword_lower in content.lower():
                matches.append(point)
                if len(matches) >= limit:
                    print(f"  Scansionati {scanned} punti, trovati {len(matches)} match")
                    return matches

        if next_offset is None:
            break
        offset = next_offset
        print(f"  Scansionati {scanned} punti, trovati {len(matches)} match...", end="\r")

    print(f"  Scansionati {scanned} punti, trovati {len(matches)} match")
    return matches


def print_results(results, keyword: str, method: str):
    """Stampa i risultati."""
    print(f"\n[RISULTATI] Trovati {len(results)} chunk con '{keyword}' ({method})")
    print("=" * 60)

    if not results:
        print("[X] Nessun risultato trovato")
        return

    # Raggruppa per documento
    docs = {}
    for point in results:
        payload = point.payload
        metadata = payload.get("metadata", payload)

        filename = metadata.get("filename", "unknown")
        if filename not in docs:
            docs[filename] = []

        docs[filename].append(
            {
                "id": point.id,
                "score": getattr(point, "score", None),
                "page": metadata.get("page_number", "?"),
                "chunk_id": metadata.get("chunk_id", "?"),
                "content_preview": payload.get("page_content", "")[:200],
            }
        )

    print(f"\n[DOCUMENTI] Trovati: {len(docs)}")
    print("-" * 60)

    for filename, chunks in docs.items():
        print(f"\n>>> {filename} ({len(chunks)} chunk)")
        print("-" * 40)

        for chunk in chunks[:5]:  # Max 5 chunk per documento
            score_str = f" | Score: {chunk['score']:.4f}" if chunk["score"] else ""
            print(f"  * Chunk {chunk['chunk_id']} | Pag. {chunk['page']}{score_str}")

            # Evidenzia la keyword nel contenuto
            preview = chunk["content_preview"].replace("\n", " ")
            keyword_lower = keyword.lower()
            preview_lower = preview.lower()

            if keyword_lower in preview_lower:
                idx = preview_lower.find(keyword_lower)
                start = max(0, idx - 50)
                end = min(len(preview), idx + len(keyword) + 50)
                snippet = preview[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(preview):
                    snippet = snippet + "..."
                print(f'     "{snippet}"')
            else:
                print(f'     "{preview[:100]}..."')

        if len(chunks) > 5:
            print(f"  ... e altri {len(chunks) - 5} chunk")


def main():
    parser = argparse.ArgumentParser(description="Cerca keyword in Qdrant")
    parser.add_argument("keyword", help="Parola da cercare")
    parser.add_argument("--limit", type=int, default=50, help="Numero max risultati")
    parser.add_argument(
        "--method",
        choices=["sparse", "scroll", "both"],
        default="both",
        help="Metodo: sparse (BM25), scroll (match esatto), both (default)",
    )
    parser.add_argument("--collection", default=None, help="Nome collection (default da .env)")

    args = parser.parse_args()

    # Connessione
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = args.collection or os.getenv("QDRANT_COLLECTION")

    if not all([qdrant_url, qdrant_api_key, collection]):
        print("[ERRORE] Configura QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION in .env")
        sys.exit(1)

    print("[CONN] Connessione a Qdrant...")
    print(f"   Collection: {collection}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)

    # Verifica collection
    try:
        info = client.get_collection(collection)
        print(f"   Punti totali: {info.points_count}")
    except Exception as e:
        print(f"[ERRORE] {e}")
        sys.exit(1)

    # Esegui ricerca
    if args.method in ["sparse", "both"]:
        try:
            results = search_with_sparse(client, collection, args.keyword, args.limit)
            print_results(results, args.keyword, "BM25 sparse")
        except Exception as e:
            print(f"[WARN] Ricerca sparse fallita: {e}")
            if args.method == "sparse":
                sys.exit(1)

    if args.method in ["scroll", "both"]:
        results = search_with_scroll(client, collection, args.keyword, args.limit)
        print_results(results, args.keyword, "scroll")


if __name__ == "__main__":
    main()
