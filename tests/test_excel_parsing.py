"""Test per verificare il parsing Excel con Pandas prima dell'ingest."""

import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.ingest_volontariato import DocumentParser, HybridConfig


def test_competenze_territoriali():
    """Testa parsing del file Competenze Territoriali."""
    filepath = Path(__file__).parent.parent / "new_cri_docs" / "Competenze Territoriali Comitati CRI Lazio (1).xlsx"

    if not filepath.exists():
        # Prova anche docs_onlyvolontario
        filepath = Path(__file__).parent.parent / "docs_onlyvolontario" / "Competenze Territoriali Comitati CRI Lazio (1).xlsx"

    if not filepath.exists():
        print(f"SKIP: file non trovato")
        return

    config = HybridConfig()
    parser = DocumentParser(config)
    result = parser.parse_file(str(filepath))

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 100, f"Expected >100 rows, got {len(result)}"

    print(f"\n{'='*60}")
    print(f"COMPETENZE TERRITORIALI: {len(result)} righe")
    print(f"{'='*60}")

    # Primi 5 esempi
    for i, row in enumerate(result[:5]):
        print(f"\n--- Riga {i+1} ({len(row)} chars) ---")
        print(row)

    # Verifica contenuto
    sample = result[0]
    assert "comune di" in sample.lower(), f"Manca 'comune di' in: {sample[:100]}"
    assert "comitato cri" in sample.lower(), f"Manca 'Comitato CRI' in: {sample[:100]}"

    # Verifica che non ci siano righe troppo corte
    short = [r for r in result if len(r) < 50]
    print(f"\nRighe corte (<50 chars): {len(short)}")
    for r in short[:3]:
        print(f"  '{r}'")

    print(f"\nOK - COMPETENZE TERRITORIALI: OK ({len(result)} righe)")


def test_delegati_territoriali():
    """Testa parsing del file Delegati Territoriali."""
    filepath = Path(__file__).parent.parent / "new_cri_docs" / "Delegati Territoriali.xlsx"

    if not filepath.exists():
        filepath = Path(__file__).parent.parent / "docs_onlyvolontario" / "Delegati Territoriali.xlsx"

    if not filepath.exists():
        print(f"SKIP: file non trovato")
        return

    config = HybridConfig()
    parser = DocumentParser(config)
    result = parser.parse_file(str(filepath))

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 50, f"Expected >50 rows, got {len(result)}"

    print(f"\n{'='*60}")
    print(f"DELEGATI TERRITORIALI: {len(result)} righe")
    print(f"{'='*60}")

    # Primi 5 esempi
    for i, row in enumerate(result[:5]):
        print(f"\n--- Riga {i+1} ({len(row)} chars) ---")
        print(row)

    # Verifica contenuto
    sample = result[0]
    assert "comitato cri" in sample.lower(), f"Manca 'Comitato CRI' in: {sample[:100]}"
    assert "presidente" in sample.lower(), f"Manca 'presidente' in: {sample[:100]}"

    # Verifica che i delegati compaiano
    has_delegato = any("delegato" in r.lower() for r in result[:10])
    assert has_delegato, "Nessun delegato trovato nei primi 10 risultati"

    print(f"\nOK - DELEGATI TERRITORIALI: OK ({len(result)} righe)")


if __name__ == "__main__":
    print("Testing Excel parsing per ingest_volontariato.py\n")
    test_competenze_territoriali()
    print()
    test_delegati_territoriali()
    print("\n" + "="*60)
    print("TUTTI I TEST PASSATI")
    print("="*60)
