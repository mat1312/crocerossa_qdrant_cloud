"""Tests for MetadataExtractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingest"))

from smart_ingest_hybrid import MetadataExtractor


class TestExtractYear:
    """Tests for extract_year. Note: uses \\b word boundary regex,
    so underscore-adjacent years (e.g., _2017.) don't match because
    _ is a word character. This is known behavior."""

    def test_year_with_dash_separator(self):
        assert MetadataExtractor.extract_year("Regolamento-2017.pdf") == 2017

    def test_year_with_space(self):
        assert MetadataExtractor.extract_year("DL 2024 documento.pdf") == 2024

    def test_no_year(self):
        assert MetadataExtractor.extract_year("codice-etico.pdf") is None

    def test_year_1900s(self):
        assert MetadataExtractor.extract_year("Legge-1948.pdf") == 1948

    def test_four_digits_not_year(self):
        assert MetadataExtractor.extract_year("doc-1899-test.pdf") is None

    def test_multiple_years_returns_first(self):
        result = MetadataExtractor.extract_year("aggiornamento 2020 rev 2023.pdf")
        assert result == 2020

    def test_year_at_start(self):
        assert MetadataExtractor.extract_year("2025-statuto.pdf") == 2025

    def test_underscore_year_does_not_match(self):
        # Known limitation: \b doesn't match between _ and digit
        assert MetadataExtractor.extract_year("Regolamento_2017.pdf") is None

    def test_year_adjacent_to_text_no_match(self):
        # "marzo2026" has no word boundary between text and digits
        assert MetadataExtractor.extract_year("Codice-del-Volontariato_REV_2-del-03marzo2026.pdf") is None

    def test_year_separated_by_dash(self):
        assert MetadataExtractor.extract_year("Codice-del-Volontariato-REV-2-del-03-marzo-2026.pdf") == 2026


class TestInferCategory:
    def test_statuto(self):
        assert MetadataExtractor.infer_category("Statuto_CRI_2024.pdf") == "Statuto"

    def test_etica(self):
        assert MetadataExtractor.infer_category("Codice-etico.pdf") == "Etica"

    def test_normativa_dlgs(self):
        assert MetadataExtractor.infer_category("dlgs_178_2012.pdf") == "Normativa"

    def test_normativa_decreto(self):
        assert MetadataExtractor.infer_category("Decreto_presidente.pdf") == "Normativa"

    def test_regolamento(self):
        assert MetadataExtractor.infer_category("Regolamento_corsi.pdf") == "Regolamento"

    def test_linee_guida(self):
        assert MetadataExtractor.infer_category("Linee_guida_emergenza.pdf") == "Linee Guida"

    def test_manuale(self):
        assert MetadataExtractor.infer_category("Manuale_operativo.pdf") == "Linee Guida"

    def test_sanitario(self):
        assert MetadataExtractor.infer_category("donatori_sangue.pdf") == "Sanitario/Donazioni"

    def test_formazione(self):
        assert MetadataExtractor.infer_category("formazione_volontari.pdf") == "Formazione"

    def test_organizzazione(self):
        assert MetadataExtractor.infer_category("delegati_nazionali.pdf") == "Organizzazione"

    def test_fallback(self):
        assert MetadataExtractor.infer_category("documento_generico.pdf") == "Altro"

    def test_case_insensitive(self):
        assert MetadataExtractor.infer_category("STATUTO_NAZIONALE.PDF") == "Statuto"

    def test_priority_statuto_over_regolamento(self):
        assert MetadataExtractor.infer_category("statuto_regolamento.pdf") == "Statuto"
