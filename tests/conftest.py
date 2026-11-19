"""Index réel partagé par les tests qui interrogent ChromaDB."""

from __future__ import annotations

import pytest

from src.config import PDF_PATH
from src.vector_store import etat_index, identifiant_document, indexer_pdf


@pytest.fixture(scope="session")
def index_pret():
    if not PDF_PATH.is_file():
        pytest.skip(f"PDF absent : {PDF_PATH}")
    attendu = identifiant_document(PDF_PATH)
    etat = etat_index()
    if not (etat.get("pret") and etat.get("document_id") == attendu and etat.get("total", 0) > 0):
        indexer_pdf(PDF_PATH)
        etat = etat_index()
    if not etat.get("pret"):
        pytest.fail("L'indexation n'a produit aucun élément.")
    return etat
