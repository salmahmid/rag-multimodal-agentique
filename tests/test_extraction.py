from pathlib import Path

import pymupdf

from src.config import FULL_TEXT_PATH, IMAGE_DIR, PDF_PATH, TEXT_DIR
from src.pdf_extractor import extraire_pdf


def test_extraction_texte_et_images():
    assert PDF_PATH.is_file(), PDF_PATH
    with pymupdf.open(PDF_PATH) as document:
        pages_attendues = len(document)

    rapport = extraire_pdf(PDF_PATH)

    assert rapport.pages == pages_attendues
    assert rapport.pages > 0
    assert rapport.text_files == pages_attendues
    assert len(list(TEXT_DIR.glob("page_*.txt"))) == pages_attendues
    assert FULL_TEXT_PATH.is_file()

    complet = FULL_TEXT_PATH.read_text(encoding="utf-8")
    assert "Attention" in complet
    assert "Transformer" in complet
    premiere = (TEXT_DIR / "page_001.txt").read_text(encoding="utf-8")
    assert premiere.strip()

    images = [chemin for chemin in IMAGE_DIR.iterdir() if chemin.is_file()]
    assert rapport.images == len(images)
    assert rapport.images == rapport.embedded_images + rapport.vector_figures
    assert rapport.embedded_images + rapport.vector_figures >= 1
    for fichier in rapport.files:
        assert fichier["page"] >= 1
        assert Path(fichier["chemin"]).is_file()
        if fichier["type"] == "image":
            assert fichier["provenance"]
