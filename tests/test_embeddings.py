from pathlib import Path

import pytest

from src.config import IMAGE_DIR, PDF_PATH
from src.config import EMBEDDING_DIM
from src.embeddings import decouper_texte, encoder_image, encoder_textes, get_bpe, norme_l2
from src.pdf_extractor import extraire_pdf


def test_decoupage_60_tokens_chevauchement_10():
    with pytest.raises(ValueError):
        list(decouper_texte("attention", taille=60, chevauchement=60))

    assert list(decouper_texte("   ")) == []

    texte = "scaled dot product attention " * 80
    fragments = list(decouper_texte(texte, taille=60, chevauchement=10))
    assert len(fragments) >= 2
    bpe = get_bpe()
    longueurs = [len(bpe.encode(fragment)) for fragment in fragments]
    assert max(longueurs) <= 60

    tokens = bpe.encode(texte)
    import math

    attendu = 1 if len(tokens) <= 60 else math.ceil((len(tokens) - 60) / 50) + 1
    assert len(fragments) == attendu


def test_dimensions_et_normalisation(index_pret):
    vecteurs = encoder_textes(["The Transformer uses multi-head attention."])
    assert len(vecteurs) == 1
    assert len(vecteurs[0]) == EMBEDDING_DIM
    assert abs(norme_l2(vecteurs[0]) - 1.0) < 1e-3

    if not any(IMAGE_DIR.glob("*")):
        extraire_pdf(PDF_PATH)
    images = sorted(chemin for chemin in IMAGE_DIR.iterdir() if chemin.suffix.lower() == ".png")
    assert images, "Aucune image PNG extraite pour tester l'encodeur visuel."
    vecteur_image = encoder_image(images[0])
    assert len(vecteur_image) == EMBEDDING_DIM
    assert abs(norme_l2(vecteur_image) - 1.0) < 1e-3
    assert Path(images[0]).is_file()
