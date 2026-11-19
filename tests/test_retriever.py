from pathlib import Path

from src.config import IMAGE_DIR, PDF_PATH
from src.retriever import rechercher
from src.vector_store import etat_index, identifiant_document, indexer_pdf, obtenir_collection


def test_identifiant_sha256():
    empreinte = identifiant_document(PDF_PATH)
    assert len(empreinte) == 64
    assert empreinte == identifiant_document(PDF_PATH)


def test_indexation_sans_doublons(index_pret):
    avant = obtenir_collection().count()
    assert avant == index_pret["total"]
    assert avant > 0
    stats = indexer_pdf(PDF_PATH, reextraire=False)
    apres = obtenir_collection().count()
    assert apres == avant
    assert stats["total"] == apres
    assert etat_index()["document_id"] == identifiant_document(PDF_PATH)


def test_mise_a_jour_si_document_change(index_pret):
    collection = obtenir_collection()
    collection.upsert(
        ids=["obsolete-test"],
        embeddings=[[0.0] * 511 + [1.0]],
        documents=["fragment obsolete"],
        metadatas=[
            {
                "type": "texte",
                "document_id": "empreinte-ancienne",
                "pdf": PDF_PATH.name,
                "page": 1,
                "fragment": 0,
                "source": "",
                "image_path": "",
                "origine": "texte",
            }
        ],
    )
    manifeste = index_pret
    from src.config import INDEX_MANIFEST_PATH
    import json

    INDEX_MANIFEST_PATH.write_text(
        json.dumps({PDF_PATH.name: "empreinte-ancienne"}),
        encoding="utf-8",
    )
    indexer_pdf(PDF_PATH, reextraire=False)
    restants = collection.get(ids=["obsolete-test"])
    assert restants["ids"] == []
    assert etat_index()["document_id"] == identifiant_document(PDF_PATH)
    assert manifeste["document_id"]


def test_quatre_modes_de_recherche(index_pret):
    texte_texte = rechercher(
        texte="What is the Transformer architecture?",
        type_resultat="texte",
        k=4,
    )
    assert texte_texte
    assert all(resultat.type == "texte" for resultat in texte_texte)
    assert texte_texte[0].page >= 1
    assert texte_texte[0].similarite <= 1.0

    texte_image = rechercher(
        texte="Transformer model architecture figure",
        type_resultat="image",
        k=3,
    )
    assert texte_image
    assert all(resultat.type == "image" for resultat in texte_image)
    assert any(resultat.image_path for resultat in texte_image)

    image = next(chemin for chemin in sorted(IMAGE_DIR.iterdir()) if chemin.suffix.lower() in {".png", ".jpg", ".jpeg"})
    image_texte = rechercher(image=image, type_resultat="texte", k=4)
    assert image_texte
    assert all(resultat.type == "texte" for resultat in image_texte)
    assert image_texte[0].contenu.strip()

    image_image = rechercher(image=image, type_resultat="image", k=3)
    assert image_image
    assert image_image[0].type == "image"
    assert image_image[0].similarite > 0.95
    chemins = {str(Path(resultat.image_path).resolve()) for resultat in image_image if resultat.image_path}
    assert str(image.resolve()) in chemins


def test_questions_documentaires(index_pret):
    questions = [
        "What is the Transformer architecture?",
        "Explain the multi-head attention mechanism.",
        "What is scaled dot-product attention?",
        "What is the role of positional encoding?",
        "Compare the encoder and decoder.",
        "Explain the formula of scaled dot-product attention.",
    ]
    for question in questions:
        resultats = rechercher(texte=question, type_resultat="texte", k=3)
        assert resultats, question
        assert resultats[0].contenu.strip()
        assert isinstance(resultats[0].page, int)

    for question in ("Find the Transformer architecture figure.", "Find figures related to attention."):
        resultats = rechercher(texte=question, type_resultat="image", k=3)
        assert resultats, question
        assert resultats[0].type == "image"
        assert resultats[0].page != 0
