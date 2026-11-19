"""Indexation persistante des embeddings CLIP dans ChromaDB."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import chromadb

from src.config import (
    BATCH_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    IMAGE_DIR,
    INDEX_MANIFEST_PATH,
    TEXT_DIR,
)
from src.pdf_extractor import IMAGE_EXTENSIONS, extraire_pdf

logger = logging.getLogger(__name__)


def identifiant_document(pdf_path: Path | str) -> str:
    """Empreinte SHA-256 du contenu du PDF."""
    pdf_path = Path(pdf_path)
    digest = hashlib.sha256()
    with pdf_path.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            digest.update(bloc)
    return digest.hexdigest()


def numero_page(chemin: Path | str) -> int:
    correspondance = re.search(r"page_(\d+)", Path(chemin).stem)
    return int(correspondance.group(1)) if correspondance else -1


def obtenir_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def indexer_pdf(pdf_path: Path | str, reextraire: bool = True) -> dict:
    """Extrait si besoin, puis indexe textes et images sans laisser de doublons."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path.resolve()}")

    rapport = extraire_pdf(pdf_path) if reextraire or not any(TEXT_DIR.glob("page_*.txt")) else None
    document_id = identifiant_document(pdf_path)
    collection = obtenir_collection()
    _retirer_versions_obsoletes(collection, pdf_path.name, document_id)
    _supprimer_document(collection, document_id)

    fragments = _indexer_textes(collection, pdf_path, document_id)
    images = _indexer_images(collection, pdf_path, document_id)
    _enregistrer_manifeste(pdf_path.name, document_id)

    stats = {
        "pdf": pdf_path.name,
        "document_id": document_id,
        "pages": rapport.pages if rapport else len(list(TEXT_DIR.glob("page_*.txt"))),
        "fichiers_texte": len(list(TEXT_DIR.glob("page_*.txt"))),
        "fichiers_images": len(_images_indexables()),
        "fragments_texte": fragments,
        "images_indexees": images,
        "total": collection.count(),
        "extraction": rapport.to_dict() if rapport else None,
    }
    logger.info("Indexation terminée : %s", stats)
    return stats


def etat_index() -> dict:
    """Résumé de la base, sans charger CLIP."""
    if not CHROMA_DIR.exists():
        return {"pret": False, "total": 0, "document_id": None, "pdf": None}
    try:
        collection = obtenir_collection()
        total = collection.count()
    except Exception as erreur:
        return {"pret": False, "total": 0, "erreur": str(erreur), "document_id": None, "pdf": None}

    manifeste = _lire_manifeste()
    pdf = next(iter(manifeste), None)
    return {
        "pret": total > 0,
        "total": total,
        "document_id": manifeste.get(pdf) if pdf else None,
        "pdf": pdf,
    }


def _indexer_textes(collection, pdf_path: Path, document_id: str) -> int:
    from src.embeddings import decouper_texte, encoder_textes

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for fichier in sorted(TEXT_DIR.glob("page_*.txt")):
        texte = fichier.read_text(encoding="utf-8")
        page = numero_page(fichier)
        for index, fragment in enumerate(decouper_texte(texte)):
            ids.append(f"{document_id}:texte:{page}:{index}")
            documents.append(fragment)
            metadatas.append(
                {
                    "type": "texte",
                    "document_id": document_id,
                    "pdf": pdf_path.name,
                    "page": page,
                    "fragment": index,
                    "source": str(fichier.resolve()),
                    "image_path": "",
                    "origine": "texte",
                }
            )

    for debut in range(0, len(documents), BATCH_SIZE):
        fin = debut + BATCH_SIZE
        lot = documents[debut:fin]
        collection.upsert(
            ids=ids[debut:fin],
            documents=lot,
            metadatas=metadatas[debut:fin],
            embeddings=encoder_textes(lot),
        )
    return len(documents)


def _indexer_images(collection, pdf_path: Path, document_id: str) -> int:
    from src.embeddings import encoder_images

    fichiers = _images_indexables()
    nombre = 0

    for debut in range(0, len(fichiers), 8):
        lot = fichiers[debut : debut + 8]
        try:
            vecteurs = encoder_images(lot)
        except Exception as erreur:
            logger.warning("Lot d'images ignoré (%s) : %s", [item.name for item in lot], erreur)
            for fichier in lot:
                try:
                    vecteurs_un = encoder_images([fichier])
                except Exception as erreur_image:
                    logger.warning("Image ignorée : %s — %s", fichier.name, erreur_image)
                    continue
                _upsert_image(collection, pdf_path, document_id, fichier, vecteurs_un[0])
                nombre += 1
            continue

        for fichier, vecteur in zip(lot, vecteurs):
            if len(vecteur) != EMBEDDING_DIM:
                logger.warning("Dimension inattendue pour %s", fichier.name)
                continue
            _upsert_image(collection, pdf_path, document_id, fichier, vecteur)
            nombre += 1

    return nombre


def _upsert_image(collection, pdf_path: Path, document_id: str, fichier: Path, vecteur: list[float]) -> None:
    origine = "vectorielle" if "_vector_" in fichier.name else "integree"
    collection.upsert(
        ids=[f"{document_id}:image:{fichier.name}"],
        embeddings=[vecteur],
        documents=[f"Image extraite du PDF : {fichier.name}"],
        metadatas=[
            {
                "type": "image",
                "document_id": document_id,
                "pdf": pdf_path.name,
                "page": numero_page(fichier),
                "fragment": -1,
                "source": str(fichier.resolve()),
                "image_path": str(fichier.resolve()),
                "origine": origine,
            }
        ],
    )


def _images_indexables() -> list[Path]:
    if not IMAGE_DIR.is_dir():
        return []
    fichiers = []
    for fichier in sorted(IMAGE_DIR.iterdir()):
        if not fichier.is_file() or fichier.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            from PIL import Image

            with Image.open(fichier) as image:
                largeur, hauteur = image.size
            if largeur < 16 or hauteur < 16:
                logger.info("Image trop petite, non indexée : %s", fichier.name)
                continue
        except Exception as erreur:
            logger.warning("Image illisible, non indexée : %s — %s", fichier.name, erreur)
            continue
        fichiers.append(fichier)
    return fichiers


def _retirer_versions_obsoletes(collection, nom_pdf: str, document_id: str) -> None:
    """Si le fichier a changé, retire l'empreinte précédente du même PDF."""
    precedente = _lire_manifeste().get(nom_pdf)
    if precedente and precedente != document_id:
        logger.info("Le PDF a changé : suppression de l'index %s", precedente)
        _supprimer_document(collection, precedente)

    try:
        anciens = collection.get(where={"pdf": nom_pdf}, include=["metadatas"])
    except Exception:
        return
    ids_a_retirer = []
    for identifiant, metadata in zip(anciens.get("ids") or [], anciens.get("metadatas") or []):
        if metadata and metadata.get("document_id") not in (None, document_id):
            ids_a_retirer.append(identifiant)
    if ids_a_retirer:
        collection.delete(ids=ids_a_retirer)


def _supprimer_document(collection, document_id: str) -> None:
    try:
        existants = collection.get(where={"document_id": document_id}, include=[])
    except Exception:
        return
    identifiants = existants.get("ids") or []
    if identifiants:
        collection.delete(ids=identifiants)


def _lire_manifeste() -> dict:
    if not INDEX_MANIFEST_PATH.is_file():
        return {}
    try:
        donnees = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return donnees if isinstance(donnees, dict) else {}


def _enregistrer_manifeste(nom_pdf: str, document_id: str) -> None:
    manifeste = _lire_manifeste()
    manifeste[nom_pdf] = document_id
    INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST_PATH.write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
