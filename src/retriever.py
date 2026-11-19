"""Recherche multimodale : texte ou image vers texte ou image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.embeddings import encoder_image, encoder_textes
from src.vector_store import obtenir_collection


@dataclass
class ResultatRecherche:
    identifiant: str
    type: str
    pdf: str
    page: int
    contenu: str
    source: str
    image_path: str
    origine: str
    distance: float
    similarite: float

    def to_dict(self) -> dict:
        return asdict(self)


def rechercher(
    texte: str | None = None,
    image: Path | str | None = None,
    type_resultat: str | None = None,
    k: int = 5,
) -> list[ResultatRecherche]:
    """Recherche par similarité cosinus.

    Modes :
    - texte seul, type texte : texte → texte
    - texte seul, type image : texte → image
    - image seule, type texte : image → texte
    - image seule, type image : image → image
    - type None : les deux types de contenu
    """
    if (texte is None) == (image is None):
        raise ValueError("Fournissez soit un texte, soit une image.")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être 'texte', 'image' ou None.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k doit être un entier supérieur ou égal à 1.")
    if texte is not None and not str(texte).strip():
        raise ValueError("Le texte de la requête est vide.")

    collection = obtenir_collection()
    where = {"type": type_resultat} if type_resultat else None
    disponibles = _compter(collection, where)
    if disponibles == 0:
        if collection.count() == 0:
            raise ValueError("La collection est vide. Indexez le PDF avant de rechercher.")
        return []

    if texte is not None:
        vecteur = encoder_textes([texte])[0]
    else:
        chemin = Path(image)
        if not chemin.is_file():
            raise FileNotFoundError(f"Image introuvable : {chemin}")
        vecteur = encoder_image(chemin)

    brut = collection.query(
        query_embeddings=[vecteur],
        n_results=min(k, disponibles),
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )
    return _parser(brut)


def mode_recherche(texte: str | None, image, type_resultat: str | None) -> str:
    source = "image" if image is not None else "texte"
    cible = type_resultat or "mixte"
    return f"{source} → {cible}"


def _compter(collection, where: dict | None) -> int:
    if where is None:
        return int(collection.count())
    trouves = collection.get(where=where, include=[])
    return len(trouves.get("ids") or [])


def _parser(brut: dict) -> list[ResultatRecherche]:
    documents = (brut.get("documents") or [[]])[0] or []
    metadatas = (brut.get("metadatas") or [[]])[0] or []
    distances = (brut.get("distances") or [[]])[0] or []
    identifiants = (brut.get("ids") or [[]])[0] or []

    resultats = []
    for identifiant, document, metadata, distance in zip(identifiants, documents, metadatas, distances):
        metadata = metadata or {}
        distance_valeur = float(distance)
        resultats.append(
            ResultatRecherche(
                identifiant=identifiant,
                type=str(metadata.get("type", "")),
                pdf=str(metadata.get("pdf", "")),
                page=int(metadata.get("page", -1)),
                contenu=document or "",
                source=str(metadata.get("source", "")),
                image_path=str(metadata.get("image_path") or ""),
                origine=str(metadata.get("origine", "")),
                distance=distance_valeur,
                similarite=1.0 - distance_valeur,
            )
        )
    return resultats
