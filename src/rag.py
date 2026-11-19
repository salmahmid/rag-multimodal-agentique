"""RAG génératif : recherche multimodale, puis réponse ancrée dans les sources."""

from __future__ import annotations

from pathlib import Path

from src.llm import etat_llm, generer, vision_active
from src.retriever import ResultatRecherche, mode_recherche, rechercher


def repondre(
    question: str,
    k: int = 6,
    type_resultat: str | None = None,
    image_requete: Path | str | None = None,
    max_images: int = 3,
) -> dict:
    """Recherche les passages et figures, puis demande une réponse en français au LLM."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question doit être un texte non vide.")
    if not isinstance(max_images, int) or max_images < 0:
        raise ValueError("max_images doit être un entier positif ou nul.")

    resultats = rechercher(
        texte=None if image_requete else question,
        image=image_requete,
        type_resultat=type_resultat,
        k=k,
    )
    sources = _numeroter(resultats)
    contexte, images, avertissements, transmises = _construire_contexte(sources, max_images)
    mode = mode_recherche(None if image_requete else question, image_requete, type_resultat)

    if not transmises:
        return {
            "question": question,
            "reponse": "Aucune source exploitable n'a été retrouvée pour répondre à cette question.",
            "sources": sources,
            "sources_transmises": [],
            "avertissements": avertissements,
            "mode": mode,
            "llm": etat_llm(),
            "vision_utilisee": False,
        }

    etat = etat_llm()
    if not etat.get("ok") or etat.get("modele_present") is False:
        return {
            "question": question,
            "reponse": "",
            "erreur": (
                "Les sources ont été retrouvées, mais aucun texte n'a été généré : "
                f"{etat.get('detail')}"
            ),
            "sources": sources,
            "sources_transmises": transmises,
            "avertissements": avertissements,
            "mode": mode,
            "llm": etat,
            "vision_utilisee": False,
            "contexte": contexte,
        }

    try:
        reponse = generer(question, contexte, images if vision_active() else None)
    except Exception as erreur:
        return {
            "question": question,
            "reponse": "",
            "erreur": f"La génération a échoué : {erreur}",
            "sources": sources,
            "sources_transmises": transmises,
            "avertissements": avertissements,
            "mode": mode,
            "llm": etat,
            "vision_utilisee": False,
            "contexte": contexte,
        }

    if images and not vision_active():
        avertissements.append(
            "Le LLM configuré ne reçoit pas les pixels des figures. "
            "Les images retrouvées sont affichées à part et n'ont pas été analysées visuellement."
        )
        reponse += _note_figures(images, transmises)

    return {
        "question": question,
        "reponse": reponse,
        "sources": sources,
        "sources_transmises": transmises,
        "avertissements": avertissements,
        "mode": mode,
        "llm": etat,
        "vision_utilisee": bool(images) and vision_active(),
    }


def _numeroter(resultats: list[ResultatRecherche]) -> list[dict]:
    sources = []
    for index, resultat in enumerate(resultats, start=1):
        source = resultat.to_dict()
        source["reference"] = f"S{index}"
        sources.append(source)
    return sources


def _construire_contexte(sources: list[dict], max_images: int):
    blocs = []
    images: list[Path] = []
    avertissements: list[str] = []
    transmises = []
    nombre_images = 0

    for source in sources:
        page = source["page"] if source["page"] != -1 else "inconnue"
        if source["type"] == "texte":
            extrait = (source.get("contenu") or "").strip()
            if not extrait:
                continue
            blocs.append(
                "\n".join(
                    [
                        f"[{source['reference']}] type=texte | {source['pdf']} | page {page} | similarité {source['similarite']:.3f}",
                        extrait,
                    ]
                )
            )
            transmises.append(source)
        elif source["type"] == "image":
            if nombre_images >= max_images:
                avertissements.append(
                    f"{source['reference']} : image non transmise (limite de {max_images} images)."
                )
                continue
            chemin = Path(source.get("image_path") or source.get("source") or "")
            if not chemin.is_file():
                avertissements.append(f"{source['reference']} : fichier image introuvable ({chemin}).")
                continue
            blocs.append(
                "\n".join(
                    [
                        f"[{source['reference']}] type=image | {source['pdf']} | page {page} | similarité {source['similarite']:.3f}",
                        f"Fichier : {chemin.name}",
                        source.get("contenu") or "",
                        "Les pixels ne font pas partie de ce texte de contexte.",
                    ]
                )
            )
            images.append(chemin)
            transmises.append(source)
            nombre_images += 1

    return "\n\n".join(blocs), images, avertissements, transmises


def _note_figures(images: list[Path], sources: list[dict]) -> str:
    lignes = [
        "",
        "Figures récupérées par la recherche, sans analyse visuelle du modèle :",
    ]
    images_sources = [source for source in sources if source["type"] == "image"]
    for source, chemin in zip(images_sources, images):
        page = source["page"] if source["page"] != -1 else "inconnue"
        lignes.append(
            f"- [{source['reference']}] {source['pdf']}, page {page}, similarité {source['similarite']:.3f} — {chemin.name}"
        )
    return "\n".join(lignes)

