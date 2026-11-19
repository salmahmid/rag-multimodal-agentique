"""Extraction du texte, des images intégrées et des figures vectorielles."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from src.config import FULL_TEXT_PATH, IMAGE_DIR, MANIFEST_PATH, OUTPUT_DIR, TEXT_DIR

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jp2"}


@dataclass
class ExtractionReport:
    pdf: str
    pages: int
    text_files: int
    images: int
    embedded_images: int
    vector_figures: int
    pages_without_text: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def extraire_pdf(pdf_path: Path | str) -> ExtractionReport:
    """Extrait le texte et les images d'un PDF et les enregistre sur disque."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {pdf_path.resolve()}")

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    _vider_dossier(TEXT_DIR)
    _vider_dossier(IMAGE_DIR)

    textes: list[str] = []
    pages_sans_texte: list[int] = []
    erreurs: list[str] = []
    fichiers: list[dict] = []
    nombre_integrees = 0
    nombre_vectorielles = 0

    with pymupdf.open(pdf_path) as document:
        nombre_pages = len(document)

        for numero_page, page in enumerate(document, start=1):
            texte = page.get_text("text", sort=True) or ""
            fichier_texte = TEXT_DIR / f"page_{numero_page:03d}.txt"
            fichier_texte.write_text(texte, encoding="utf-8")
            textes.append(f"===== PAGE {numero_page} =====\n\n{texte}")
            fichiers.append(
                {
                    "type": "texte",
                    "page": numero_page,
                    "chemin": str(fichier_texte.resolve()),
                    "provenance": "page.get_text",
                }
            )
            if not texte.strip():
                pages_sans_texte.append(numero_page)

            nombre_integrees += _extraire_images_integrees(
                document, page, numero_page, erreurs, fichiers
            )
            nombre_vectorielles += _rendre_figures_vectorielles(
                page, numero_page, erreurs, fichiers
            )

    FULL_TEXT_PATH.write_text("\n\n".join(textes), encoding="utf-8")
    rapport = ExtractionReport(
        pdf=pdf_path.name,
        pages=nombre_pages,
        text_files=nombre_pages,
        images=nombre_integrees + nombre_vectorielles,
        embedded_images=nombre_integrees,
        vector_figures=nombre_vectorielles,
        pages_without_text=pages_sans_texte,
        errors=erreurs,
        files=fichiers,
    )
    MANIFEST_PATH.write_text(
        json.dumps(rapport.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Extraction terminée : %s pages, %s images (%s intégrées, %s vectorielles).",
        rapport.pages,
        rapport.images,
        rapport.embedded_images,
        rapport.vector_figures,
    )
    return rapport


def _vider_dossier(dossier: Path) -> None:
    for chemin in dossier.iterdir():
        if chemin.is_file():
            chemin.unlink()


def _extraire_images_integrees(
    document: pymupdf.Document,
    page: pymupdf.Page,
    numero_page: int,
    erreurs: list[str],
    fichiers: list[dict],
) -> int:
    nombre = 0
    images_vues: set[int] = set()

    for image_info in page.get_images(full=True):
        xref = int(image_info[0])
        if xref in images_vues:
            continue
        images_vues.add(xref)

        try:
            image = document.extract_image(xref)
            if not image or not image.get("image"):
                message = f"Image non extractible : page {numero_page}, xref {xref}"
                erreurs.append(message)
                logger.warning(message)
                continue

            extension = str(image.get("ext") or "png").lower().lstrip(".")
            fichier_image = IMAGE_DIR / f"page_{numero_page:03d}_image_{xref}.{extension}"
            fichier_image.write_bytes(image["image"])
            fichiers.append(
                {
                    "type": "image",
                    "kind": "integree",
                    "page": numero_page,
                    "xref": xref,
                    "chemin": str(fichier_image.resolve()),
                    "provenance": "image intégrée au PDF",
                }
            )
            nombre += 1
        except Exception as erreur:
            message = f"Erreur d'extraction : page {numero_page}, image {xref} : {erreur}"
            erreurs.append(message)
            logger.warning(message)

    return nombre


def _rendre_figures_vectorielles(
    page: pymupdf.Page,
    numero_page: int,
    erreurs: list[str],
    fichiers: list[dict],
) -> int:
    """Rend les groupes de tracés vectoriels assez grands pour être des figures.

    Les schémas de l'article (architecture, attention) sont des tracés, pas des
    images intégrées. Les petits glyphes d'équations sont écartés par la taille
    du regroupement.
    """
    try:
        drawings = page.get_drawings()
    except Exception as erreur:
        message = f"Tracés vectoriels illisibles : page {numero_page} : {erreur}"
        erreurs.append(message)
        logger.warning(message)
        return 0

    if not drawings:
        return 0

    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        rect = _rectangle_trace(drawing)
        if rect is None:
            continue
        rects.append(rect)

    if not rects:
        return 0

    clusters = _regrouper_rectangles(rects, marge=14)
    page_rect = page.rect
    nombre = 0

    for cluster in clusters:
        clip = (cluster & page_rect)
        if clip.is_empty:
            continue
        # Une figure occupe un bloc visible. Une ligne d'équation reste trop plate.
        if clip.width < 150 or clip.height < 70:
            continue
        if clip.width * clip.height < 18_000:
            continue
        # Écarter un fond qui couvre presque toute la page.
        if clip.width > 0.96 * page_rect.width and clip.height > 0.96 * page_rect.height:
            continue

        clip = pymupdf.Rect(
            max(page_rect.x0, clip.x0 - 4),
            max(page_rect.y0, clip.y0 - 4),
            min(page_rect.x1, clip.x1 + 4),
            min(page_rect.y1, clip.y1 + 4),
        )
        nombre += 1
        fichier_image = IMAGE_DIR / f"page_{numero_page:03d}_vector_{nombre:02d}.png"
        try:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip, alpha=False)
            pixmap.save(fichier_image)
            fichiers.append(
                {
                    "type": "image",
                    "kind": "vectorielle",
                    "page": numero_page,
                    "xref": None,
                    "chemin": str(fichier_image.resolve()),
                    "provenance": "rendu d'un groupe de tracés vectoriels",
                    "bbox": [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                }
            )
        except Exception as erreur:
            nombre -= 1
            message = f"Rendu vectoriel impossible : page {numero_page} : {erreur}"
            erreurs.append(message)
            logger.warning(message)

    return nombre


def _rectangle_trace(drawing: dict) -> pymupdf.Rect | None:
    """Élargit les filets de largeur nulle pour pouvoir les regrouper."""
    rect = pymupdf.Rect(drawing.get("rect"))
    if rect.is_infinite:
        return None
    epaisseur = float(drawing.get("width") or 0.8)
    marge = max(epaisseur, 0.8)
    if rect.height < 1:
        rect.y0 -= marge
        rect.y1 += marge
    if rect.width < 1:
        rect.x0 -= marge
        rect.x1 += marge
    if rect.is_empty or rect.width < 1.5 or rect.height < 1.5:
        return None
    return rect


def _regrouper_rectangles(rects: list[pymupdf.Rect], marge: float) -> list[pymupdf.Rect]:
    clusters: list[pymupdf.Rect] = []
    for rect in sorted(rects, key=lambda item: (item.y0, item.x0)):
        place = False
        for index, cluster in enumerate(clusters):
            if _proches(cluster, rect, marge):
                clusters[index] = cluster | rect
                place = True
                break
        if not place:
            clusters.append(pymupdf.Rect(rect))

    stable = False
    while not stable:
        stable = True
        fusionnes: list[pymupdf.Rect] = []
        for rect in clusters:
            place = False
            for index, cluster in enumerate(fusionnes):
                if _proches(cluster, rect, marge):
                    fusionnes[index] = cluster | rect
                    place = True
                    stable = False
                    break
            if not place:
                fusionnes.append(rect)
        clusters = fusionnes
    return clusters


def _proches(gauche: pymupdf.Rect, droite: pymupdf.Rect, marge: float) -> bool:
    elargi = pymupdf.Rect(
        gauche.x0 - marge,
        gauche.y0 - marge,
        gauche.x1 + marge,
        gauche.y1 + marge,
    )
    return bool(elargi.intersects(droite))


def lister_images() -> list[Path]:
    if not IMAGE_DIR.is_dir():
        return []
    return sorted(
        chemin
        for chemin in IMAGE_DIR.iterdir()
        if chemin.is_file() and chemin.suffix.lower() in IMAGE_EXTENSIONS
    )
