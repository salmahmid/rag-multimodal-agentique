"""Extrait le PDF puis construit l'index ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PDF_PATH
from src.vector_store import indexer_pdf


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Indexer un PDF pour le RAG multimodal.")
    parser.add_argument("pdf", nargs="?", default=str(PDF_PATH), help="Chemin du PDF.")
    args = parser.parse_args()
    stats = indexer_pdf(args.pdf)
    extraction = stats.get("extraction") or {}
    print(f"Pages : {stats['pages']}")
    print(f"Fichiers texte : {stats['fichiers_texte']}")
    print(f"Images extraites : {extraction.get('images', stats['fichiers_images'])}")
    print(f"  intégrées : {extraction.get('embedded_images', 'n/d')}")
    print(f"  vectorielles : {extraction.get('vector_figures', 'n/d')}")
    print(f"Fragments de texte indexés : {stats['fragments_texte']}")
    print(f"Images indexées : {stats['images_indexees']}")
    print(f"Total ChromaDB : {stats['total']}")
    print(f"SHA-256 : {stats['document_id']}")
    if extraction.get("errors"):
        print("Erreurs d'extraction :")
        print(json.dumps(extraction["errors"], ensure_ascii=False, indent=2))
    if extraction.get("pages_without_text"):
        print("Pages sans texte :", extraction["pages_without_text"])


if __name__ == "__main__":
    main()
