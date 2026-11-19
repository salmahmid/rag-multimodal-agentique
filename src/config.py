"""Chemins, modèle CLIP et configuration du LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PDF_PATH = Path(os.getenv("PDF_PATH", ROOT / "pdfs" / "attention is all you need.pdf")).expanduser()
if not PDF_PATH.is_absolute():
    PDF_PATH = (ROOT / PDF_PATH).resolve()

OUTPUT_DIR = ROOT / "resultats_pdf"
TEXT_DIR = OUTPUT_DIR / "textes"
IMAGE_DIR = OUTPUT_DIR / "images"
FULL_TEXT_PATH = OUTPUT_DIR / "texte_complet.txt"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CHROMA_DIR = ROOT / "chroma_pdf"
INDEX_MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"
UPLOAD_DIR = ROOT / "uploads"

COLLECTION_NAME = "pdf_clip_vit_b32"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10
CLIP_CONTEXT_TOKENS = 77
BATCH_SIZE = 32

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LLM_MODEL = os.getenv("LLM_MODEL", OLLAMA_MODEL)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_SUPPORTS_VISION = os.getenv("LLM_SUPPORTS_VISION", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "oui",
}


def ensure_runtime_dirs() -> None:
    """Crée les dossiers de travail locaux, sans toucher au PDF source."""
    for directory in (TEXT_DIR, IMAGE_DIR, CHROMA_DIR, UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)
