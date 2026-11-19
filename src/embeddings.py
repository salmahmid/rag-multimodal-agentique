"""Découpage BPE et encodage CLIP texte / image dans un espace commun."""

from __future__ import annotations

import threading
from pathlib import Path

import open_clip
import torch
from open_clip.tokenizer import SimpleTokenizer
from PIL import Image

from src.config import (
    CLIP_CONTEXT_TOKENS,
    EMBEDDING_DIM,
    MODEL_NAME,
    PRETRAINED,
)

_LOCK = threading.Lock()
_BPE: SimpleTokenizer | None = None
_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = "cpu"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_bpe() -> SimpleTokenizer:
    """Tokeniseur BPE de CLIP, sans charger les poids du modèle."""
    global _BPE
    if _BPE is None:
        _BPE = SimpleTokenizer()
    return _BPE


def decouper_texte(texte: str, taille: int = 60, chevauchement: int = 10):
    """Découpe un texte en fragments d'au plus ``taille`` tokens, avec chevauchement."""
    if not 0 <= chevauchement < taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille.")

    bpe = get_bpe()
    tokens = bpe.encode(texte or "")
    if not tokens:
        return

    pas = taille - chevauchement
    for debut in range(0, len(tokens), pas):
        fragment = bpe.decode(tokens[debut : debut + taille]).strip()
        garde = 0
        while fragment and len(bpe.encode(fragment)) > taille and garde < taille * 4:
            fragment = fragment[:-1].strip() if " " not in fragment else fragment.rsplit(" ", 1)[0].strip()
            garde += 1
        if fragment:
            yield fragment
        if debut + taille >= len(tokens):
            break


def charger_clip():
    """Charge ViT-B-32 / laion2b_s34b_b79k une seule fois."""
    global _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE
    with _LOCK:
        if _MODEL is None:
            _DEVICE = get_device()
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME,
                pretrained=PRETRAINED,
            )
            _MODEL = model.to(_DEVICE).eval()
            _PREPROCESS = preprocess
            _TOKENIZER = open_clip.get_tokenizer(MODEL_NAME)
        return _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE


def _normaliser(vecteurs: torch.Tensor) -> torch.Tensor:
    return vecteurs / vecteurs.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encoder_textes(textes: list[str]) -> list[list[float]]:
    """Encode un lot de textes et renvoie des vecteurs L2 de dimension 512."""
    if not textes:
        return []

    model, _, tokenizer, device = charger_clip()
    # Le tokenizer CLIP fixe la longueur à 77, tokens spéciaux inclus.
    # Les fragments sont déjà limités à 60 tokens BPE de contenu.
    tokens = tokenizer(textes).to(device)
    if tokens.shape[-1] > CLIP_CONTEXT_TOKENS:
        tokens = tokens[:, :CLIP_CONTEXT_TOKENS]

    with torch.inference_mode():
        vecteurs = _normaliser(model.encode_text(tokens).float())

    _verifier_dimensions(vecteurs)
    return vecteurs.cpu().tolist()


def encoder_image(chemin: Path | str) -> list[float]:
    """Encode une image et renvoie son vecteur L2."""
    vecteurs = encoder_images([chemin])
    return vecteurs[0]


def encoder_images(chemins: list[Path | str]) -> list[list[float]]:
    if not chemins:
        return []

    model, preprocess, _, device = charger_clip()
    tenseurs = []
    for chemin in chemins:
        with Image.open(chemin) as image:
            tenseurs.append(preprocess(image.convert("RGB")))

    lot = torch.stack(tenseurs).to(device)
    with torch.inference_mode():
        vecteurs = _normaliser(model.encode_image(lot).float())

    _verifier_dimensions(vecteurs)
    return vecteurs.cpu().tolist()


def _verifier_dimensions(vecteurs: torch.Tensor) -> None:
    if vecteurs.ndim != 2 or vecteurs.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Dimension d'embedding inattendue : {tuple(vecteurs.shape)}, "
            f"attendu (*, {EMBEDDING_DIM})."
        )


def norme_l2(vecteur: list[float]) -> float:
    return float(sum(valeur * valeur for valeur in vecteur) ** 0.5)
