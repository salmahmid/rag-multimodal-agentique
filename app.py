"""Interface Streamlit du RAG multimodal agentique."""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from src.agent import interroger
from src.config import LLM_MODEL, LLM_PROVIDER, PDF_PATH, UPLOAD_DIR, ensure_runtime_dirs
from src.llm import etat_llm, vision_active
from src.rag import repondre
from src.vector_store import etat_index, indexer_pdf

st.set_page_config(
    page_title="RAG Multimodal Agentique",
    page_icon="◆",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f7f9; }
    .bloc-titre { padding: 0.2rem 0 0.8rem 0; }
    .bloc-titre h1 { font-size: 1.8rem; margin-bottom: 0.2rem; color: #1c2833; }
    .bloc-titre p { color: #52616b; margin: 0; }
    .etat-ok { color: #0e7a3d; font-weight: 600; }
    .etat-ko { color: #a12622; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialiser() -> None:
    ensure_runtime_dirs()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex


def _pastille(ok: bool) -> str:
    return "etat-ok" if ok else "etat-ko"


def _barre_laterale() -> tuple[int, str]:
    st.sidebar.header("État du système")
    pdf_ok = PDF_PATH.is_file()
    st.sidebar.markdown("**PDF**")
    st.sidebar.markdown(
        f"<span class='{_pastille(pdf_ok)}'>{'Présent' if pdf_ok else 'Introuvable'}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(str(PDF_PATH.name if pdf_ok else PDF_PATH))

    index = etat_index()
    index_ok = bool(index.get("pret"))
    st.sidebar.markdown("**ChromaDB**")
    st.sidebar.markdown(
        f"<span class='{_pastille(index_ok)}'>{'Index prêt' if index_ok else 'Non indexé'}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"{index.get('total', 0)} éléments")
    if index.get("erreur"):
        st.sidebar.error(index["erreur"])

    llm = etat_llm()
    llm_ok = bool(llm.get("ok")) and llm.get("modele_present") is not False
    st.sidebar.markdown("**LLM**")
    st.sidebar.markdown(
        f"<span class='{_pastille(llm_ok)}'>{LLM_PROVIDER} · {LLM_MODEL}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(llm.get("detail", ""))
    st.sidebar.caption("Vision : " + ("activée" if vision_active() else "désactivée"))

    st.sidebar.divider()
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=12, value=5)
    mode = st.sidebar.radio("Mode", ["Agent ReAct", "RAG direct"], index=0)
    if st.sidebar.button("Lancer l'indexation", type="primary", width="stretch"):
        _indexer()
    if st.sidebar.button("Effacer la conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()
    return top_k, mode


def _indexer() -> None:
    if not PDF_PATH.is_file():
        st.sidebar.error(f"PDF introuvable : {PDF_PATH}")
        return
    with st.spinner("Extraction du PDF et calcul des embeddings CLIP. La première exécution télécharge le modèle."):
        try:
            stats = indexer_pdf(PDF_PATH)
        except Exception as erreur:
            st.sidebar.error(f"Indexation impossible : {erreur}")
            return
    extraction = stats.get("extraction") or {}
    st.sidebar.success(
        f"{stats['pages']} pages, {stats['fragments_texte']} fragments, "
        f"{stats['images_indexees']} images indexées "
        f"({extraction.get('embedded_images', 0)} intégrées, "
        f"{extraction.get('vector_figures', 0)} vectorielles)."
    )


def _enregistrer_image(televersement) -> Path | None:
    if televersement is None:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffixe = Path(televersement.name).suffix.lower() or ".png"
    if suffixe not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError("Format d'image non pris en charge. Utilisez PNG, JPG ou WEBP.")
    destination = UPLOAD_DIR / f"requete{suffixe}"
    destination.write_bytes(televersement.getvalue())
    return destination


def _poser(question: str, image: Path | None, top_k: int, mode: str) -> None:
    if not etat_index().get("pret"):
        st.error("L'index ChromaDB est vide. Lancez l'indexation depuis la barre latérale.")
        return
    try:
        if mode == "Agent ReAct":
            resultat = interroger(
                question,
                image_path=str(image) if image else None,
                top_k=top_k,
                thread_id=st.session_state.thread_id,
            )
        else:
            resultat = repondre(
                question,
                k=top_k,
                image_requete=str(image) if image else None,
            )
            resultat["appels"] = []
    except Exception as erreur:
        st.error(f"La requête a échoué : {erreur}")
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "contenu": question,
            "image": str(image) if image else "",
        }
    )
    st.session_state.messages.append({"role": "assistant", "resultat": resultat})


def _afficher_resultat(resultat: dict) -> None:
    if resultat.get("erreur"):
        st.error(resultat["erreur"])
    if resultat.get("reponse"):
        st.markdown(resultat["reponse"])
    elif not resultat.get("erreur"):
        st.warning("Le système n'a pas produit de réponse.")

    sources = resultat.get("sources_transmises") or resultat.get("sources") or []
    if sources:
        st.markdown("**Sources**")
        lignes = []
        for source in sources:
            page = source.get("page")
            page_texte = "inconnue" if page in (-1, None) else str(page)
            similarite = source.get("similarite")
            score = f"{float(similarite):.3f}" if isinstance(similarite, (int, float)) else "n/d"
            lignes.append(
                {
                    "Référence": source.get("reference", ""),
                    "Type": source.get("type", ""),
                    "Page": page_texte,
                    "Similarité": score,
                    "PDF": source.get("pdf", ""),
                }
            )
        st.dataframe(lignes, width="stretch", hide_index=True)

        images = []
        for source in sources:
            chemin = source.get("chemin_image") or source.get("image_path") or ""
            if source.get("type") == "image" and chemin and Path(chemin).is_file():
                images.append((source, Path(chemin)))
        if images:
            st.markdown("**Images pertinentes**")
            colonnes = st.columns(min(len(images), 3))
            for index, (source, chemin) in enumerate(images):
                page = source.get("page")
                page_texte = "inconnue" if page in (-1, None) else str(page)
                score = source.get("similarite")
                score_texte = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/d"
                with colonnes[index % len(colonnes)]:
                    st.image(
                        str(chemin),
                        caption=f"Page {page_texte} · similarité {score_texte}",
                        width="stretch",
                    )

    for avertissement in resultat.get("avertissements") or []:
        st.warning(avertissement)

    appels = resultat.get("appels") or []
    if appels:
        st.markdown("**Appels d'outils**")
        for numero, appel in enumerate(appels, start=1):
            with st.expander(f"{numero}. {appel.get('nom', 'outil')}"):
                st.markdown("**Arguments**")
                st.json(appel.get("arguments") or {})
                observation = appel.get("observation") or ""
                if observation:
                    st.markdown("**Observation**")
                    st.code(observation, language="json")


def main() -> None:
    _initialiser()
    top_k, mode = _barre_laterale()

    st.markdown(
        """
        <div class="bloc-titre">
            <h1>RAG Multimodal Agentique — Attention Is All You Need</h1>
            <p>Recherche croisée texte et image, réponse ancrée dans l'article, agent ReAct.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["contenu"])
                if message.get("image") and Path(message["image"]).is_file():
                    st.image(message["image"], caption="Image jointe", width=280)
            else:
                _afficher_resultat(message["resultat"])

    televersement = st.file_uploader(
        "Image de requête (optionnelle)",
        type=["png", "jpg", "jpeg", "webp"],
    )
    question = st.chat_input("Posez une question sur l'article")
    if question:
        try:
            image = _enregistrer_image(televersement)
        except Exception as erreur:
            st.error(str(erreur))
            return
        _poser(question, image, top_k, mode)
        st.rerun()


if __name__ == "__main__":
    main()
