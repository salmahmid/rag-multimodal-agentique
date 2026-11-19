import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.agent import construire_agent, interroger, reinitialiser_budget, retriever_tool
from src.config import IMAGE_DIR
from src.llm import etat_llm


def test_retriever_tool_texte(index_pret):
    reinitialiser_budget(top_k=3)
    brut = retriever_tool.invoke(
        {
            "query": "What is scaled dot-product attention?",
            "content_type": "texte",
            "top_k": 3,
            "image_path": "",
        }
    )
    donnees = json.loads(brut)
    assert donnees["mode"] == "texte → texte"
    assert donnees["resultats"]
    premier = donnees["resultats"][0]
    assert premier["type"] == "texte"
    assert premier["pdf"]
    assert "page" in premier
    assert "similarite" in premier
    assert premier["extrait"]


def test_retriever_tool_image(index_pret):
    reinitialiser_budget(top_k=2)
    image = next(chemin for chemin in sorted(IMAGE_DIR.iterdir()) if chemin.suffix.lower() == ".png")
    brut = retriever_tool.invoke(
        {
            "query": "figure proche de cette image",
            "content_type": "image",
            "top_k": 2,
            "image_path": str(image),
        }
    )
    donnees = json.loads(brut)
    assert donnees["mode"] == "image → image"
    assert donnees["resultats"]
    assert donnees["resultats"][0]["type"] == "image"
    assert donnees["resultats"][0]["chemin_image"]


def test_agent_react_avec_modele_scripte(index_pret):
    class ModeleScripte:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            if not getattr(self, "appele", False):
                self.appele = True
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retriever_tool",
                            "args": {
                                "query": "multi-head attention",
                                "content_type": "texte",
                                "top_k": 2,
                                "image_path": "",
                            },
                            "id": "appel-test",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content="Le Transformer utilise plusieurs têtes d'attention [attention is all you need.pdf, p. 4]."
            )

    agent = construire_agent(model=ModeleScripte(), checkpointer=None)
    resultat = interroger(
        "Explique l'attention multi-têtes.",
        top_k=2,
        agent=agent,
    )
    assert resultat["reponse"]
    assert resultat["appels"]
    assert resultat["appels"][0]["nom"] == "retriever_tool"
    assert resultat["appels"][0]["observation"]
    assert resultat["sources"]
    assert resultat["sources"][0]["page"]


def test_agent_llm_reel_si_disponible():
    etat = etat_llm()
    if not etat.get("ok") or etat.get("modele_present") is False:
        pytest.skip(f"LLM non exécutable dans cet environnement : {etat.get('detail')}")
    resultat = interroger("What is the Transformer architecture?", top_k=3, thread_id="test-reel")
    assert not resultat.get("erreur")
    assert resultat["reponse"]
    assert resultat["appels"]
    assert any(appel["nom"] == "retriever_tool" for appel in resultat["appels"])


def test_etat_llm_est_un_dictionnaire():
    etat = etat_llm()
    assert "ok" in etat
    assert "detail" in etat
    assert isinstance(etat["detail"], str)
    SimpleNamespace(**{cle: etat[cle] for cle in ("ok", "detail")})
