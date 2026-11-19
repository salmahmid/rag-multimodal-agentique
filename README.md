# RAG Multimodal Agentique — Attention Is All You Need

Application Python du TP de Pr. Mohamed YOUSSFI. Elle extrait l'article *Attention Is All You Need*, encode le texte et les figures avec OpenCLIP, les indexe dans ChromaDB, puis répond à des questions en français à partir des passages retrouvés. Un agent ReAct, écrit avec LangGraph, décide quand interroger l'index.

## Objectifs

- Extraire le texte, les images intégrées et les figures vectorielles du PDF.
- Placer les textes et les images dans le même espace latent CLIP.
- Rechercher en texte vers texte, texte vers image, image vers texte et image vers image.
- Produire une réponse ancrée dans les pages du document, sans inventer de contenu.
- Laisser un agent choisir, reformuler et répéter les recherches avant la réponse finale.
- Offrir une interface Streamlit pour interroger le système.

## Architecture

Le schéma du cours (`docs/tableau.jpg`) se retrouve dans les modules :

```text
PDF
  → PyMuPDF                         src/pdf_extractor.py
  → textes, images, texte complet   resultats_pdf/
  → découpage 60 tokens             src/embeddings.py
  → CLIP texte et CLIP image        même espace de 512 dimensions
  → ChromaDB, similarité cosinus    src/vector_store.py  chroma_pdf/
  → recherche multimodale           src/retriever.py
  → RAG : question + contexte + LLM src/rag.py
  → agent ReAct                     src/agent.py
       MIND   : le LLM décide et rédige
       BODY   : l'outil retriever_tool
       MEMORY : historique de conversation
  → interface                       app.py
```

L'agent suit la boucle **raisonnement → action → observation → réponse finale**. Le graphe LangGraph a deux nœuds : `raisonner` appelle le LLM avec les outils, `agir` exécute `retriever_tool`, puis le contrôle revient au LLM. Les appels d'outils et leurs observations sont affichés. Le raisonnement interne du modèle n'est pas exposé : seule la réponse finale est présentée comme réponse.

## Technologies

| Élément | Choix retenu |
|---|---|
| Langage | Python 3.12.10. Python 3.11 n'est pas installé sur la machine de développement. Les wheels testés sont ceux de CPython 3.12. |
| PDF | PyMuPDF 2.1 |
| Images | Pillow |
| Embeddings | PyTorch, OpenCLIP `ViT-B-32`, poids `laion2b_s34b_b79k` |
| Base vectorielle | ChromaDB 1.5.9, collection `pdf_clip_vit_b32`, distance cosinus |
| Agent | LangChain 1.5 et LangGraph 1.2 |
| LLM | Ollama, modèle par défaut `qwen2.5:3b`, configurable par variables d'environnement |
| Interface | Streamlit |
| Tests | Pytest |

Le calcul CLIP utilise CUDA s'il est disponible, sinon le CPU. Cette machine n'a pas de GPU CUDA : l'indexation et les tests ont tourné sur CPU.

## Les sept étapes

### 1. Extraction multimodale

`src/pdf_extractor.py` lit chaque page avec `page.get_text("text", sort=True)`, écrit `resultats_pdf/textes/page_001.txt`, etc., puis `resultats_pdf/texte_complet.txt`.

Les images intégrées sont enregistrées sous la forme `page_003_image_94.png`. Le numéro de page et la provenance (image intégrée ou rendu vectoriel) sont conservés dans `resultats_pdf/manifest.json`.

Les tracés vectoriels trop fins pour être des images, comme les filets d'un tableau, sont élargis puis regroupés. Si le groupe forme un bloc assez grand, la zone est rendue en PNG. Sur cet article, cela produit le tableau 3 de la page 9. Les figures 1 et 2 sont déjà des images intégrées : l'architecture du Transformer, l'attention multi-têtes et l'attention par produit scalaire.

Résultat mesuré sur le PDF fourni :

- 11 pages
- 11 fichiers texte
- 3 images intégrées
- 1 figure vectorielle rendue
- 0 erreur d'extraction

### 2. Embeddings CLIP

Le modèle est `ViT-B-32` / `laion2b_s34b_b79k`. Le texte est découpé en fenêtres d'au plus 60 tokens BPE, avec un pas de 50 tokens, donc un chevauchement de 10. Cette taille reste sous la fenêtre de 77 tokens de CLIP, tokens spéciaux compris.

Les vecteurs texte et image sont normalisés L2. Leur dimension vérifiée est 512. Le produit scalaire de deux vecteurs normalisés est leur similarité cosinus.

### 3. Indexation ChromaDB

La base persistante est le dossier `chroma_pdf`. L'identifiant du document est le SHA-256 complet du fichier PDF. Pour ce PDF :

`d87d482d5ae7960e2e43d7dd6d21377e60e73e8fce1bf2a01aff7aca8a08c537`

Chaque fragment et chaque image est enregistré avec son type, sa page, son texte ou son étiquette, le chemin de l'image et l'identifiant du PDF. Une réindexation supprime d'abord les entrées de la même empreinte, donc elle ne crée pas de doublons. Si le fichier change, l'empreinte précédente du même nom de PDF est retirée avant la reconstruction.

Index mesuré :

- 171 fragments de texte
- 4 images
- 175 éléments au total

### 4. Recherche multimodale

`rechercher(texte=..., image=..., type_resultat=..., k=...)` accepte exactement une requête, textuelle ou visuelle, et un filtre optionnel `texte` ou `image`. La similarité affichée vaut `1 - distance`, la collection étant en distance cosinus.

Mesures sur cet index, premier résultat :

| Requête | Mode | Page | Similarité | Extrait ou fichier |
|---|---|---:|---:|---|
| What is the Transformer architecture? | texte → texte | 2 | 0.709 | « we propose the transformer , a model architecture » |
| Explain the multi-head attention mechanism. | texte → texte | 2 | 0.812 | « multi - head self - attention mechanism » |
| What is scaled dot-product attention? | texte → texte | 3 | 0.720 | « scaled dot - product attention » |
| Find the Transformer architecture figure. | texte → image | 4 | 0.261 | `page_004_image_104.png` |
| What is the role of positional encoding? | texte → texte | 5 | 0.754 | « positional encodings » |
| Compare the encoder and decoder. | texte → texte | 5 | 0.806 | encodeur et décodeur |
| Find figures related to attention. | texte → image | 4 | 0.237 | `page_004_image_102.png` |
| Explain the formula of scaled dot-product attention. | texte → texte | 3 | 0.768 | scaled dot-product attention |
| `page_003_image_94.png` | image → image | 3 | 1.000 | la même figure d'architecture |

La recherche image → texte à partir de la figure d'architecture renvoie des passages, avec une similarité plus basse (environ 0.33). CLIP compare un dessin technique à des phrases : le score est moins élevé que pour une recherche texte → texte, et le premier passage n'est pas toujours la légende de la figure.

### 5. RAG génératif

`src/rag.py` récupère les sources, construit un contexte avec les pages et les similarités, puis l'envoie au LLM avec la question. La consigne impose une réponse en français, appuyée seulement sur les sources, avec des citations du type `[attention is all you need.pdf, p. 3]`.

Si `LLM_SUPPORTS_VISION` n'est pas activé, les pixels ne sont pas envoyés au modèle. Les figures retrouvées sont affichées à côté de la réponse, et le texte signale que le modèle ne les a pas vues. Avec un modèle vision et `LLM_SUPPORTS_VISION=true`, les images réduites sont jointes à la requête Ollama ou à l'API compatible.

Si Ollama ne répond pas, la recherche reste disponible et l'interface affiche l'erreur de génération avec les sources déjà retrouvées. Elle ne fabrique pas de réponse à la place du modèle.

### 6. Agent ReAct

L'outil `retriever_tool` a :

- le nom `retriever_tool` ;
- une description destinée au LLM ;
- des paramètres typés : `query`, `content_type` (`texte`, `image` ou `tous`), `top_k`, `image_path` ;
- une sortie JSON avec le mode, les extraits, les pages, les similarités et les chemins d'images.

L'agent peut reformuler une recherche. Le nombre d'appels est limité à trois par demande. Une image jointe dans l'interface est transmise par un chemin contrôlé, restreint aux dossiers du projet. L'historique de la conversation est conservé par un `MemorySaver` de LangGraph, identifié par un fil de discussion.

Le notebook du cours propose aussi d'exposer le RAG par un serveur MCP. Cette séparation n'est pas implémentée : l'outil est appelé dans le même processus que l'agent. MCP reste une évolution possible.

### 7. Interface Streamlit

Titre affiché : **RAG Multimodal Agentique — Attention Is All You Need**.

La barre latérale indique la présence du PDF, l'état de ChromaDB et celui du LLM. Elle contient le réglage Top-K, le choix entre l'agent et le RAG direct, le bouton d'indexation et l'effacement de la conversation.

La zone principale contient le dialogue, le champ de question, le dépôt d'une image, la réponse, les sources, les pages, les scores, les images pertinentes et les traces d'outils. Les erreurs de fichier, d'index vide ou de LLM injoignable sont affichées explicitement.

L'interface a été exécutée avec le testeur Streamlit et par HTTP sur `http://127.0.0.1:8501` (réponse 200). Une question en mode RAG direct a affiché l'erreur Ollama, tant que le modèle n'était pas joignable, et le tableau des sources : pages 3, 4 et 2 pour « What is scaled dot-product attention? », similarités 0.720, 0.696, 0.694, 0.673 et 0.652. Aucune capture d'écran du navigateur n'a été enregistrée.

## Structure du projet

```text
rag-multimodal-agentique/
├── app.py
├── requirements.txt
├── README.md
├── pytest.ini
├── .env.example
├── pdfs/attention is all you need.pdf
├── notebooks/rag_multimodal_agentique.ipynb
├── docs/RAG Multi Modal Agentique.pdf
├── docs/tableau.jpg
├── src/
│   ├── config.py
│   ├── pdf_extractor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   ├── rag.py
│   └── agent.py
├── scripts/index_pdf.py
└── tests/
```

`resultats_pdf/`, `chroma_pdf/`, `.venv/` et `.env` sont produits localement et exclus de Git.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Le premier encodage télécharge les poids CLIP depuis Hugging Face, environ 600 Mo, dans le cache de l'utilisateur.

## Configuration du LLM

Copier l'exemple puis l'adapter :

```bash
cp .env.example .env
```

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
LLM_MODEL=qwen2.5:3b
LLM_SUPPORTS_VISION=false
```

Démarrer Ollama et installer le modèle :

```bash
ollama serve
ollama pull qwen2.5:3b
```

`qwen2.5:3b` accepte les appels d'outils et reste utilisable sur CPU. Un modèle plus grand, par exemple `qwen2.5:7b`, se configure avec les mêmes variables. Pour un modèle vision, choisir un modèle qui lit les images et mettre `LLM_SUPPORTS_VISION=true`.

`LLM_PROVIDER=openai` ou `compatible` envoie la génération RAG vers `OPENAI_BASE_URL` avec `OPENAI_API_KEY`. L'agent, dans ce mode, demande le paquet `langchain-openai`, qui n'est pas dans `requirements.txt`. Aucune clé n'est stockée dans le dépôt.

## Indexer le PDF

```bash
python scripts/index_pdf.py
```

Le script extrait le PDF, calcule les embeddings et affiche le nombre de pages, d'images et d'éléments indexés. Le même bouton existe dans la barre latérale de Streamlit.

## Lancer l'interface

Depuis la racine du projet :

```bash
python -m streamlit run app.py
```

## Exemples de requêtes

Textuelles, de préférence en anglais pour CLIP, la réponse finale étant demandée en français :

- What is the Transformer architecture?
- Explain the multi-head attention mechanism.
- What is scaled dot-product attention?
- What is the role of positional encoding?
- Compare the encoder and decoder.
- Explain the formula of scaled dot-product attention.

Visuelles :

- Find the Transformer architecture figure.
- Find figures related to attention.
- Déposer `resultats_pdf/images/page_003_image_94.png` pour retrouver la figure et les passages proches.
- Déposer `resultats_pdf/images/page_004_image_102.png` pour l'attention multi-têtes.

En mode agent, une question qui compare deux mécanismes, par exemple l'attention multi-têtes et l'encodage positionnel, peut produire deux recherches successives.

## Résultats des tests

Commande :

```bash
python -m pytest tests -q -rs
```

Avant l'installation du modèle, la série donnait **12 passed, 1 skipped** en 7,64 s. Le test ignoré était `test_agent_llm_reel_si_disponible`, parce qu'Ollama refusait la connexion sur `http://127.0.0.1:11434`.

Ollama 0.40.16 a ensuite été installé et le modèle `qwen2.5:3b` a été chargé. Le test d'agent réel a alors réussi en 11,20 s : le modèle a appelé `retriever_tool` puis a produit une réponse. La série complète relancée ensuite affiche 13 points et le code de sortie 0 : les 13 tests passent.

Exemple réellement produit par l'agent pour « What is the Transformer architecture? » : un appel à `retriever_tool` avec `content_type=texte`, puis une réponse en français qui cite `[attention is all you need.pdf, p. 2]` et décrit une architecture fondée sur l'attention, sans récurrence. Le RAG direct, sur l'attention multi-têtes, a utilisé les pages 2, 8 et 5 (similarités 0,812, 0,729, 0,723 et 0,691). La rédaction de `qwen2.5:3b` reste imparfaite : elle peut mélanger des passages voisins, par exemple un écart de BLEU avec l'explication des têtes d'attention. Les sources affichées servent à contrôler ce point.

Tests couverts :

- extraction du texte, des 11 pages et des images ;
- découpage à 60 tokens avec chevauchement de 10 ;
- dimension 512 et normalisation L2 des embeddings texte et image ;
- identifiant SHA-256 ;
- réindexation sans augmentation du nombre d'éléments ;
- suppression d'une ancienne empreinte quand le document change ;
- recherches texte → texte, texte → image, image → texte et image → image ;
- les huit questions documentaires du TP, au niveau de la recherche ;
- `retriever_tool` sur un texte et sur une image extraite du PDF ;
- graphe ReAct avec un modèle scripté ;
- graphe ReAct avec Ollama `qwen2.5:3b`.

## Limitations

- CLIP est plus fiable sur des requêtes anglaises que françaises. Les scores texte → image restent modestes, autour de 0,2 à 0,3 : une requête « architecture » peut classer un schéma d'attention avant la figure 1.
- Le découpage suit les tokens, pas les phrases. Le texte décodé peut modifier certains espaces.
- Les tableaux 1 et 2 sont surtout des filets espacés. Leur texte est indexé, mais leur rendu vectoriel n'a pas formé un bloc assez dense. Seul le tableau 3 est rendu en image.
- Sans modèle vision, le LLM ne décrit pas les pixels. Il s'appuie sur les passages textuels.
- `qwen2.5:3b` répond en français et cite une page, mais il peut fusionner des passages distincts. Un modèle plus grand améliore la rédaction. Les sources restent affichées pour vérification.
- L'agent et le retriever tournent dans le même processus. Il n'y a pas de serveur MCP.
- Les poids CLIP et le modèle Ollama ne sont pas versionnés.

## Améliorations possibles

- Ajouter un seuil minimal de similarité avant d'envoyer un passage au LLM.
- Rendre aussi les tableaux 1 et 2 lorsque leurs filets sont trop espacés.
- Séparer l'agent et le retriever par MCP, comme le propose la suite du notebook.
- Évaluer les citations de la réponse générée de façon automatique.

## Droits des documents

L'article *Attention Is All You Need* (Vaswani et al.) est joint pour que le projet s'exécute. Les droits restent ceux des auteurs. Le support `docs/RAG Multi Modal Agentique.pdf`, le notebook et le schéma sont le cours de Pr. Mohamed YOUSSFI, ENSET Mohammedia. Ils sont présents dans ce dépôt d'étudiante pour l'évaluation du TP et ne sont pas une republication indépendante du cours.

## Auteure

Salma — GitHub [`salmahmid`](https://github.com/salmahmid) — salmahmid2023@gmail.com
