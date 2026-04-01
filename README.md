# Service de traduction EN/FR vers MG

Ce projet fournit un backend minimal pour traduire du texte depuis l'anglais (`en`) ou le francais (`fr`) vers le malgache (`mg`).

## Choix technique

- API rapide avec `FastAPI`
- Provider interchangeable pour garder le projet reutilisable
- Mode gratuit avec un modele open source local via Hugging Face
- Deploiement portable avec `Docker` et `docker compose`

Le provider configure par defaut utilise `facebook/nllb-200-distilled-600M`. Le telechargement du modele se fait au premier demarrage ou a la premiere traduction, selon la variable `LOAD_MODEL_ON_STARTUP`.

## Endpoints

- `GET /health` pour verifier l'etat du service
- `POST /translate` pour traduire un texte

Exemple de requete:

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "mg"
  }'
```

## Demarrage local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Demarrage Docker

```bash
docker compose up --build
```

## Demarrage Google Colab

Pour une machine locale faible, le chemin le plus simple est:

- executer le backend directement dans Colab
- exposer le port avec `ngrok`
- consommer ensuite l'URL publique depuis votre application

Fichiers ajoutes pour ce mode:

- `run_colab.py` pour lancer `uvicorn` et ouvrir le tunnel `ngrok`
- `requirements-colab.txt` pour installer uniquement le necessaire dans Colab
- `notebooks/colab_backend_launcher.ipynb` comme notebook de depart
- `.env.colab.example` comme exemple de variables d'environnement

Workflow recommande avec GitHub:

- mettre ce projet dans un depot GitHub
- dans Colab, utiliser le notebook qui fait automatiquement `git clone` au premier lancement
- apres une modification du code, faire seulement `git push`
- dans Colab, relancer la cellule de synchronisation pour faire `git pull`

Workflow minimal dans Colab:

```bash
pip install -r requirements-colab.txt
export NGROK_AUTHTOKEN="votre_token"
export ENABLE_NGROK=true
export MODEL_CACHE_DIR=/content/hf_models
python run_colab.py
```

Notes importantes pour Colab:

- ne pas utiliser Docker pour ce cas, lancez plutot `python run_colab.py`
- le modele sera telecharge au premier appel, donc le premier demarrage peut etre long
- si la session Colab se ferme, le processus backend s'arrete aussi
- pour garder le cache du modele entre sessions, vous pouvez monter Google Drive et pointer `MODEL_CACHE_DIR` vers Drive
- le notebook d'exemple peut maintenant cloner ou mettre a jour le depot GitHub automatiquement
- si Colab affiche un warning de conflits `pip`, relancez la cellule d'installation apres synchronisation du depot pour recuperer les versions corrigees
- le projet utilise maintenant le SDK Python officiel `ngrok` dans Colab, au lieu de `pyngrok`, pour eviter les echecs de telechargement du binaire

## GitHub

Le flux le plus confortable pour vous est:

1. versionner le projet dans GitHub
2. pousser vos changements depuis votre machine
3. relancer dans Colab uniquement la cellule qui synchronise le depot
4. redemarrer le backend si vous avez modifie le code Python en cours d'execution

Commandes minimales cote local:

```bash
git init
git add .
git commit -m "Initial backend version"
git branch -M main
git remote add origin https://github.com/VOTRE_USER/VOTRE_REPO.git
git push -u origin main
```

Le fichier `.gitignore` exclut deja `.env`, les caches Python et les fichiers locaux a ne pas pousser.

## Banc De Test Qualite

Un petit banc de test est disponible pour evaluer la qualite des traductions sur une liste stable de cas.

Fichiers:

- `eval/cases.json` contient les phrases de reference
- `eval/run_quality_benchmark.py` appelle `/translate` et produit un rapport JSON
- `eval/reports/latest_report.json` est le chemin de sortie par defaut

Exemple contre un backend local:

```bash
python3 eval/run_quality_benchmark.py --base-url http://127.0.0.1:8000
```

Exemple contre le backend Colab expose par ngrok:

```bash
python3 eval/run_quality_benchmark.py --base-url https://votre-url-ngrok
```

Ce script ne remplace pas une validation humaine, mais il aide a reperer rapidement:

- les traductions vides ou inchangees
- les sorties anormalement courtes
- les pertes de phrases sur les cas longs
- certains termes ou concepts medicaux critiques qui disparaissent

Le rapport JSON facilite ensuite la comparaison entre plusieurs modeles ou plusieurs versions du backend.

## Limites actuelles

- Le service cible uniquement `en/fr -> mg`
- La detection automatique entre `en` et `fr` repose sur une heuristique simple
- Le premier lancement peut etre lent a cause du telechargement du modele
- La qualite depend du modele open source choisi

## Evolution facile

- Ajouter d'autres providers dans `app/services/providers/`
- Brancher une file de taches si vous voulez de gros volumes
- Ajouter une authentification si plusieurs applications consomment l'API
